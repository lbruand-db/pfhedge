from typing import List
from typing import Optional
from typing import Union

import torch
from torch import Tensor

from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.doc import _set_docstring
from pfhedge._utils.str import _format_float

from ..primary.base import BasePrimary
from .base import BaseDerivative


class WorstOfBasketAutocall(BaseDerivative):
    r"""Worst-of basket autocall note.

    An autocall note that can redeem early on observation dates if the worst 
    performing asset in the basket is at or above the autocall barrier. 
    If it autocalls, it pays a fixed coupon. If it doesn't autocall by maturity,
    the final payoff depends on the final performance of the worst performer.

    The autocall condition is checked on observation dates. If the worst performer 
    is at or above the autocall barrier on any observation date, the note autocalls
    and pays::

        coupon_rate * notional * (observation_date / maturity)

    If no autocall occurs, the final payoff is::

        - If worst performer >= protection_barrier: notional
        - If worst performer < protection_barrier: 
          notional * (final_worst_performance / initial_worst_performance)

    Args:
        underlier (:class:`BasePrimary`): The underlying basket (typically WorstOfBasketStock).
        autocall_barrier (float, default=1.0): The barrier level for autocall (as fraction of initial).
        protection_barrier (float, default=0.65): The protection barrier for final payoff.
        coupon_rate (float, default=0.08): Annual coupon rate paid on autocall.
        observation_dates (List[float], optional): Observation dates as fractions of maturity.
            If None, defaults to quarterly observations [0.25, 0.5, 0.75, 1.0].
        notional (float, default=1.0): The notional amount.
        maturity (float, default=1.0): The maturity of the note in years.

    Attributes:
        dtype (torch.dtype): The dtype with which the simulated time-series are
            represented.
        device (torch.device): The device where the simulated time-series are.

    Examples:
        Create an autocall on a 3-asset worst-of basket:

        >>> import torch
        >>> from pfhedge.instruments import WorstOfBasketStock
        >>> from pfhedge.instruments.derivative.autocall import WorstOfBasketAutocall
        >>>
        >>> # Create correlation matrix
        >>> correlation = torch.tensor([[1.0, 0.5, 0.3],
        ...                            [0.5, 1.0, 0.7], 
        ...                            [0.3, 0.7, 1.0]])
        >>> 
        >>> # Create basket with different volatilities
        >>> basket = WorstOfBasketStock(
        ...     n_assets=3,
        ...     sigma=[0.2, 0.25, 0.18],
        ...     correlation_matrix=correlation,
        ...     dt=1/250
        ... )
        >>>
        >>> # Create autocall note
        >>> autocall = WorstOfBasketAutocall(
        ...     underlier=basket,
        ...     autocall_barrier=1.0,
        ...     protection_barrier=0.65,
        ...     coupon_rate=0.08,
        ...     maturity=1.0
        ... )
        >>> 
        >>> _ = torch.manual_seed(42)
        >>> autocall.simulate(n_paths=1000)
        >>> payoffs = autocall.payoff()
        >>> payoffs.shape
        torch.Size([1000])

        Custom observation dates (semi-annual):

        >>> autocall_custom = WorstOfBasketAutocall(
        ...     underlier=basket,
        ...     observation_dates=[0.5, 1.0],
        ...     coupon_rate=0.06,
        ...     maturity=1.0
        ... )
    """

    def __init__(
        self,
        underlier: BasePrimary,
        autocall_barrier: float = 1.0,
        protection_barrier: float = 0.65,
        coupon_rate: float = 0.08,
        observation_dates: Optional[List[float]] = None,
        notional: float = 1.0,
        maturity: float = 1.0,
    ) -> None:
        super().__init__()
        self.register_underlier("underlier", underlier)
        
        self.autocall_barrier = autocall_barrier
        self.protection_barrier = protection_barrier
        self.coupon_rate = coupon_rate
        self.notional = notional
        self.maturity = maturity
        
        # Default to quarterly observations if not specified
        if observation_dates is None:
            self.observation_dates = [0.25, 0.5, 0.75, 1.0]
        else:
            self.observation_dates = sorted(observation_dates)
            # Ensure maturity is included
            if self.observation_dates[-1] != 1.0:
                self.observation_dates.append(1.0)

    def extra_repr(self) -> str:
        params = []
        params.append("autocall_barrier=" + _format_float(self.autocall_barrier))
        params.append("protection_barrier=" + _format_float(self.protection_barrier))
        params.append("coupon_rate=" + _format_float(self.coupon_rate))
        params.append("notional=" + _format_float(self.notional))
        params.append("maturity=" + _format_float(self.maturity))
        params.append("observation_dates=" + str(len(self.observation_dates)) + " dates")
        return ", ".join(params)

    def _get_observation_indices(self) -> List[int]:
        """Convert observation dates to time step indices."""
        dt = self.ul().dt
        time_horizon = self.maturity
        n_steps = int(time_horizon / dt) + 1
        
        indices = []
        for obs_date in self.observation_dates:
            time_from_start = obs_date * self.maturity
            index = min(int(time_from_start / dt), n_steps - 1)
            indices.append(index)
        
        return indices

    def payoff_fn(self) -> Tensor:
        """Calculate the autocall payoff."""
        # Get worst-of spot prices
        worst_spot = self.ul().spot  # Shape: (n_paths, n_steps)
        n_paths = worst_spot.shape[0]
        
        # Initial price (assuming all paths start at same level)
        initial_price = worst_spot[0, 0].item()
        
        # Convert observation dates to indices
        obs_indices = self._get_observation_indices()
        
        # Initialize payoff tensor
        payoffs = torch.zeros(n_paths, device=worst_spot.device, dtype=worst_spot.dtype)
        autocalled = torch.zeros(n_paths, device=worst_spot.device, dtype=torch.bool)
        
        # Check each observation date for autocall
        for i, obs_idx in enumerate(obs_indices[:-1]):  # Exclude final observation for now
            if obs_idx >= worst_spot.shape[1]:
                continue
                
            # Current worst performance relative to initial
            current_performance = worst_spot[:, obs_idx] / initial_price
            
            # Check autocall condition (not already autocalled and above barrier)
            autocall_condition = (~autocalled) & (current_performance >= self.autocall_barrier)
            
            # Calculate coupon payment for autocalling paths
            observation_time = self.observation_dates[i]
            coupon_payment = self.notional * (1 + self.coupon_rate * observation_time)
            
            # Update payoffs and autocalled flag
            payoffs.masked_fill_(autocall_condition, coupon_payment)
            autocalled |= autocall_condition
        
        # Final payoff for non-autocalled paths
        final_idx = obs_indices[-1] if obs_indices[-1] < worst_spot.shape[1] else -1
        final_performance = worst_spot[:, final_idx] / initial_price
        
        # Paths that haven't autocalled by final observation
        not_autocalled = ~autocalled
        
        # Check final autocall (at maturity)
        final_autocall_condition = not_autocalled & (final_performance >= self.autocall_barrier)
        final_coupon = self.notional * (1 + self.coupon_rate * 1.0)  # Full year coupon
        payoffs.masked_fill_(final_autocall_condition, final_coupon)
        
        # Update autocalled flag
        autocalled |= final_autocall_condition
        
        # For paths that never autocalled, apply protection structure
        still_not_autocalled = ~autocalled
        
        # Above protection barrier: capital protection
        protected = still_not_autocalled & (final_performance >= self.protection_barrier)
        payoffs.masked_fill_(protected, self.notional)
        
        # Below protection barrier: participation in worst performance
        at_risk = still_not_autocalled & (final_performance < self.protection_barrier)
        at_risk_payoff = self.notional * final_performance
        payoffs = torch.where(at_risk, at_risk_payoff, payoffs)
        
        return payoffs

    def get_autocall_probabilities(self) -> Tensor:
        """Calculate the probability of autocalling at each observation date.
        
        Returns:
            torch.Tensor: Probabilities of autocalling at each observation date.
                Shape: (n_observation_dates,)
        """
        # Get worst-of spot prices
        worst_spot = self.ul().spot  # Shape: (n_paths, n_steps)
        n_paths = worst_spot.shape[0]
        
        # Initial price
        initial_price = worst_spot[0, 0].item()
        
        # Convert observation dates to indices
        obs_indices = self._get_observation_indices()
        
        # Track autocall probabilities
        probs = torch.zeros(len(obs_indices), device=worst_spot.device, dtype=worst_spot.dtype)
        autocalled = torch.zeros(n_paths, device=worst_spot.device, dtype=torch.bool)
        
        for i, obs_idx in enumerate(obs_indices):
            if obs_idx >= worst_spot.shape[1]:
                continue
                
            # Current performance
            current_performance = worst_spot[:, obs_idx] / initial_price
            
            # Autocall condition for this observation
            autocall_this_period = (~autocalled) & (current_performance >= self.autocall_barrier)
            
            # Calculate probability
            probs[i] = autocall_this_period.float().mean()
            
            # Update autocalled flag
            autocalled |= autocall_this_period
        
        return probs

    def get_average_payoff_by_scenario(self) -> dict:
        """Calculate average payoffs broken down by scenario.
        
        Returns:
            dict: Dictionary with average payoffs for different scenarios.
        """
        # Get payoffs and spot prices
        payoffs = self.payoff_fn()
        worst_spot = self.ul().spot
        initial_price = worst_spot[0, 0].item()
        
        # Get observation indices  
        obs_indices = self._get_observation_indices()
        
        # Track which paths autocall and when
        n_paths = worst_spot.shape[0]
        autocalled = torch.zeros(n_paths, device=worst_spot.device, dtype=torch.bool)
        autocall_times = torch.zeros(n_paths, device=worst_spot.device, dtype=torch.long) - 1
        
        # Find autocall times
        for i, obs_idx in enumerate(obs_indices):
            if obs_idx >= worst_spot.shape[1]:
                continue
                
            current_performance = worst_spot[:, obs_idx] / initial_price
            autocall_this_period = (~autocalled) & (current_performance >= self.autocall_barrier)
            
            autocall_times.masked_fill_(autocall_this_period, i)
            autocalled |= autocall_this_period
        
        # Calculate scenario breakdown
        results = {}
        
        # Autocalled scenarios
        for i, obs_date in enumerate(self.observation_dates):
            autocalled_at_i = (autocall_times == i)
            if autocalled_at_i.any():
                avg_payoff = payoffs[autocalled_at_i].mean().item()
                prob = autocalled_at_i.float().mean().item()
                results[f"autocall_at_{obs_date:.2f}"] = {
                    "probability": prob,
                    "average_payoff": avg_payoff
                }
        
        # Never autocalled scenarios
        never_autocalled = ~autocalled
        if never_autocalled.any():
            final_performance = worst_spot[:, -1] / initial_price
            never_autocalled_payoffs = payoffs[never_autocalled]
            final_performances = final_performance[never_autocalled]
            
            # Protected scenarios
            protected = never_autocalled_payoffs == self.notional
            if protected.any():
                prob = protected.float().mean().item() * never_autocalled.float().mean().item()
                results["protected"] = {
                    "probability": prob,
                    "average_payoff": self.notional
                }
            
            # At-risk scenarios
            at_risk = never_autocalled_payoffs < self.notional
            if at_risk.any():
                prob = at_risk.float().mean().item() * never_autocalled.float().mean().item()
                avg_payoff = never_autocalled_payoffs[at_risk].mean().item()
                avg_performance = final_performances[at_risk].mean().item()
                results["at_risk"] = {
                    "probability": prob,
                    "average_payoff": avg_payoff,
                    "average_worst_performance": avg_performance
                }
        
        return results


# Assign docstrings so they appear in Sphinx documentation
_set_attr_and_docstring(WorstOfBasketAutocall, "simulate", BaseDerivative.simulate)
_set_attr_and_docstring(WorstOfBasketAutocall, "to", BaseDerivative.to)
_set_attr_and_docstring(WorstOfBasketAutocall, "ul", BaseDerivative.ul)
_set_attr_and_docstring(WorstOfBasketAutocall, "list", BaseDerivative.list)
_set_docstring(WorstOfBasketAutocall, "payoff", BaseDerivative.payoff)
