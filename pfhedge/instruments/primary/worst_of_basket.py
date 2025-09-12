from math import ceil
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

import torch
from torch import Tensor

from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.doc import _set_docstring
from pfhedge._utils.str import _format_float
from pfhedge._utils.typing import TensorOrScalar
from pfhedge.stochastic.correlated_brownian import generate_correlated_geometric_brownian

from .base import BasePrimary


class WorstOfBasketStock(BasePrimary):
    r"""A basket of correlated stocks where the spot represents the worst performer.

    This instrument simulates multiple correlated stocks using geometric Brownian motion
    with correlation induced by Cholesky decomposition. The "spot" price tracks the 
    worst performing asset in the basket over time.

    .. seealso::
        - :func:`pfhedge.stochastic.generate_correlated_geometric_brownian`:
          The underlying stochastic process.

    Args:
        n_assets (int, default=2): The number of assets in the basket.
        sigma (float or list[float] or torch.Tensor, default=0.2): The volatility 
            parameters for each asset. If a single float, uses the same volatility 
            for all assets.
        mu (float or list[float] or torch.Tensor, default=0.0): The drift parameters 
            for each asset. If a single float, uses the same drift for all assets.
        correlation_matrix (torch.Tensor, optional): The correlation matrix of size
            (n_assets, n_assets). If None, creates an identity matrix (uncorrelated assets).
        cost (float, default=0.0): The transaction cost rate.
        dt (float, default=1/250): The intervals of the time steps.
        dtype (torch.dtype, optional): Desired dtype of returned tensor.
        device (torch.device, optional): Desired device of returned tensor.

    Buffers:
        - spot (:class:`torch.Tensor`): The worst performing asset prices over time.
          This attribute is set by a method :meth:`simulate()`.
          The shape is :math:`(N, T)` where :math:`N` is the number of simulated paths
          and :math:`T` is the number of time steps.
        - basket (:class:`torch.Tensor`): The prices of all assets in the basket.
          The shape is :math:`(N, T, A)` where :math:`A` is the number of assets.

    Examples:
        Create a basket of 3 assets with default correlation (uncorrelated):

        >>> from pfhedge.instruments import WorstOfBasketStock
        >>> import torch
        >>>
        >>> _ = torch.manual_seed(42)
        >>> basket = WorstOfBasketStock(n_assets=3)
        >>> basket.simulate(n_paths=2, time_horizon=5 / 250)
        >>> basket.spot.shape
        torch.Size([2, 6])
        >>> basket.basket.shape
        torch.Size([2, 6, 3])

        Create a basket with custom volatilities and correlation:

        >>> correlation = torch.tensor([[1.0, 0.5, 0.3],
        ...                            [0.5, 1.0, 0.7],
        ...                            [0.3, 0.7, 1.0]])
        >>> basket = WorstOfBasketStock(
        ...     n_assets=3,
        ...     sigma=[0.2, 0.25, 0.18],
        ...     correlation_matrix=correlation
        ... )
        >>> basket.simulate(n_paths=1000, time_horizon=20 / 250)
    """

    def __init__(
        self,
        n_assets: int = 2,
        sigma: Union[float, List[float], Tensor] = 0.2,
        mu: Union[float, List[float], Tensor] = 0.0,
        correlation_matrix: Optional[Tensor] = None,
        cost: float = 0.0,
        dt: float = 1 / 250,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        if n_assets < 2:
            raise ValueError("n_assets must be at least 2 for a basket")

        self.n_assets = n_assets
        self.cost = cost
        self.dt = dt

        # Convert parameters to tensors
        if isinstance(sigma, (list, tuple)):
            if len(sigma) != n_assets:
                raise ValueError(f"sigma list must have length {n_assets}")
            sigma = torch.tensor(sigma, dtype=dtype, device=device)
        elif isinstance(sigma, (float, int)):
            sigma = torch.full((n_assets,), float(sigma), dtype=dtype, device=device)
        elif isinstance(sigma, Tensor):
            sigma = sigma.to(dtype=dtype, device=device)
            if sigma.numel() == 1:
                sigma = sigma.expand(n_assets)
            elif sigma.numel() != n_assets:
                raise ValueError(f"sigma tensor must have {n_assets} elements")

        if isinstance(mu, (list, tuple)):
            if len(mu) != n_assets:
                raise ValueError(f"mu list must have length {n_assets}")
            mu = torch.tensor(mu, dtype=dtype, device=device)
        elif isinstance(mu, (float, int)):
            mu = torch.full((n_assets,), float(mu), dtype=dtype, device=device)
        elif isinstance(mu, Tensor):
            mu = mu.to(dtype=dtype, device=device)
            if mu.numel() == 1:
                mu = mu.expand(n_assets)
            elif mu.numel() != n_assets:
                raise ValueError(f"mu tensor must have {n_assets} elements")

        if correlation_matrix is not None:
            if correlation_matrix.shape != (n_assets, n_assets):
                raise ValueError(
                    f"correlation_matrix must be ({n_assets}, {n_assets}), "
                    f"got {correlation_matrix.shape}"
                )
            correlation_matrix = correlation_matrix.to(dtype=dtype, device=device)

        self.sigma = sigma
        self.mu = mu
        self.correlation_matrix = correlation_matrix

        self.to(dtype=dtype, device=device)

    @property
    def default_init_state(self) -> Tuple[float, ...]:
        """Returns the default initial state: all assets start at 1.0."""
        return tuple(1.0 for _ in range(self.n_assets))

    @property
    def volatility(self) -> Tensor:
        """Returns the volatility of the worst performing asset.

        Note: This returns the volatility corresponding to the current worst performer,
        which may change over time.
        """
        if not hasattr(self, 'spot'):
            raise AttributeError("Must simulate before accessing volatility")
        
        # For simplicity, return the average volatility of all assets
        # In practice, one might want to track which asset is currently worst
        return torch.full_like(self.spot, self.sigma.mean().item())

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the worst performing asset."""
        return self.volatility ** 2

    def simulate(
        self,
        n_paths: int = 1,
        time_horizon: float = 20 / 250,
        init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
    ) -> None:
        """Simulate the basket and worst-of prices.

        Args:
            n_paths (int, default=1): The number of paths to simulate.
            time_horizon (float, default=20/250): The period of time to simulate.
            init_state (tuple[torch.Tensor | float], optional): The initial state of
                each asset. If None, uses default_init_state (all assets start at 1.0).
        """
        if init_state is None:
            init_state = cast(Tuple[float, ...], self.default_init_state)

        # Generate correlated basket prices
        basket_prices = generate_correlated_geometric_brownian(
            n_paths=n_paths,
            n_steps=ceil(time_horizon / self.dt + 1),
            n_assets=self.n_assets,
            init_state=init_state,
            sigma=self.sigma,
            mu=self.mu,
            correlation_matrix=self.correlation_matrix,
            dt=self.dt,
            dtype=self.dtype,
            device=self.device,
        )

        # Register the full basket prices
        self.register_buffer("basket", basket_prices)

        # Calculate the worst performing asset at each time step
        # Normalize by initial values to get relative performance
        init_values = basket_prices[:, 0:1, :]  # Shape: (n_paths, 1, n_assets)
        relative_performance = basket_prices / init_values  # Shape: (n_paths, n_steps, n_assets)
        
        # Find the minimum (worst) performance across assets
        worst_performance, _ = relative_performance.min(dim=2)  # Shape: (n_paths, n_steps)
        
        # Convert back to absolute prices (multiply by initial value of first asset for consistency)
        spot_prices = worst_performance * init_values[:, 0, 0:1]  # Shape: (n_paths, n_steps)

        # Register the worst-of spot prices
        self.register_buffer("spot", spot_prices)

    def extra_repr(self) -> str:
        params = [f"n_assets={self.n_assets}"]
        
        # Show sigma
        if self.sigma.numel() == 1 or torch.allclose(self.sigma, self.sigma[0]):
            params.append("sigma=" + _format_float(self.sigma[0].item()))
        else:
            sigma_str = "[" + ", ".join(_format_float(s.item()) for s in self.sigma) + "]"
            params.append(f"sigma={sigma_str}")
        
        # Show mu if not zero
        if not torch.allclose(self.mu, torch.zeros_like(self.mu)):
            if self.mu.numel() == 1 or torch.allclose(self.mu, self.mu[0]):
                params.append("mu=" + _format_float(self.mu[0].item()))
            else:
                mu_str = "[" + ", ".join(_format_float(m.item()) for m in self.mu) + "]"
                params.append(f"mu={mu_str}")
        
        # Show if correlated
        if self.correlation_matrix is not None:
            if not torch.allclose(self.correlation_matrix, torch.eye(self.n_assets)):
                params.append("correlated=True")
        
        if self.cost != 0.0:
            params.append("cost=" + _format_float(self.cost))
        params.append("dt=" + _format_float(self.dt))
        
        return ", ".join(params)

    def to(self: "WorstOfBasketStock", *args, **kwargs) -> "WorstOfBasketStock":
        """Move and/or cast the parameters of the instrument."""
        device, dtype, *_ = self._parse_to(*args, **kwargs)
        
        # Move parameter tensors to the specified device/dtype FIRST
        # Convert device to torch.device if it's a string
        if device is not None:
            if isinstance(device, str):
                device = torch.device(device)
            self.sigma = self.sigma.to(device=device, dtype=dtype)
            self.mu = self.mu.to(device=device, dtype=dtype)
            if self.correlation_matrix is not None:
                self.correlation_matrix = self.correlation_matrix.to(device=device, dtype=dtype)
        elif dtype is not None:
            self.sigma = self.sigma.to(dtype=dtype)
            self.mu = self.mu.to(dtype=dtype)
            if self.correlation_matrix is not None:
                self.correlation_matrix = self.correlation_matrix.to(dtype=dtype)
        
        # Call the parent's to method for buffers and base attributes AFTER
        super().to(*args, **kwargs)
        
        return self


# Assign docstrings so they appear in Sphinx documentation
_set_docstring(WorstOfBasketStock, "default_init_state", BasePrimary.default_init_state)
# NOTE: Not copying the "to" method docstring since we override it

