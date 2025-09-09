from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import Tensor

from pfhedge._utils.typing import TensorOrScalar

from ._utils import cast_state


def generate_correlated_geometric_brownian(
    n_paths: int,
    n_steps: int,
    n_assets: int,
    init_state: Union[Tuple[TensorOrScalar, ...], TensorOrScalar] = None,
    sigma: Union[float, Tensor] = 0.2,
    mu: Union[float, Tensor] = 0.0,
    correlation_matrix: Optional[Tensor] = None,
    dt: float = 1 / 250,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    engine: Callable[..., Tensor] = torch.randn,
) -> Tensor:
    r"""Returns time series of correlated geometric Brownian motions using Cholesky decomposition.

    The time evolution of each process is given by:

    .. math::

        dS_i(t) = \mu_i S_i(t) dt + \sigma_i S_i(t) dW_i(t)

    where the Brownian motions :math:`dW_i(t)` are correlated according to the
    provided correlation matrix using Cholesky decomposition.

    Args:
        n_paths (int): The number of simulated paths.
        n_steps (int): The number of time steps.
        n_assets (int): The number of assets in the basket.
        init_state (tuple[torch.Tensor | float] or float, optional): The initial state of
            the time series. Can be:
            - A tuple of length n_assets: :math:`(S_1(0), S_2(0), ..., S_n(0))`
            - A single float (used for all assets)
            - None (defaults to 1.0 for all assets)
        sigma (float or torch.Tensor, default=0.2): The volatility parameters.
            Can be a single float (used for all assets) or a tensor of length n_assets.
        mu (float or torch.Tensor, default=0.0): The drift parameters.
            Can be a single float (used for all assets) or a tensor of length n_assets.
        correlation_matrix (torch.Tensor, optional): The correlation matrix of size
            (n_assets, n_assets). If None, defaults to identity matrix (uncorrelated). Must be positive definite.
        dt (float, default=1/250): The intervals of the time steps.
        dtype (torch.dtype, optional): The desired data type of returned tensor.
        device (torch.device, optional): The desired device of returned tensor.
        engine (callable, default=torch.randn): The desired generator of random numbers
            from a standard normal distribution.

    Shape:
        - Output: :math:`(N, T, A)` where
          :math:`N` is the number of paths,
          :math:`T` is the number of time steps, and
          :math:`A` is the number of assets.

    Returns:
        torch.Tensor

    Examples:
        Generate two correlated assets with correlation 0.5:

        >>> import torch
        >>> from pfhedge.stochastic import generate_correlated_geometric_brownian
        >>>
        >>> _ = torch.manual_seed(42)
        >>> correlation_matrix = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
        >>> paths = generate_correlated_geometric_brownian(
        ...     n_paths=2, n_steps=5, n_assets=2, correlation_matrix=correlation_matrix
        ... )
        >>> paths.shape
        torch.Size([2, 5, 2])
    """
    # Handle default init_state
    if init_state is None:
        init_state = tuple(1.0 for _ in range(n_assets))
    elif isinstance(init_state, (float, int)):
        init_state = tuple(float(init_state) for _ in range(n_assets))
    elif len(init_state) == 1:
        init_state = tuple(float(init_state[0]) for _ in range(n_assets))
    elif len(init_state) != n_assets:
        raise ValueError(f"init_state must have length {n_assets}, got {len(init_state)}")

    init_state = cast_state(init_state, dtype=dtype, device=device)

    # Handle parameters
    if isinstance(sigma, (float, int)):
        sigma = torch.full((n_assets,), float(sigma), dtype=dtype, device=device)
    elif isinstance(sigma, Tensor):
        sigma = sigma.to(dtype=dtype, device=device)
        if sigma.numel() == 1:
            sigma = sigma.expand(n_assets)
        elif sigma.numel() != n_assets:
            raise ValueError(f"sigma must have {n_assets} elements, got {sigma.numel()}")

    if isinstance(mu, (float, int)):
        mu = torch.full((n_assets,), float(mu), dtype=dtype, device=device)
    elif isinstance(mu, Tensor):
        mu = mu.to(dtype=dtype, device=device)
        if mu.numel() == 1:
            mu = mu.expand(n_assets)
        elif mu.numel() != n_assets:
            raise ValueError(f"mu must have {n_assets} elements, got {mu.numel()}")

    # Handle correlation matrix
    if correlation_matrix is None:
        correlation_matrix = torch.eye(n_assets, dtype=dtype, device=device)
    else:
        correlation_matrix = correlation_matrix.to(dtype=dtype, device=device)
        if correlation_matrix.shape != (n_assets, n_assets):
            raise ValueError(
                f"correlation_matrix must be ({n_assets}, {n_assets}), "
                f"got {correlation_matrix.shape}"
            )

    # Cholesky decomposition for correlation
    try:
        chol = torch.linalg.cholesky(correlation_matrix)
    except RuntimeError as e:
        raise ValueError(
            "correlation_matrix must be positive definite for Cholesky decomposition"
        ) from e

    # Generate independent standard normal random numbers
    randn = engine(n_paths, n_steps, n_assets, dtype=dtype, device=device)
    randn[:, 0, :] = 0.0

    # Apply correlation via Cholesky decomposition
    # Shape: (n_paths, n_steps, n_assets) @ (n_assets, n_assets) -> (n_paths, n_steps, n_assets)
    correlated_randn = randn @ chol.T

    # Generate geometric Brownian motion for each asset
    result = torch.zeros(n_paths, n_steps, n_assets, dtype=dtype, device=device)

    for i in range(n_assets):
        # Time vector
        t = dt * torch.arange(n_steps, dtype=dtype, device=device).unsqueeze(0)
        
        # Cumulative sum of correlated random numbers for asset i
        brown = torch.sqrt(torch.tensor(dt, dtype=dtype, device=device)) * correlated_randn[:, :, i].cumsum(1)
        
        # Drift component
        drift = mu[i] * t
        
        # Brownian motion component
        brownian_component = drift + sigma[i] * brown - (sigma[i] ** 2) * t / 2
        
        # Geometric Brownian motion
        result[:, :, i] = init_state[i] * brownian_component.exp()

    return result
