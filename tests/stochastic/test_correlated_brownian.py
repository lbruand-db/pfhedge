from math import sqrt

import pytest
import torch
from torch.testing import assert_close

from pfhedge.stochastic import generate_correlated_geometric_brownian
from tests._utils import select_most_accurate_gpu_device, get_available_dtypes


def test_generate_correlated_geometric_brownian_shape(device: str = "cpu"):
    """Test that output has correct shape."""
    n_paths = 10
    n_steps = 50
    n_assets = 3
    
    device = torch.device(device) if device else None
    output = generate_correlated_geometric_brownian(
        n_paths=n_paths, 
        n_steps=n_steps, 
        n_assets=n_assets, 
        device=device
    )
    
    assert output.size() == torch.Size((n_paths, n_steps, n_assets))
    assert output.device.type == (device.type if device else "cpu")


@pytest.mark.gpu
def test_generate_correlated_geometric_brownian_shape_gpu():
    test_generate_correlated_geometric_brownian_shape(device=select_most_accurate_gpu_device())


def test_generate_correlated_geometric_brownian_initial_values(device: str = "cpu"):
    """Test that initial values are correct."""
    n_paths = 5
    n_steps = 10
    n_assets = 2
    init_state = (100.0, 110.0)
    
    device = torch.device(device) if device else None
    output = generate_correlated_geometric_brownian(
        n_paths=n_paths,
        n_steps=n_steps, 
        n_assets=n_assets,
        init_state=init_state,
        device=device
    )
    
    # Check initial values
    assert_close(output[:, 0, 0], torch.full((n_paths,), 100.0, device=device))
    assert_close(output[:, 0, 1], torch.full((n_paths,), 110.0, device=device))


@pytest.mark.gpu
def test_generate_correlated_geometric_brownian_initial_values_gpu():
    test_generate_correlated_geometric_brownian_initial_values(device=select_most_accurate_gpu_device())


def test_generate_correlated_geometric_brownian_single_init_value(device: str = "cpu"):
    """Test with single initial value for all assets."""
    n_paths = 3
    n_steps = 5
    n_assets = 4
    init_value = 50.0
    
    device = torch.device(device) if device else None
    output = generate_correlated_geometric_brownian(
        n_paths=n_paths,
        n_steps=n_steps,
        n_assets=n_assets,
        init_state=init_value,
        device=device
    )
    
    # All assets should start at the same value
    for i in range(n_assets):
        assert_close(output[:, 0, i], torch.full((n_paths,), init_value, device=device))


@pytest.mark.gpu
def test_generate_correlated_geometric_brownian_single_init_value_gpu():
    test_generate_correlated_geometric_brownian_single_init_value(device=select_most_accurate_gpu_device())


def test_generate_correlated_geometric_brownian_default_init_state(device: str = "cpu"):
    """Test default initialization (all assets start at 1.0)."""
    n_paths = 2
    n_steps = 5
    n_assets = 3
    
    device = torch.device(device) if device else None
    output = generate_correlated_geometric_brownian(
        n_paths=n_paths,
        n_steps=n_steps,
        n_assets=n_assets,
        device=device
    )
    
    # All assets should start at 1.0
    for i in range(n_assets):
        assert_close(output[:, 0, i], torch.full((n_paths,), 1.0, device=device))


@pytest.mark.gpu
def test_generate_correlated_geometric_brownian_default_init_state_gpu():
    test_generate_correlated_geometric_brownian_default_init_state(device=select_most_accurate_gpu_device())


def test_generate_correlated_geometric_brownian_correlation_identity(device: str = "cpu"):
    """Test with identity correlation matrix (uncorrelated)."""
    torch.manual_seed(42)
    n_paths = 1000
    n_steps = 100
    n_assets = 2
    
    # Identity correlation matrix
    correlation_matrix = torch.eye(n_assets)
    if device:
        correlation_matrix = correlation_matrix.to(device)
    
    output = generate_correlated_geometric_brownian(
        n_paths=n_paths,
        n_steps=n_steps,
        n_assets=n_assets,
        correlation_matrix=correlation_matrix,
        device=device
    )
    
    # Calculate log returns
    log_returns = torch.log(output[:, 1:, :] / output[:, :-1, :])
    
    # Calculate empirical correlation
    returns_asset0 = log_returns[:, :, 0].flatten()
    returns_asset1 = log_returns[:, :, 1].flatten()
    
    empirical_corr = torch.corrcoef(torch.stack([returns_asset0, returns_asset1]))[0, 1]
    
    # Should be close to 0 for uncorrelated assets
    assert abs(empirical_corr) < 0.1


@pytest.mark.gpu  
def test_generate_correlated_geometric_brownian_correlation_identity_gpu():
    test_generate_correlated_geometric_brownian_correlation_identity(device=select_most_accurate_gpu_device())


def test_generate_correlated_geometric_brownian_high_correlation(device: str = "cpu"):
    """Test with high correlation."""
    torch.manual_seed(123)
    n_paths = 2000
    n_steps = 100
    n_assets = 2
    
    # High correlation matrix
    correlation_matrix = torch.tensor([[1.0, 0.8], [0.8, 1.0]])
    if device:
        correlation_matrix = correlation_matrix.to(device)
    
    output = generate_correlated_geometric_brownian(
        n_paths=n_paths,
        n_steps=n_steps,
        n_assets=n_assets,
        correlation_matrix=correlation_matrix,
        device=device
    )
    
    # Calculate log returns
    log_returns = torch.log(output[:, 1:, :] / output[:, :-1, :])
    
    # Calculate empirical correlation
    returns_asset0 = log_returns[:, :, 0].flatten()
    returns_asset1 = log_returns[:, :, 1].flatten()
    
    empirical_corr = torch.corrcoef(torch.stack([returns_asset0, returns_asset1]))[0, 1]
    
    # Should be close to 0.8
    assert_close(empirical_corr, torch.tensor(0.8, device=empirical_corr.device), atol=0.1, rtol=0.1)


@pytest.mark.gpu
def test_generate_correlated_geometric_brownian_high_correlation_gpu():
    test_generate_correlated_geometric_brownian_high_correlation(device=select_most_accurate_gpu_device())


def test_generate_correlated_geometric_brownian_different_parameters(device: str = "cpu"):
    """Test with different sigma and mu for each asset."""
    n_paths = 5
    n_steps = 10
    n_assets = 3
    
    sigma = torch.tensor([0.1, 0.2, 0.3])
    mu = torch.tensor([0.0, 0.05, 0.1])
    
    if device:
        sigma = sigma.to(device)
        mu = mu.to(device)
    
    output = generate_correlated_geometric_brownian(
        n_paths=n_paths,
        n_steps=n_steps,
        n_assets=n_assets,
        sigma=sigma,
        mu=mu,
        device=device
    )
    
    assert output.size() == torch.Size((n_paths, n_steps, n_assets))
    # All paths should start at 1.0
    assert_close(output[:, 0, :], torch.ones(n_paths, n_assets).to(device))


@pytest.mark.gpu
def test_generate_correlated_geometric_brownian_different_parameters_gpu():
    test_generate_correlated_geometric_brownian_different_parameters(device=select_most_accurate_gpu_device())


def test_generate_correlated_geometric_brownian_single_parameters(device: str = "cpu"):
    """Test with single sigma and mu values (applied to all assets)."""
    n_paths = 3
    n_steps = 5
    n_assets = 2
    sigma = 0.25
    mu = 0.03
    
    output = generate_correlated_geometric_brownian(
        n_paths=n_paths,
        n_steps=n_steps,
        n_assets=n_assets,
        sigma=sigma,
        mu=mu,
        device=device
    )
    
    assert output.size() == torch.Size((n_paths, n_steps, n_assets))


@pytest.mark.gpu
def test_generate_correlated_geometric_brownian_single_parameters_gpu():
    test_generate_correlated_geometric_brownian_single_parameters(device=select_most_accurate_gpu_device())


def test_generate_correlated_geometric_brownian_errors():
    """Test error handling for invalid inputs."""
    
    # Wrong init_state length
    with pytest.raises(ValueError, match="init_state must have length 3"):
        generate_correlated_geometric_brownian(
            n_paths=5, n_steps=10, n_assets=3, init_state=(1.0, 2.0)
        )
    
    # Wrong sigma length
    with pytest.raises(ValueError, match="sigma must have 2 elements"):
        generate_correlated_geometric_brownian(
            n_paths=5, n_steps=10, n_assets=2, sigma=torch.tensor([0.1, 0.2, 0.3])
        )
    
    # Wrong mu length
    with pytest.raises(ValueError, match="mu must have 3 elements"):
        generate_correlated_geometric_brownian(
            n_paths=5, n_steps=10, n_assets=3, mu=torch.tensor([0.1, 0.2])
        )
    
    # Wrong correlation matrix shape
    correlation = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    with pytest.raises(ValueError, match="correlation_matrix must be \\(3, 3\\)"):
        generate_correlated_geometric_brownian(
            n_paths=5, n_steps=10, n_assets=3, correlation_matrix=correlation
        )
    
    # Non-positive definite correlation matrix
    bad_correlation = torch.tensor([[1.0, 1.1], [1.1, 1.0]])
    with pytest.raises(ValueError, match="correlation_matrix must be positive definite"):
        generate_correlated_geometric_brownian(
            n_paths=5, n_steps=10, n_assets=2, correlation_matrix=bad_correlation
        )


@pytest.mark.parametrize("dtype", get_available_dtypes())
def test_generate_correlated_geometric_brownian_dtype(dtype, device: str = "cpu"):
    """Test with different dtypes."""
    n_paths = 3
    n_steps = 5
    n_assets = 2
    
    device = torch.device(device) if device else None
    output = generate_correlated_geometric_brownian(
        n_paths=n_paths,
        n_steps=n_steps,
        n_assets=n_assets,
        dtype=dtype,
        device=device
    )
    
    assert output.dtype == dtype
    assert output.device.type == (device.type if device else "cpu")


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", get_available_dtypes())
def test_generate_correlated_geometric_brownian_dtype_gpu(dtype):
    test_generate_correlated_geometric_brownian_dtype(dtype, device=select_most_accurate_gpu_device())


def test_generate_correlated_geometric_brownian_reproducibility(device: str = "cpu"):
    """Test reproducibility with same seed."""
    n_paths = 10
    n_steps = 20
    n_assets = 2
    
    # First generation
    torch.manual_seed(42)
    output1 = generate_correlated_geometric_brownian(
        n_paths=n_paths,
        n_steps=n_steps,
        n_assets=n_assets,
        device=device
    )
    
    # Second generation with same seed
    torch.manual_seed(42)
    output2 = generate_correlated_geometric_brownian(
        n_paths=n_paths,
        n_steps=n_steps,
        n_assets=n_assets,
        device=device
    )
    
    # Should be identical
    assert_close(output1, output2)


@pytest.mark.gpu
def test_generate_correlated_geometric_brownian_reproducibility_gpu():
    test_generate_correlated_geometric_brownian_reproducibility(device=select_most_accurate_gpu_device())


def test_generate_correlated_geometric_brownian_positive_prices(device: str = "cpu"):
    """Test that all generated prices are positive (property of geometric Brownian motion)."""
    n_paths = 100
    n_steps = 50
    n_assets = 3
    
    output = generate_correlated_geometric_brownian(
        n_paths=n_paths,
        n_steps=n_steps,
        n_assets=n_assets,
        device=device
    )
    
    # All prices should be positive
    assert (output > 0).all()


@pytest.mark.gpu
def test_generate_correlated_geometric_brownian_positive_prices_gpu():
    test_generate_correlated_geometric_brownian_positive_prices(device=select_most_accurate_gpu_device())

