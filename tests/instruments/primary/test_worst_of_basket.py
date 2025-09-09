import pytest
import torch
from torch.testing import assert_close

from pfhedge.instruments import WorstOfBasketStock


class TestWorstOfBasketStock:
    def test_repr(self):
        # Basic representation
        s = WorstOfBasketStock(n_assets=2, dt=1 / 100)
        expect = "WorstOfBasketStock(n_assets=2, sigma=0.2000, dt=0.0100)"
        assert repr(s) == expect

        # With cost
        s = WorstOfBasketStock(n_assets=3, dt=1 / 100, cost=0.001)
        expect = "WorstOfBasketStock(n_assets=3, sigma=0.2000, cost=0.0010, dt=0.0100)"
        assert repr(s) == expect

        # With different volatilities
        s = WorstOfBasketStock(n_assets=2, sigma=[0.2, 0.3], dt=1 / 100)
        expect = "WorstOfBasketStock(n_assets=2, sigma=[0.2000, 0.3000], dt=0.0100)"
        assert repr(s) == expect

        # With non-zero mu
        s = WorstOfBasketStock(n_assets=2, mu=0.05, dt=1 / 100)
        expect = "WorstOfBasketStock(n_assets=2, sigma=0.2000, mu=0.0500, dt=0.0100)"
        assert repr(s) == expect

        # With correlation
        correlation = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
        s = WorstOfBasketStock(n_assets=2, correlation_matrix=correlation, dt=1 / 100)
        expect = "WorstOfBasketStock(n_assets=2, sigma=0.2000, correlated=True, dt=0.0100)"
        assert repr(s) == expect

        # With dtype
        s = WorstOfBasketStock(n_assets=2, dt=1 / 100, dtype=torch.float64)
        expect = "WorstOfBasketStock(n_assets=2, sigma=0.2000, dt=0.0100, dtype=torch.float64)"
        assert repr(s) == expect

    def test_init_errors(self):
        # Too few assets
        with pytest.raises(ValueError, match="n_assets must be at least 2"):
            WorstOfBasketStock(n_assets=1)

        # Wrong sigma length
        with pytest.raises(ValueError, match="sigma list must have length 3"):
            WorstOfBasketStock(n_assets=3, sigma=[0.2, 0.3])

        # Wrong mu length  
        with pytest.raises(ValueError, match="mu list must have length 2"):
            WorstOfBasketStock(n_assets=2, mu=[0.05])

        # Wrong correlation matrix shape
        correlation = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
        with pytest.raises(ValueError, match="correlation_matrix must be \\(3, 3\\)"):
            WorstOfBasketStock(n_assets=3, correlation_matrix=correlation)

    def test_simulate_shape(self, device: str = "cpu"):
        # Test basic shapes
        s = WorstOfBasketStock(n_assets=3, dt=0.1).to(device)
        s.simulate(time_horizon=0.2, n_paths=10)
        
        assert s.spot.size() == torch.Size((10, 3))  # (n_paths, n_steps)
        assert s.basket.size() == torch.Size((10, 3, 3))  # (n_paths, n_steps, n_assets)

        # Test different time horizon
        s = WorstOfBasketStock(n_assets=2, dt=0.1).to(device)
        s.simulate(time_horizon=0.25, n_paths=5)
        
        assert s.spot.size() == torch.Size((5, 4))
        assert s.basket.size() == torch.Size((5, 4, 2))

    @pytest.mark.gpu
    def test_simulate_shape_gpu(self):
        self.test_simulate_shape(device="cuda")

    def test_worst_of_logic(self, device: str = "cpu"):
        """Test that spot really tracks the worst performing asset."""
        torch.manual_seed(42)
        
        # Create basket with uncorrelated assets
        s = WorstOfBasketStock(n_assets=3, sigma=0.3).to(device)
        s.simulate(n_paths=10, time_horizon=1.0)
        
        # Check that spot is indeed the worst performer at each time step
        basket_normalized = s.basket / s.basket[:, 0:1, :]  # Normalize by initial values
        worst_idx = basket_normalized.argmin(dim=2)  # Find worst asset index
        worst_performance = basket_normalized.gather(2, worst_idx.unsqueeze(2)).squeeze(2)
        
        # Convert to absolute prices (using first asset's initial price as reference)
        expected_spot = worst_performance * s.basket[:, 0, 0:1]
        
        assert_close(s.spot, expected_spot, atol=1e-6, rtol=1e-5)

    @pytest.mark.gpu
    def test_worst_of_logic_gpu(self):
        self.test_worst_of_logic(device="cuda")

    def test_correlation_effect(self, device: str = "cpu"):
        """Test that correlation affects the simulation."""
        torch.manual_seed(123)
        
        # Uncorrelated basket
        s_uncorr = WorstOfBasketStock(n_assets=2, sigma=0.2).to(device)
        s_uncorr.simulate(n_paths=1000, time_horizon=0.5)
        
        # Highly correlated basket
        torch.manual_seed(123)  # Same seed for comparison
        correlation = torch.tensor([[1.0, 0.9], [0.9, 1.0]]).to(device)
        s_corr = WorstOfBasketStock(n_assets=2, sigma=0.2, correlation_matrix=correlation).to(device)
        s_corr.simulate(n_paths=1000, time_horizon=0.5)
        
        # Correlation should reduce the spread of outcomes
        uncorr_std = s_uncorr.spot[:, -1].std()
        corr_std = s_corr.spot[:, -1].std()
        
        # With high correlation, worst-of should have lower volatility
        assert corr_std < uncorr_std

    @pytest.mark.gpu
    def test_correlation_effect_gpu(self):
        self.test_correlation_effect(device="cuda")

    def test_invalid_correlation_matrix(self):
        """Test error handling for invalid correlation matrices."""
        # Non-positive definite matrix
        bad_correlation = torch.tensor([[1.0, 1.1], [1.1, 1.0]])
        
        s = WorstOfBasketStock(n_assets=2, correlation_matrix=bad_correlation)
        with pytest.raises(ValueError, match="correlation_matrix must be positive definite"):
            s.simulate(n_paths=2, time_horizon=0.1)

    def test_volatility_property(self, device: str = "cpu"):
        """Test volatility and variance properties."""
        sigma_vals = [0.1, 0.2, 0.3]
        s = WorstOfBasketStock(n_assets=3, sigma=sigma_vals).to(device)
        s.simulate(n_paths=5, time_horizon=0.1)
        
        # Check that volatility property returns appropriate shape
        assert s.volatility.shape == s.spot.shape
        assert s.variance.shape == s.spot.shape
        
        # Variance should be volatility squared
        assert_close(s.variance, s.volatility ** 2)

    @pytest.mark.gpu  
    def test_volatility_property_gpu(self):
        self.test_volatility_property(device="cuda")

    def test_volatility_property_before_simulation(self):
        """Test that volatility property raises error before simulation."""
        s = WorstOfBasketStock(n_assets=2)
        
        with pytest.raises(AttributeError, match="Must simulate before accessing volatility"):
            _ = s.volatility

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_init_dtype(self, dtype, device: str = "cpu"):
        s = WorstOfBasketStock(n_assets=2, dtype=dtype, device=device)
        s.simulate(n_paths=2, time_horizon=0.1)
        
        assert s.dtype == dtype
        assert s.spot.dtype == dtype
        assert s.basket.dtype == dtype

    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_init_dtype_gpu(self, dtype):
        self.test_init_dtype(dtype, device="cuda")

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_to_dtype(self, dtype, device: str = "cpu"):
        # to(dtype) before simulate()
        s = WorstOfBasketStock(n_assets=2).to(dtype=dtype, device=device)
        s.simulate(n_paths=2, time_horizon=0.1)
        
        assert s.dtype == dtype
        assert s.spot.dtype == dtype
        assert s.basket.dtype == dtype

        # to(dtype) after simulate()
        s = WorstOfBasketStock(n_assets=2).to(device)
        s.simulate(n_paths=2, time_horizon=0.1)
        s.to(dtype=dtype)
        
        assert s.dtype == dtype
        assert s.spot.dtype == dtype
        assert s.basket.dtype == dtype

    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_to_dtype_gpu(self, dtype):
        self.test_to_dtype(dtype, device="cuda")

    def test_default_init_state(self):
        """Test default initial state property."""
        s = WorstOfBasketStock(n_assets=3)
        
        init_state = s.default_init_state
        assert len(init_state) == 3
        assert all(val == 1.0 for val in init_state)

    def test_custom_init_state(self, device: str = "cpu"):
        """Test simulation with custom initial state."""
        s = WorstOfBasketStock(n_assets=2).to(device)
        init_state = (100.0, 110.0)
        
        s.simulate(n_paths=3, time_horizon=0.1, init_state=init_state)
        
        # Check initial values
        assert_close(s.basket[:, 0, 0], torch.tensor(100.0).to(device))
        assert_close(s.basket[:, 0, 1], torch.tensor(110.0).to(device))

    @pytest.mark.gpu
    def test_custom_init_state_gpu(self):
        self.test_custom_init_state(device="cuda")

    def test_different_parameters(self, device: str = "cpu"):
        """Test with different mu and sigma for each asset."""
        sigmas = [0.1, 0.2, 0.3]
        mus = [0.0, 0.05, 0.1]
        
        s = WorstOfBasketStock(n_assets=3, sigma=sigmas, mu=mus).to(device)
        s.simulate(n_paths=10, time_horizon=0.5)
        
        # Check that simulation completed without errors
        assert s.spot.shape == (10, 126)  # ceil(0.5 / (1/250)) + 1 = 126
        assert s.basket.shape == (10, 126, 3)

    @pytest.mark.gpu
    def test_different_parameters_gpu(self):
        self.test_different_parameters(device="cuda")

    def test_tensor_parameters(self, device: str = "cpu"):
        """Test with tensor parameters."""
        sigma_tensor = torch.tensor([0.15, 0.25]).to(device)
        mu_tensor = torch.tensor([0.02, 0.08]).to(device)
        
        s = WorstOfBasketStock(n_assets=2, sigma=sigma_tensor, mu=mu_tensor).to(device)
        s.simulate(n_paths=5, time_horizon=0.2)
        
        assert s.spot.shape == (5, 51)  # ceil(0.2 / (1/250)) + 1 = 51
        assert s.basket.shape == (5, 51, 2)

    @pytest.mark.gpu
    def test_tensor_parameters_gpu(self):
        self.tensor_parameters(device="cuda")

    def test_is_listed(self):
        """Test that basket is listed instrument."""
        s = WorstOfBasketStock(n_assets=2)
        assert s.is_listed

    def test_to_device(self):
        """Test device conversion."""
        # to(device)
        s = WorstOfBasketStock(n_assets=2).to(device="cuda:0")
        assert s.device == torch.device("cuda:0")

        # Check parameter tensors are on correct device
        assert s.sigma.device == torch.device("cuda:0")
        assert s.mu.device == torch.device("cuda:0")

        # Test CPU conversion
        s_cpu = s.cpu()
        assert s_cpu.device == torch.device("cpu")
        assert s_cpu.sigma.device == torch.device("cpu")
        assert s_cpu.mu.device == torch.device("cpu")

    def test_reproducibility(self, device: str = "cpu"):
        """Test that simulations are reproducible with same seed."""
        
        # First simulation
        torch.manual_seed(42)
        s1 = WorstOfBasketStock(n_assets=2, sigma=0.2).to(device)
        s1.simulate(n_paths=10, time_horizon=0.2)
        
        # Second simulation with same seed
        torch.manual_seed(42) 
        s2 = WorstOfBasketStock(n_assets=2, sigma=0.2).to(device)
        s2.simulate(n_paths=10, time_horizon=0.2)
        
        # Results should be identical
        assert_close(s1.spot, s2.spot)
        assert_close(s1.basket, s2.basket)

    @pytest.mark.gpu
    def test_reproducibility_gpu(self):
        self.test_reproducibility(device="cuda")
