import pytest
import torch
from torch.testing import assert_close

from pfhedge.instruments import WorstOfBasketStock, WorstOfBasketAutocall
from tests._utils import select_most_accurate_gpu_device, get_available_dtypes


class TestWorstOfBasketAutocall:
    
    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        basket = WorstOfBasketStock(n_assets=2)
        autocall = WorstOfBasketAutocall(underlier=basket)
        
        assert autocall.autocall_barrier == 1.0
        assert autocall.protection_barrier == 0.65
        assert autocall.coupon_rate == 0.08
        assert autocall.notional == 1.0
        assert autocall.maturity == 1.0
        assert autocall.observation_dates == [0.25, 0.5, 0.75, 1.0]
    
    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        basket = WorstOfBasketStock(n_assets=3)
        observation_dates = [0.5, 1.0]
        
        autocall = WorstOfBasketAutocall(
            underlier=basket,
            autocall_barrier=0.95,
            protection_barrier=0.70,
            coupon_rate=0.10,
            observation_dates=observation_dates,
            notional=100.0,
            maturity=2.0
        )
        
        assert autocall.autocall_barrier == 0.95
        assert autocall.protection_barrier == 0.70
        assert autocall.coupon_rate == 0.10
        assert autocall.notional == 100.0
        assert autocall.maturity == 2.0
        assert autocall.observation_dates == [0.5, 1.0]
    
    def test_observation_dates_automatic_maturity(self):
        """Test that maturity is automatically added to observation dates."""
        basket = WorstOfBasketStock(n_assets=2)
        autocall = WorstOfBasketAutocall(
            underlier=basket,
            observation_dates=[0.25, 0.5]
        )
        
        # Should automatically include maturity (1.0)
        assert autocall.observation_dates == [0.25, 0.5, 1.0]
    
    def test_repr(self):
        """Test string representation."""
        basket = WorstOfBasketStock(n_assets=2)
        autocall = WorstOfBasketAutocall(underlier=basket)
        
        repr_str = repr(autocall)
        assert "WorstOfBasketAutocall" in repr_str
        assert "autocall_barrier=1.0000" in repr_str
        assert "protection_barrier=0.6500" in repr_str
        assert "coupon_rate=0.0800" in repr_str
    
    def test_simulate_and_payoff_shape(self, device: str = "cpu"):
        """Test simulation and payoff shape."""
        basket = WorstOfBasketStock(n_assets=2, dt=0.1).to(device)
        autocall = WorstOfBasketAutocall(underlier=basket, maturity=0.5)
        
        n_paths = 10
        autocall.simulate(n_paths=n_paths)
        payoffs = autocall.payoff()
        
        assert payoffs.shape == (n_paths,)
        assert payoffs.device.type == device if device != "cpu" else "cpu"
    
    @pytest.mark.gpu
    def test_simulate_and_payoff_shape_gpu(self):
        self.test_simulate_and_payoff_shape(device=select_most_accurate_gpu_device())
    
    def test_autocall_behavior_certain_autocall(self):
        """Test autocall behavior when autocall is certain to happen."""
        # Create a basket that will definitely be above barrier
        basket = WorstOfBasketStock(n_assets=2, sigma=0.0, mu=0.1, dt=0.1)
        autocall = WorstOfBasketAutocall(
            underlier=basket,
            autocall_barrier=0.9,  # Low barrier, should trigger
            coupon_rate=0.08,
            observation_dates=[0.5, 1.0],
            maturity=1.0
        )
        
        autocall.simulate(n_paths=10)
        payoffs = autocall.payoff()
        
        # All should autocall at first observation (0.5 years)
        expected_payoff = autocall.notional * (1 + autocall.coupon_rate * 0.5)
        assert_close(payoffs, torch.full_like(payoffs, expected_payoff), rtol=1e-3, atol=1e-5)
    
    def test_protection_barrier_behavior(self):
        """Test protection barrier behavior."""
        # Create a basket that will be below autocall but above protection
        basket = WorstOfBasketStock(n_assets=2, sigma=0.0, mu=-0.3, dt=1.0)
        autocall = WorstOfBasketAutocall(
            underlier=basket,
            autocall_barrier=1.0,  # Won't trigger
            protection_barrier=0.6,  # Should provide protection
            observation_dates=[1.0],
            maturity=1.0
        )
        
        autocall.simulate(n_paths=10)
        payoffs = autocall.payoff()
        
        # Should get capital protection (notional = 1.0)
        assert_close(payoffs, torch.ones_like(payoffs), atol=0.1)
    
    def test_at_risk_behavior(self):
        """Test at-risk behavior when below protection barrier."""
        # Create a basket that will be well below protection barrier
        basket = WorstOfBasketStock(n_assets=2, sigma=0.0, mu=-0.8, dt=1.0)
        autocall = WorstOfBasketAutocall(
            underlier=basket,
            autocall_barrier=1.0,
            protection_barrier=0.7,
            observation_dates=[1.0],
            maturity=1.0
        )
        
        autocall.simulate(n_paths=10)
        payoffs = autocall.payoff()
        
        # Should get participation in worst performance (< protection barrier)
        assert (payoffs < autocall.protection_barrier).all()
        assert (payoffs > 0).all()  # Should still be positive
    
    def test_get_observation_indices(self):
        """Test conversion of observation dates to indices."""
        basket = WorstOfBasketStock(n_assets=2, dt=0.1)
        autocall = WorstOfBasketAutocall(
            underlier=basket,
            observation_dates=[0.25, 0.5, 1.0],
            maturity=1.0
        )
        
        # Simulate to create spot prices
        autocall.simulate(n_paths=1)
        
        indices = autocall._get_observation_indices()
        
        # With dt=0.1 and maturity=1.0, we expect indices roughly at 2, 5, 10
        assert len(indices) == 3
        assert all(isinstance(idx, int) for idx in indices)
        assert all(idx >= 0 for idx in indices)
    
    def test_autocall_probabilities(self):
        """Test autocall probability calculation."""
        basket = WorstOfBasketStock(n_assets=2, sigma=0.2)
        autocall = WorstOfBasketAutocall(
            underlier=basket,
            observation_dates=[0.25, 0.5, 0.75, 1.0]
        )
        
        autocall.simulate(n_paths=1000)
        probs = autocall.get_autocall_probabilities()
        
        assert probs.shape == (4,)  # 4 observation dates
        assert (probs >= 0).all()
        assert (probs <= 1).all()
        assert probs.sum() <= 1.0  # Total probability can't exceed 1
    
    def test_scenario_analysis(self):
        """Test scenario breakdown analysis."""
        basket = WorstOfBasketStock(n_assets=2, sigma=0.3)
        autocall = WorstOfBasketAutocall(underlier=basket)
        
        autocall.simulate(n_paths=1000)
        scenarios = autocall.get_average_payoff_by_scenario()
        
        assert isinstance(scenarios, dict)
        
        # Check that probabilities sum to approximately 1
        total_prob = sum(data['probability'] for data in scenarios.values())
        assert_close(torch.tensor(total_prob), torch.tensor(1.0), atol=0.05)
        
        # Check that all payoffs are reasonable
        for scenario, data in scenarios.items():
            assert data['probability'] >= 0
            assert data['probability'] <= 1
            assert data['average_payoff'] >= 0
    
    @pytest.mark.parametrize("dtype", get_available_dtypes())
    def test_dtype_consistency(self, dtype, device: str = "cpu"):
        """Test dtype consistency."""
        basket = WorstOfBasketStock(n_assets=2, dtype=dtype, device=device)
        autocall = WorstOfBasketAutocall(underlier=basket)
        
        autocall.simulate(n_paths=5)
        payoffs = autocall.payoff()
        
        assert payoffs.dtype == dtype
        assert payoffs.device.type == device if device != "cpu" else "cpu"
    
    @pytest.mark.gpu
    @pytest.mark.parametrize("dtype", get_available_dtypes())
    def test_dtype_consistency_gpu(self, dtype):
        self.test_dtype_consistency(dtype, device=select_most_accurate_gpu_device())
    
    def test_to_device(self):
        """Test device transfer."""
        basket = WorstOfBasketStock(n_assets=2)
        autocall = WorstOfBasketAutocall(underlier=basket)
        
        # Test CPU to GPU
        gpu_device = select_most_accurate_gpu_device()
        if gpu_device != "cpu":
            autocall.to(device=gpu_device)
            assert autocall.ul().device.type == gpu_device
            
            # Test simulation on GPU
            autocall.simulate(n_paths=5)
            payoffs = autocall.payoff()
            assert payoffs.device.type == gpu_device
    
    def test_multiple_assets(self):
        """Test with different numbers of assets."""
        for n_assets in [2, 3, 5]:
            basket = WorstOfBasketStock(n_assets=n_assets, sigma=0.2)
            autocall = WorstOfBasketAutocall(underlier=basket)
            
            autocall.simulate(n_paths=10)
            payoffs = autocall.payoff()
            
            assert payoffs.shape == (10,)
            assert (payoffs >= 0).all()
    
    def test_different_maturities(self):
        """Test with different maturities."""
        basket = WorstOfBasketStock(n_assets=2, sigma=0.2)
        
        for maturity in [0.5, 1.0, 2.0]:
            autocall = WorstOfBasketAutocall(
                underlier=basket,
                maturity=maturity,
                observation_dates=[maturity]  # Single observation at maturity
            )
            
            autocall.simulate(n_paths=10)
            payoffs = autocall.payoff()
            
            assert payoffs.shape == (10,)
            assert (payoffs >= 0).all()
    
    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        basket1 = WorstOfBasketStock(n_assets=2, sigma=0.2)
        basket2 = WorstOfBasketStock(n_assets=2, sigma=0.2)
        
        autocall1 = WorstOfBasketAutocall(underlier=basket1)
        autocall2 = WorstOfBasketAutocall(underlier=basket2)
        
        # First simulation
        torch.manual_seed(42)
        autocall1.simulate(n_paths=10)
        payoffs1 = autocall1.payoff()
        
        # Second simulation with same seed
        torch.manual_seed(42)
        autocall2.simulate(n_paths=10)
        payoffs2 = autocall2.payoff()
        
        assert_close(payoffs1, payoffs2)
