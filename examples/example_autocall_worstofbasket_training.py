"""
Deep Hedging an Autocall on Worst-of-Basket

This example demonstrates how to use deep hedging with a multi-layer perceptron
to hedge a worst-of-basket autocall note. The autocall can redeem early if the
worst performing asset in the basket is at or above the autocall barrier on
observation dates, paying a coupon. Otherwise, the final payoff depends on
capital protection and downside participation.
"""

import sys

import torch
from torch.nn import ReLU

sys.path.append("..")

from pfhedge.instruments import WorstOfBasketStock, WorstOfBasketAutocall
from pfhedge.nn import Hedger, MultiLayerPerceptron


def main():
    print("=== Deep Hedging Autocall on Worst-of-Basket ===\n")
    
    torch.manual_seed(42)

    # Create correlation matrix for 3 assets  
    correlation_matrix = torch.tensor([
        [1.0, 0.6, 0.4],
        [0.6, 1.0, 0.5], 
        [0.4, 0.5, 1.0]
    ])
    
    # Create the worst-of basket with transaction costs
    print("1. Setting up the underlying basket")
    basket = WorstOfBasketStock(
        n_assets=3,
        sigma=[0.20, 0.25, 0.18],    # Different volatilities per asset
        correlation_matrix=correlation_matrix,
        cost=1e-4,                   # Transaction cost for hedging
        dt=1/250                     # Daily time steps
    )
    print(f"   - {basket.n_assets} assets with correlations")
    print(f"   - Volatilities: {[0.20, 0.25, 0.18]}")
    print(f"   - Transaction cost: {basket.cost}")
    print()

    # Create the autocall derivative
    print("2. Creating the autocall note")
    autocall = WorstOfBasketAutocall(
        underlier=basket,
        autocall_barrier=1.0,        # 100% of initial level
        protection_barrier=0.65,     # 65% capital protection
        coupon_rate=0.08,            # 8% annual coupon
        observation_dates=[0.25, 0.5, 0.75, 1.0],  # Quarterly observations
        maturity=1.0                 # 1 year maturity
    )
    print(f"   - Autocall barrier: {autocall.autocall_barrier * 100}%")
    print(f"   - Protection barrier: {autocall.protection_barrier * 100}%")
    print(f"   - Annual coupon: {autocall.coupon_rate * 100}%")
    print(f"   - Maturity: {autocall.maturity} years")
    print()

    # Create the deep hedging model
    print("3. Setting up deep hedging model")
    model = MultiLayerPerceptron(
        n_layers=4,                  # Deeper network for complex payoff
        n_units=32,                  # More units for basket complexity
        activation=ReLU()            # ReLU activation function
    )
    
    # Use features that work with autocall derivatives  
    features = [
        "volatility",                # Volatility of underlying basket
        "prev_hedge"                 # Previous hedge position
    ]
    
    hedger = Hedger(model, features)
    print(f"   - Model: {model.__class__.__name__} (4 layers, 32 units, ReLU activation)")
    print(f"   - Features: {features}")
    print()

    # Train the hedger
    print("4. Training the deep hedger")
    print("   (This may take a few minutes...)")
    
    history = hedger.fit(
        autocall,
        n_paths=8000,               # Number of simulation paths
        n_epochs=100,               # Training epochs (reduced for example)
        n_times=1,                  # Times per epoch for variance reduction
        verbose=True                # Show progress
    )
    print()

    # Evaluate the hedger
    print("5. Evaluating hedging performance")
    
    # Calculate the price using the trained hedger
    torch.manual_seed(42)  # For reproducible evaluation
    price = hedger.price(autocall, n_paths=10000, n_times=3)
    print(f"   - Deep hedge price: {price:.5f}")
    
    # Calculate profit and loss statistics
    torch.manual_seed(42)
    autocall.simulate(n_paths=10000)
    pl = hedger.compute_pl(autocall)
    
    print(f"   - P&L mean: {pl.mean():.5f}")
    print(f"   - P&L std: {pl.std():.5f}")
    print(f"   - P&L min: {pl.min():.5f}")
    print(f"   - P&L max: {pl.max():.5f}")
    
    # Calculate risk metrics
    var_95 = torch.quantile(pl, 0.05)  # 5% Value at Risk
    cvar_95 = pl[pl <= var_95].mean()  # Expected Shortfall
    
    print(f"   - VaR (95%): {var_95:.5f}")
    print(f"   - CVaR (95%): {cvar_95:.5f}")
    print()

    # Analyze autocall probabilities from the simulation
    print("6. Autocall probability analysis")
    autocall_probs = autocall.get_autocall_probabilities()
    total_autocall_prob = autocall_probs.sum()
    
    print(f"   - Probability of autocalling:")
    for i, (date, prob) in enumerate(zip(autocall.observation_dates, autocall_probs)):
        print(f"     * {date * 12:4.0f} months: {prob:.1%}")
    print(f"   - Total autocall probability: {total_autocall_prob:.1%}")
    print(f"   - No autocall probability: {1 - total_autocall_prob:.1%}")
    print()

    print("7. Summary")
    print(f"   ✅ Successfully trained deep hedger for worst-of-basket autocall")
    print(f"   ✅ Final hedge price: {price:.5f}")
    print(f"   ✅ Hedging P&L std: {pl.std():.5f}")
    print(f"   ✅ Risk metrics computed for {len(pl)} scenarios")
    print()
    print("   The deep hedger has learned to dynamically hedge the complex")
    print("   autocall payoff by trading the underlying worst-of basket.")
    print("   Lower P&L volatility indicates better hedging performance.")


if __name__ == "__main__":
    main()
