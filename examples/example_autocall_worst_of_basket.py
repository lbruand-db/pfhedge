"""
Worst-of Basket Autocall Example

This example demonstrates how to create and use a worst-of basket autocall note.
An autocall is a structured product that can redeem early if certain conditions are met,
paying a coupon. If it doesn't redeem early, the final payoff depends on the 
worst performing asset in the basket.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

from pfhedge.instruments import WorstOfBasketStock, WorstOfBasketAutocall


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    print("=== Worst-of Basket Autocall Example ===\n")
    
    # Create a worst-of basket as the underlying
    print("1. Setting up the underlying basket")
    
    # Create correlation matrix for 3 assets
    correlation_matrix = torch.tensor([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.7],
        [0.3, 0.7, 1.0]
    ])
    
    # Different volatilities for each asset
    volatilities = [0.2, 0.25, 0.18]
    
    # Create the worst-of basket
    basket = WorstOfBasketStock(
        n_assets=3,
        sigma=volatilities,
        correlation_matrix=correlation_matrix,
        dt=1/250  # Daily time steps
    )
    
    print(f"Created basket with {basket.n_assets} assets")
    print(f"Volatilities: {volatilities}")
    print(f"Daily time step: {basket.dt}")
    print()
    
    # Create the autocall note
    print("2. Creating the autocall note")
    
    autocall = WorstOfBasketAutocall(
        underlier=basket,
        autocall_barrier=1.0,      # Can autocall if worst performer >= 100% of initial
        protection_barrier=0.65,   # Capital protection if worst performer >= 65% at maturity
        coupon_rate=0.08,          # 8% annual coupon on autocall
        observation_dates=[0.25, 0.5, 0.75, 1.0],  # Quarterly observations
        notional=1.0,              # €1 notional
        maturity=1.0               # 1 year maturity
    )
    
    print(f"Autocall barrier: {autocall.autocall_barrier * 100}%")
    print(f"Protection barrier: {autocall.protection_barrier * 100}%")
    print(f"Annual coupon rate: {autocall.coupon_rate * 100}%")
    print(f"Observation dates: {autocall.observation_dates}")
    print(f"Maturity: {autocall.maturity} years")
    print()
    
    # Simulate paths and calculate payoffs
    print("3. Simulating paths and calculating payoffs")
    
    n_paths = 10000
    autocall.simulate(n_paths=n_paths)
    payoffs = autocall.payoff()
    
    print(f"Simulated {n_paths} paths")
    print(f"Average payoff: {payoffs.mean():.4f}")
    print(f"Payoff standard deviation: {payoffs.std():.4f}")
    print(f"Minimum payoff: {payoffs.min():.4f}")
    print(f"Maximum payoff: {payoffs.max():.4f}")
    print()
    
    # Analyze autocall probabilities
    print("4. Autocall probability analysis")
    
    autocall_probs = autocall.get_autocall_probabilities()
    
    print("Probability of autocalling at each observation date:")
    for i, (date, prob) in enumerate(zip(autocall.observation_dates, autocall_probs)):
        print(f"  {date * 12:4.0f} months: {prob:.1%}")
    
    cumulative_autocall_prob = autocall_probs.sum()
    print(f"Total autocall probability: {cumulative_autocall_prob:.1%}")
    print(f"Probability of no autocall: {1 - cumulative_autocall_prob:.1%}")
    print()
    
    # Detailed scenario analysis
    print("5. Scenario breakdown")
    
    scenario_analysis = autocall.get_average_payoff_by_scenario()
    
    for scenario, data in scenario_analysis.items():
        print(f"{scenario.replace('_', ' ').title()}:")
        print(f"  Probability: {data['probability']:.1%}")
        print(f"  Average payoff: {data['average_payoff']:.4f}")
        if 'average_worst_performance' in data:
            print(f"  Average worst performance: {data['average_worst_performance']:.1%}")
        print()
    
    # Create visualizations
    print("6. Creating visualizations")
    
    # Plot 1: Payoff distribution
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.hist(payoffs.numpy(), bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Payoff')
    plt.ylabel('Frequency')
    plt.title('Payoff Distribution')
    plt.axvline(payoffs.mean(), color='red', linestyle='--', label=f'Mean: {payoffs.mean():.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Worst-of performance paths (sample)
    plt.subplot(2, 3, 2)
    sample_size = 20
    time_axis = torch.arange(basket.spot.shape[1]) * basket.dt
    
    for i in range(sample_size):
        plt.plot(time_axis, basket.spot[i, :], alpha=0.6, linewidth=0.8)
    
    plt.axhline(autocall.autocall_barrier, color='green', linestyle='--', 
                label=f'Autocall barrier ({autocall.autocall_barrier})')
    plt.axhline(autocall.protection_barrier, color='red', linestyle='--', 
                label=f'Protection barrier ({autocall.protection_barrier})')
    plt.xlabel('Time (years)')
    plt.ylabel('Worst-of Performance')
    plt.title(f'Sample Worst-of Paths (n={sample_size})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Autocall probabilities by observation date
    plt.subplot(2, 3, 3)
    months = [date * 12 for date in autocall.observation_dates]
    plt.bar(months, autocall_probs.numpy(), alpha=0.7, edgecolor='black')
    plt.xlabel('Observation Date (months)')
    plt.ylabel('Autocall Probability')
    plt.title('Autocall Probabilities')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Final worst-of performance distribution (for non-autocalled paths)
    plt.subplot(2, 3, 4)
    
    # Identify non-autocalled paths
    final_performance = basket.spot[:, -1]
    autocalled_mask = torch.zeros(n_paths, dtype=torch.bool)
    
    # Check which paths autocalled
    obs_indices = autocall._get_observation_indices()
    for obs_idx in obs_indices[:-1]:  # Exclude final observation
        if obs_idx < basket.spot.shape[1]:
            current_perf = basket.spot[:, obs_idx]
            autocall_condition = (~autocalled_mask) & (current_perf >= autocall.autocall_barrier)
            autocalled_mask |= autocall_condition
    
    # Final autocall check
    final_autocall = (~autocalled_mask) & (final_performance >= autocall.autocall_barrier)
    autocalled_mask |= final_autocall
    
    non_autocalled_performance = final_performance[~autocalled_mask]
    
    if len(non_autocalled_performance) > 0:
        plt.hist(non_autocalled_performance.numpy(), bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(autocall.protection_barrier, color='red', linestyle='--', 
                    label=f'Protection barrier ({autocall.protection_barrier})')
        plt.xlabel('Final Worst-of Performance')
        plt.ylabel('Frequency')
        plt.title('Final Performance (Non-autocalled paths)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'All paths autocalled', ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=12)
        plt.title('Final Performance (Non-autocalled paths)')
    
    # Plot 5: Payoff vs worst-of performance
    plt.subplot(2, 3, 5)
    plt.scatter(final_performance.numpy(), payoffs.numpy(), alpha=0.5, s=1)
    plt.xlabel('Final Worst-of Performance')
    plt.ylabel('Payoff')
    plt.title('Payoff vs Final Performance')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Cumulative autocall probability
    plt.subplot(2, 3, 6)
    cumulative_probs = torch.cumsum(autocall_probs, dim=0)
    plt.plot(months, cumulative_probs.numpy(), 'o-', linewidth=2, markersize=6)
    plt.xlabel('Time (months)')
    plt.ylabel('Cumulative Autocall Probability')
    plt.title('Cumulative Autocall Probability')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('output/autocall_analysis.png', 
                dpi=150, bbox_inches='tight')
    print("Analysis plots saved to examples/output/autocall_analysis.png")
    
    # Compare different autocall structures
    print("\n7. Comparing different autocall structures")
    
    # More aggressive structure (lower barriers, higher coupon)
    autocall_aggressive = WorstOfBasketAutocall(
        underlier=basket,
        autocall_barrier=0.90,     # Lower autocall barrier
        protection_barrier=0.70,   # Higher protection
        coupon_rate=0.12,          # Higher coupon
        observation_dates=[0.25, 0.5, 0.75, 1.0],
        maturity=1.0
    )
    
    # Conservative structure (higher barriers, lower coupon)
    autocall_conservative = WorstOfBasketAutocall(
        underlier=basket,
        autocall_barrier=1.05,     # Higher autocall barrier
        protection_barrier=0.60,   # Lower protection
        coupon_rate=0.06,          # Lower coupon
        observation_dates=[0.25, 0.5, 0.75, 1.0],
        maturity=1.0
    )
    
    # Simulate all structures with same underlying paths
    torch.manual_seed(42)  # Ensure same underlying paths
    basket_copy1 = WorstOfBasketStock(
        n_assets=3, sigma=volatilities, correlation_matrix=correlation_matrix, dt=1/250
    )
    autocall_aggressive.register_underlier("underlier", basket_copy1)
    autocall_aggressive.simulate(n_paths=n_paths)
    payoffs_aggressive = autocall_aggressive.payoff()
    
    torch.manual_seed(42)  # Ensure same underlying paths
    basket_copy2 = WorstOfBasketStock(
        n_assets=3, sigma=volatilities, correlation_matrix=correlation_matrix, dt=1/250
    )
    autocall_conservative.register_underlier("underlier", basket_copy2)
    autocall_conservative.simulate(n_paths=n_paths)
    payoffs_conservative = autocall_conservative.payoff()
    
    # Compare results
    structures = [
        ("Base", autocall, payoffs),
        ("Aggressive", autocall_aggressive, payoffs_aggressive),
        ("Conservative", autocall_conservative, payoffs_conservative)
    ]
    
    print("Structure comparison:")
    print(f"{'Structure':<12} {'Avg Payoff':<12} {'Std Dev':<12} {'Min':<8} {'Max':<8} {'Autocall %':<12}")
    print("-" * 70)
    
    for name, structure, payoff in structures:
        autocall_prob = structure.get_autocall_probabilities().sum()
        print(f"{name:<12} {payoff.mean():<12.4f} {payoff.std():<12.4f} "
              f"{payoff.min():<8.4f} {payoff.max():<8.4f} {autocall_prob:<12.1%}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
