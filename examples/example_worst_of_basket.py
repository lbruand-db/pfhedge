"""
Worst of Basket Example

This example demonstrates how to use the WorstOfBasketStock primary instrument
to simulate a basket of correlated assets and track the worst performer.
"""

import torch
import matplotlib.pyplot as plt

from pfhedge.instruments import WorstOfBasketStock


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    print("=== Worst of Basket Stock Example ===\n")
    
    # Example 1: Simple uncorrelated basket
    print("1. Simple 3-asset basket (uncorrelated)")
    basket_simple = WorstOfBasketStock(n_assets=3, sigma=0.2)
    basket_simple.simulate(n_paths=5, time_horizon=20/250)
    
    print(f"Basket shape: {basket_simple.basket.shape}")
    print(f"Spot (worst-of) shape: {basket_simple.spot.shape}")
    print(f"Final worst-of prices (first 3 paths): {basket_simple.spot[:3, -1]}")
    print()
    
    # Example 2: Correlated basket with different volatilities
    print("2. Correlated basket with different volatilities")
    
    # Create correlation matrix
    correlation_matrix = torch.tensor([
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.7],
        [0.3, 0.7, 1.0]
    ])
    
    # Different volatilities for each asset
    volatilities = [0.15, 0.25, 0.20]
    
    basket_corr = WorstOfBasketStock(
        n_assets=3,
        sigma=volatilities,
        correlation_matrix=correlation_matrix
    )
    
    basket_corr.simulate(n_paths=1000, time_horizon=50/250)
    
    print(f"Correlation matrix:\n{correlation_matrix}")
    print(f"Volatilities: {volatilities}")
    print(f"Simulated {basket_corr.basket.shape[0]} paths over {basket_corr.basket.shape[1]} time steps")
    print(f"Average final worst-of price: {basket_corr.spot[:, -1].mean():.4f}")
    print(f"Std of final worst-of price: {basket_corr.spot[:, -1].std():.4f}")
    print()
    
    # Example 3: Visualize sample paths
    print("3. Visualizing sample paths")
    
    # Simulate fewer paths for visualization
    basket_viz = WorstOfBasketStock(
        n_assets=3,
        sigma=[0.2, 0.25, 0.18],
        correlation_matrix=correlation_matrix
    )
    
    basket_viz.simulate(n_paths=5, time_horizon=60/250)
    
    # Create time axis (in years)
    time_axis = torch.arange(basket_viz.basket.shape[1]) * basket_viz.dt
    
    # Plot individual assets and worst-of for first path
    plt.figure(figsize=(12, 8))
    
    # Plot all assets for first path
    for i in range(3):
        plt.plot(time_axis, basket_viz.basket[0, :, i], 
                label=f'Asset {i+1} (σ={basket_viz.sigma[i]:.2f})', alpha=0.7)
    
    # Plot worst-of
    plt.plot(time_axis, basket_viz.spot[0, :], 
            'k--', linewidth=2, label='Worst-of Basket')
    
    plt.xlabel('Time (years)')
    plt.ylabel('Price')
    plt.title('Worst of Basket: Individual Assets vs Worst Performer')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('/Users/lucas.bruand/w/pfhedge/examples/output/worst_of_basket_example.png', 
                dpi=150, bbox_inches='tight')
    print("Plot saved to examples/output/worst_of_basket_example.png")
    
    # Example 4: Compare correlated vs uncorrelated baskets
    print("\n4. Correlation impact on worst-of performance")
    
    torch.manual_seed(123)  # New seed for comparison
    
    # Uncorrelated basket
    basket_uncorr = WorstOfBasketStock(n_assets=3, sigma=0.2)
    basket_uncorr.simulate(n_paths=1000, time_horizon=50/250)
    
    # Highly correlated basket
    high_corr = torch.tensor([
        [1.0, 0.8, 0.8],
        [0.8, 1.0, 0.8],
        [0.8, 0.8, 1.0]
    ])
    
    basket_corr_high = WorstOfBasketStock(
        n_assets=3, 
        sigma=0.2, 
        correlation_matrix=high_corr
    )
    basket_corr_high.simulate(n_paths=1000, time_horizon=50/250)
    
    # Compare final worst-of distributions
    final_uncorr = basket_uncorr.spot[:, -1]
    final_corr = basket_corr_high.spot[:, -1]
    
    print(f"Uncorrelated - Mean: {final_uncorr.mean():.4f}, Std: {final_uncorr.std():.4f}")
    print(f"Highly correlated - Mean: {final_corr.mean():.4f}, Std: {final_corr.std():.4f}")
    print(f"Correlation reduces worst-of volatility by: {(1 - final_corr.std()/final_uncorr.std())*100:.1f}%")


if __name__ == "__main__":
    main()
