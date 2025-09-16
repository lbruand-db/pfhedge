# Example to track payoff distribution history during hedging model training
# Based on the No-Transaction Band Network from example_no_transaction_band.py
# This example records the profit/loss distribution statistics epoch by epoch

import sys
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as fn
from torch import Tensor
from torch.nn import Module
from torch.optim import Adam
from tqdm import tqdm

sys.path.append("..")
from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption
from pfhedge.nn import BlackScholes
from pfhedge.nn import Clamp
from pfhedge.nn import Hedger
from pfhedge.nn import MultiLayerPerceptron


class NoTransactionBandNet(Module):
    """Initialize a no-transaction band network.

    The `forward` method returns the next hedge ratio.

    Args:
        derivative (pfhedge.instruments.BaseDerivative): The derivative to hedge.

    Shape:
        - Input: :math:`(N, H_{\\text{in}})`, where :math:`(N, H_{\\text{in}})` is the
        number of input features. See `inputs()` for the names of input features.
        - Output: :math:`(N, 1)`.

    Examples:

        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanOption

        >>> derivative = EuropeanOption(BrownianStock(cost=1e-4))
        >>> m = NoTransactionBandNet(derivative)
        >>> m.inputs()
        ['log_moneyness', 'expiry_time', 'volatility', 'prev_hedge']
        >>> input = torch.tensor([
        ...     [-0.05, 0.1, 0.2, 0.5],
        ...     [-0.01, 0.1, 0.2, 0.5],
        ...     [ 0.00, 0.1, 0.2, 0.5],
        ...     [ 0.01, 0.1, 0.2, 0.5],
        ...     [ 0.05, 0.1, 0.2, 0.5]])
        >>> m(input)
        tensor([[0.2232],
                [0.4489],
                [0.5000],
                [0.5111],
                [0.7310]], grad_fn=<SWhereBackward>)
    """

    def __init__(self, derivative):
        super().__init__()

        self.delta = BlackScholes(derivative)
        self.mlp = MultiLayerPerceptron(out_features=2)
        self.clamp = Clamp()

    def inputs(self):
        return self.delta.inputs() + ["prev_hedge"]

    def forward(self, input: Tensor) -> Tensor:
        prev_hedge = input[..., [-1]]

        delta = self.delta(input[..., :-1])
        width = self.mlp(input[..., :-1])

        min = delta - fn.leaky_relu(width[..., [0]])
        max = delta + fn.leaky_relu(width[..., [1]])

        return self.clamp(prev_hedge, min=min, max=max)


def compute_distribution_stats(pl_tensor):
    """Compute statistics of the profit/loss distribution."""
    pl_np = pl_tensor.detach().cpu().numpy()
    
    stats = {
        'mean': np.mean(pl_np),
        'std': np.std(pl_np),
        'min': np.min(pl_np),
        'max': np.max(pl_np),
        'q01': np.quantile(pl_np, 0.01),
        'q05': np.quantile(pl_np, 0.05),
        'q25': np.quantile(pl_np, 0.25),
        'q50': np.quantile(pl_np, 0.50),  # median
        'q75': np.quantile(pl_np, 0.75),
        'q95': np.quantile(pl_np, 0.95),
        'q99': np.quantile(pl_np, 0.99),
    }
    
    return stats


def fit_with_distribution_tracking(hedger, derivative, n_epochs=200, n_paths=10000, 
                                   hedge=None, optimizer=Adam, init_state=None, 
                                   verbose=True, track_full_distributions=False):
    """
    Custom training loop that tracks profit/loss distribution history.
    
    Args:
        hedger: The Hedger instance
        derivative: The derivative to hedge
        n_epochs: Number of training epochs
        n_paths: Number of simulation paths
        hedge: Hedging instruments (optional)
        optimizer: Optimizer class or instance
        init_state: Initial state (optional)
        verbose: Whether to show progress
        track_full_distributions: Whether to store full distributions (memory intensive)
        
    Returns:
        dict: Contains training history and distribution statistics
    """
    
    # Configure optimizer
    if isinstance(optimizer, type):
        # Initialize parameters first if needed
        _ = hedger.compute_pnl(derivative, hedge=hedge, n_paths=1, init_state=init_state)
        optimizer = optimizer(hedger.parameters())
    
    # Initialize tracking lists
    training_losses = []
    validation_losses = []
    distribution_stats = []
    full_distributions = [] if track_full_distributions else None
    
    # Training loop
    progress = tqdm(range(n_epochs), disable=not verbose)
    
    for epoch in progress:
        # Training step
        hedger.train()
        optimizer.zero_grad()
        
        # Compute training loss
        loss = hedger.compute_loss(
            derivative,
            hedge=hedge,
            n_paths=n_paths,
            init_state=init_state,
        )
        
        loss.backward()
        optimizer.step()
        training_losses.append(loss.item())
        
        # Validation and distribution tracking
        hedger.eval()
        with torch.no_grad():
            # Compute validation loss
            val_loss = hedger.compute_loss(
                derivative,
                hedge=hedge,
                n_paths=n_paths,
                init_state=init_state,
                enable_grad=False,
            )
            validation_losses.append(val_loss.item())
            
            # Simulate and compute profit/loss distribution
            derivative.simulate(n_paths=n_paths, init_state=init_state)
            pl_distribution = hedger.compute_pl(derivative, hedge=hedge)
            
            # Compute and store distribution statistics
            stats = compute_distribution_stats(pl_distribution)
            stats['epoch'] = epoch
            distribution_stats.append(stats)
            
            # Optionally store full distribution
            if track_full_distributions:
                full_distributions.append(pl_distribution.detach().cpu().numpy())
        
        # Update progress bar
        progress.desc = f"Epoch {epoch+1}/{n_epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | PL Mean: {stats['mean']:.4f}"
    
    results = {
        'training_losses': training_losses,
        'validation_losses': validation_losses,
        'distribution_stats': distribution_stats,
        'epochs': list(range(n_epochs)),
    }
    
    if track_full_distributions:
        results['full_distributions'] = full_distributions
    
    return results


def plot_distribution_evolution(results, save_path=None):
    """Plot the evolution of distribution statistics over epochs."""
    
    stats = results['distribution_stats']
    epochs = [s['epoch'] for s in stats]
    
    # Extract statistics over time
    means = [s['mean'] for s in stats]
    stds = [s['std'] for s in stats]
    q05s = [s['q05'] for s in stats]
    q95s = [s['q95'] for s in stats]
    medians = [s['q50'] for s in stats]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Profit/Loss Distribution Evolution During Training', fontsize=16)
    
    # Plot 1: Mean and standard deviation
    ax1 = axes[0, 0]
    ax1.plot(epochs, means, 'b-', label='Mean', linewidth=2)
    ax1.fill_between(epochs, 
                     [m - s for m, s in zip(means, stds)], 
                     [m + s for m, s in zip(means, stds)], 
                     alpha=0.3, color='blue', label='±1 Std')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Profit/Loss')
    ax1.set_title('Mean ± Standard Deviation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Quantiles
    ax2 = axes[0, 1]
    ax2.plot(epochs, q05s, 'r--', label='5th percentile', alpha=0.7)
    ax2.plot(epochs, medians, 'g-', label='Median', linewidth=2)
    ax2.plot(epochs, q95s, 'r--', label='95th percentile', alpha=0.7)
    ax2.fill_between(epochs, q05s, q95s, alpha=0.2, color='red', label='5%-95% range')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Profit/Loss')
    ax2.set_title('Distribution Quantiles')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Training and validation losses
    ax3 = axes[1, 0]
    ax3.plot(results['epochs'], results['training_losses'], 'b-', label='Training Loss', alpha=0.7)
    ax3.plot(results['epochs'], results['validation_losses'], 'r-', label='Validation Loss', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training Progress')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Standard deviation evolution
    ax4 = axes[1, 1]
    ax4.plot(epochs, stds, 'purple', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Standard Deviation')
    ax4.set_title('Distribution Spread Evolution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_distribution_snapshots(results, epochs_to_show=None, save_path=None):
    """Plot histograms of the distribution at specific epochs."""
    
    if 'full_distributions' not in results:
        print("Full distributions not tracked. Set track_full_distributions=True when training.")
        return
    
    full_distributions = results['full_distributions']
    n_epochs = len(full_distributions)
    
    if epochs_to_show is None:
        # Show distributions at beginning, middle, and end
        epochs_to_show = [0, n_epochs//4, n_epochs//2, 3*n_epochs//4, n_epochs-1]
    
    fig, axes = plt.subplots(1, len(epochs_to_show), figsize=(4*len(epochs_to_show), 4))
    if len(epochs_to_show) == 1:
        axes = [axes]
    
    fig.suptitle('Profit/Loss Distribution Snapshots', fontsize=16)
    
    for i, epoch in enumerate(epochs_to_show):
        if epoch >= len(full_distributions):
            continue
            
        distribution = full_distributions[epoch]
        
        axes[i].hist(distribution, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        axes[i].axvline(np.mean(distribution), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(distribution):.4f}')
        axes[i].axvline(np.median(distribution), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(distribution):.4f}')
        axes[i].set_xlabel('Profit/Loss')
        axes[i].set_ylabel('Density')
        axes[i].set_title(f'Epoch {epoch}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution snapshots saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(42)
    
    print("Training No-Transaction Band Network with Payoff Distribution Tracking")
    print("=" * 70)

    # Prepare a derivative to hedge
    derivative = EuropeanOption(BrownianStock(cost=1e-4))
    print(f"Derivative: {derivative}")
    print(f"Underlying: {derivative.underlier}")

    # Create hedger
    model = NoTransactionBandNet(derivative)
    hedger = Hedger(model, model.inputs())
    print(f"Model inputs: {model.inputs()}")

    # Fit with distribution tracking
    print("\nStarting training with distribution tracking...")
    results = fit_with_distribution_tracking(
        hedger=hedger,
        derivative=derivative,
        n_paths=10000,
        n_epochs=200,
        track_full_distributions=True,  # Enable full distribution tracking
        verbose=True
    )

    # Final price
    final_price = hedger.price(derivative, n_paths=10000)
    print(f"\nFinal hedging price: {final_price:.5e}")

    # Print some final statistics
    final_stats = results['distribution_stats'][-1]
    print(f"\nFinal Distribution Statistics:")
    print(f"  Mean P&L: {final_stats['mean']:.6f}")
    print(f"  Std P&L: {final_stats['std']:.6f}")
    print(f"  Median P&L: {final_stats['q50']:.6f}")
    print(f"  5th-95th percentile range: [{final_stats['q05']:.6f}, {final_stats['q95']:.6f}]")

    # Create visualizations
    print("\nGenerating plots...")
    plot_distribution_evolution(results, save_path="output/distribution_evolution.png")
    plot_distribution_snapshots(results, save_path="output/distribution_snapshots.png")
    
    print("\nAnalysis complete!")
