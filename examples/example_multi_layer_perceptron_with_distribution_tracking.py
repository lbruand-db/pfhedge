# Example to use a multi-layer perceptron as a hedging model with payoff distribution tracking
# Based on example_multi_layer_perceptron.py and example_payoff_distribution_history.py
# This example records the profit/loss distribution statistics epoch by epoch

import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os

import torch
from torch.optim import Adam
from tqdm import tqdm

sys.path.append("..")

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption
from pfhedge.nn import Hedger
from pfhedge.nn import MultiLayerPerceptron


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
    fig.suptitle('Profit/Loss Distribution Evolution During Training (Multi-Layer Perceptron)', fontsize=16)
    
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
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
    
    fig.suptitle('Profit/Loss Distribution Snapshots (Multi-Layer Perceptron)', fontsize=16)
    
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
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution snapshots saved to {save_path}")
    
    plt.show()


def create_distribution_animation(results, save_path=None, fps=10, interval_epochs=1):
    """Create an animated video showing histogram evolution during training."""
    
    if 'full_distributions' not in results:
        print("Full distributions not tracked. Set track_full_distributions=True when training.")
        return
    
    full_distributions = results['full_distributions']
    distribution_stats = results['distribution_stats']
    n_epochs = len(full_distributions)
    
    # Sample epochs for animation (to avoid too many frames)
    epochs_to_animate = list(range(0, n_epochs, interval_epochs))
    if epochs_to_animate[-1] != n_epochs - 1:
        epochs_to_animate.append(n_epochs - 1)
    
    # Find global min/max for consistent axis scaling
    all_data = np.concatenate(full_distributions)
    global_min, global_max = np.min(all_data), np.max(all_data)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(global_min * 1.1, global_max * 1.1)
    ax.set_xlabel('Profit/Loss', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Initialize empty histogram
    n_bins = 50
    bins = np.linspace(global_min * 1.1, global_max * 1.1, n_bins + 1)
    
    def animate(frame_idx):
        ax.clear()
        
        epoch_idx = epochs_to_animate[frame_idx]
        distribution = full_distributions[epoch_idx]
        stats = distribution_stats[epoch_idx]
        
        # Plot histogram
        ax.hist(distribution, bins=bins, alpha=0.7, density=True, 
                color='skyblue', edgecolor='black', linewidth=0.5)
        
        # Add mean and median lines
        mean_val = np.mean(distribution)
        median_val = np.median(distribution)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_val:.4f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, 
                   label=f'Median: {median_val:.4f}')
        
        # Set consistent axis limits
        ax.set_xlim(global_min * 1.1, global_max * 1.1)
        
        # Calculate y-limit based on current histogram
        counts, _ = np.histogram(distribution, bins=bins, density=True)
        y_max = np.max(counts) * 1.1 if len(counts) > 0 else 1
        ax.set_ylim(0, y_max)
        
        # Add labels and title
        ax.set_xlabel('Profit/Loss', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'P&L Distribution Evolution - Epoch {epoch_idx}\n'
                    f'Mean: {stats["mean"]:.4f}, Std: {stats["std"]:.4f}', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = (f'Statistics:\n'
                     f'5th percentile: {stats["q05"]:.4f}\n'
                     f'95th percentile: {stats["q95"]:.4f}\n'
                     f'Std Dev: {stats["std"]:.4f}')
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Create animation
    print(f"Creating animation with {len(epochs_to_animate)} frames...")
    anim = animation.FuncAnimation(
        fig, animate, frames=len(epochs_to_animate), 
        interval=1000//fps, blit=False, repeat=True
    )
    
    # Save animation
    if save_path:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print(f"Saving animation to {save_path}... This may take a few minutes.")
        
        # Try to save as MP4 (requires ffmpeg)
        try:
            writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='PFHedge'), bitrate=1800)
            anim.save(save_path, writer=writer)
            print(f"Animation saved successfully to {save_path}")
        except Exception as e:
            print(f"Could not save as MP4: {e}")
            # Fallback to GIF
            gif_path = save_path.replace('.mp4', '.gif')
            print(f"Trying to save as GIF: {gif_path}")
            try:
                anim.save(gif_path, writer='pillow', fps=fps)
                print(f"Animation saved as GIF: {gif_path}")
            except Exception as e2:
                print(f"Could not save animation: {e2}")
                return None
    
    return anim


if __name__ == "__main__":
    torch.manual_seed(42)
    
    print("Training Multi-Layer Perceptron with Payoff Distribution Tracking")
    print("=" * 65)

    # Prepare a derivative to hedge
    derivative = EuropeanOption(BrownianStock(cost=1e-4))
    print(f"Derivative: {derivative}")
    print(f"Underlying: {derivative.underlier}")

    # Create your hedger
    model = MultiLayerPerceptron()
    hedger = Hedger(model, ["log_moneyness", "expiry_time", "volatility", "prev_hedge"])
    print(f"Model inputs: {hedger.inputs}")

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

    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Create visualizations
    print("\nGenerating plots...")
    plot_distribution_evolution(results, save_path="output/mlp_distribution_evolution.png")
    plot_distribution_snapshots(results, save_path="output/mlp_distribution_snapshots.png")
    
    # Create animated video
    print("\nGenerating animated video...")
    animation_obj = create_distribution_animation(
        results, 
        save_path="output/mlp_distribution_animation.mp4",
        fps=5,  # 5 frames per second for smoother viewing
        interval_epochs=5  # Show every 5th epoch to reduce file size
    )
    
    print("\nAnalysis complete!")
    print(f"Plots and animation saved in '{output_dir}' directory")
    print("Files created:")
    print("  - mlp_distribution_evolution.png (static evolution plots)")
    print("  - mlp_distribution_snapshots.png (histogram snapshots)")
    print("  - mlp_distribution_animation.mp4 (animated histogram video)")
