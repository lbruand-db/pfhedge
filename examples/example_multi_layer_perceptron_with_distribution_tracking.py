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
from pfhedge.nn import BlackScholes


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


# Utility Functions
def save_plot_safely(fig, save_path, dpi=300):
    """Save plot with directory creation and user feedback."""
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {save_path}")


def calculate_global_axes(distributions, margin_percent=0.1, n_bins=50):
    """Calculate consistent axis limits and bins for multiple distributions."""
    all_data = np.concatenate(distributions)
    global_min, global_max = np.min(all_data), np.max(all_data)
    x_range = global_max - global_min
    x_margin = x_range * margin_percent
    
    return {
        'x_min': global_min - x_margin,
        'x_max': global_max + x_margin,
        'bins': np.linspace(global_min - x_margin, global_max + x_margin, n_bins + 1)
    }


def print_distribution_stats(stats, title="Distribution Statistics"):
    """Print formatted distribution statistics."""
    print(f"\n{title}:")
    print(f"  Mean P&L: {stats['mean']:.6f}")
    print(f"  Std P&L: {stats['std']:.6f}")
    print(f"  Median P&L: {stats['q50']:.6f}")
    print(f"  5th-95th percentile range: [{stats['q05']:.6f}, {stats['q95']:.6f}]")


def compute_blackscholes_comparison(derivative, n_paths=10000):
    """Compute Black-Scholes hedge and return distribution and statistics."""
    bs_model = BlackScholes(derivative)
    bs_hedger = Hedger(bs_model, bs_model.inputs())
    derivative.simulate(n_paths=n_paths, init_state=None)
    bs_pl_distribution = bs_hedger.compute_pl(derivative, hedge=None)
    bs_stats = compute_distribution_stats(bs_pl_distribution)
    return bs_pl_distribution, bs_stats


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


def plot_mean_std(ax, epochs, means, stds):
    """Plot mean with standard deviation bands."""
    ax.plot(epochs, means, 'b-', label='Mean', linewidth=2)
    ax.fill_between(epochs, 
                   [m - s for m, s in zip(means, stds)], 
                   [m + s for m, s in zip(means, stds)], 
                   alpha=0.3, color='blue', label='±1 Std')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Profit/Loss')
    ax.set_title('Mean ± Standard Deviation')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_quantiles(ax, epochs, q05s, medians, q95s):
    """Plot distribution quantiles."""
    ax.plot(epochs, q05s, 'r--', label='5th percentile', alpha=0.7)
    ax.plot(epochs, medians, 'g-', label='Median', linewidth=2)
    ax.plot(epochs, q95s, 'r--', label='95th percentile', alpha=0.7)
    ax.fill_between(epochs, q05s, q95s, alpha=0.2, color='red', label='5%-95% range')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Profit/Loss')
    ax.set_title('Distribution Quantiles')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_training_progress(ax, epochs, training_losses, validation_losses):
    """Plot training and validation losses."""
    ax.plot(epochs, training_losses, 'b-', label='Training Loss', alpha=0.7)
    ax.plot(epochs, validation_losses, 'r-', label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')


def plot_std_evolution(ax, epochs, stds):
    """Plot standard deviation evolution."""
    ax.plot(epochs, stds, 'purple', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Distribution Spread Evolution')
    ax.grid(True, alpha=0.3)


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
    
    # Create individual plots using helper functions
    plot_mean_std(axes[0, 0], epochs, means, stds)
    plot_quantiles(axes[0, 1], epochs, q05s, medians, q95s)
    plot_training_progress(axes[1, 0], results['epochs'], results['training_losses'], results['validation_losses'])
    plot_std_evolution(axes[1, 1], epochs, stds)
    
    plt.tight_layout()
    save_plot_safely(fig, save_path)
    plt.show()


def calculate_consistent_y_axis(full_distributions, epochs_to_show, bins):
    """Calculate consistent y-axis limit for snapshot plots."""
    global_y_max = 0
    for epoch in epochs_to_show:
        if epoch < len(full_distributions):
            distribution = full_distributions[epoch]
            counts, _ = np.histogram(distribution, bins=bins, density=True)
            y_max = np.max(counts) if len(counts) > 0 else 0
            global_y_max = max(global_y_max, y_max)
    return global_y_max * 1.1  # Add margin


def plot_single_snapshot(ax, distribution, bins, x_limits, y_limit, epoch):
    """Plot a single distribution snapshot."""
    ax.hist(distribution, bins=bins, alpha=0.7, density=True, color='skyblue', edgecolor='black')
    ax.axvline(np.mean(distribution), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(distribution):.4f}')
    ax.axvline(np.median(distribution), color='green', linestyle='--', linewidth=2, 
               label=f'Median: {np.median(distribution):.4f}')
    
    ax.set_xlim(x_limits['x_min'], x_limits['x_max'])
    ax.set_ylim(0, y_limit)
    ax.set_xlabel('Profit/Loss')
    ax.set_ylabel('Density')
    ax.set_title(f'Epoch {epoch}')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_distribution_snapshots(results, epochs_to_show=None, save_path=None):
    """Plot histograms of the distribution at specific epochs with consistent axes."""
    
    if 'full_distributions' not in results:
        print("Full distributions not tracked. Set track_full_distributions=True when training.")
        return
    
    full_distributions = results['full_distributions']
    n_epochs = len(full_distributions)
    
    if epochs_to_show is None:
        # Show distributions at beginning, middle, and end
        epochs_to_show = [0, n_epochs//4, n_epochs//2, 3*n_epochs//4, n_epochs-1]
    
    # Calculate global axis limits and bins using utility function
    axis_info = calculate_global_axes(full_distributions, margin_percent=0.1, n_bins=50)
    global_y_max = calculate_consistent_y_axis(full_distributions, epochs_to_show, axis_info['bins'])
    
    fig, axes = plt.subplots(1, len(epochs_to_show), figsize=(4*len(epochs_to_show), 4))
    if len(epochs_to_show) == 1:
        axes = [axes]
    
    fig.suptitle('Profit/Loss Distribution Snapshots (Multi-Layer Perceptron)', fontsize=16)
    
    for i, epoch in enumerate(epochs_to_show):
        if epoch >= len(full_distributions):
            continue
        distribution = full_distributions[epoch]
        plot_single_snapshot(axes[i], distribution, axis_info['bins'], axis_info, global_y_max, epoch)
    
    plt.tight_layout()
    save_plot_safely(fig, save_path)
    plt.show()


def setup_animation_data(results, interval_epochs=1, blackscholes_distribution=None):
    """Setup animation data and calculate axis limits."""
    full_distributions = results['full_distributions']
    distribution_stats = results['distribution_stats']
    n_epochs = len(full_distributions)
    
    # Sample epochs for animation (to avoid too many frames)
    epochs_to_animate = list(range(0, n_epochs, interval_epochs))
    if epochs_to_animate[-1] != n_epochs - 1:
        epochs_to_animate.append(n_epochs - 1)
    
    # Prepare distributions for axis calculation
    distributions_for_axes = full_distributions
    if blackscholes_distribution is not None:
        distributions_for_axes = full_distributions + [blackscholes_distribution]
    
    # Calculate axis limits and bins
    axis_info = calculate_global_axes(distributions_for_axes, margin_percent=0.1, n_bins=50)
    
    # Calculate y-axis limit including Black-Scholes if provided
    global_y_max = 0
    for epoch_idx in epochs_to_animate:
        distribution = full_distributions[epoch_idx]
        counts, _ = np.histogram(distribution, bins=axis_info['bins'], density=True)
        y_max = np.max(counts) if len(counts) > 0 else 0
        global_y_max = max(global_y_max, y_max)
    
    if blackscholes_distribution is not None:
        bs_counts, _ = np.histogram(blackscholes_distribution, bins=axis_info['bins'], density=True)
        bs_y_max = np.max(bs_counts) if len(bs_counts) > 0 else 0
        global_y_max = max(global_y_max, bs_y_max)
    
    axis_info['y_max'] = global_y_max * 1.1  # Add margin
    
    return {
        'epochs_to_animate': epochs_to_animate,
        'full_distributions': full_distributions,
        'distribution_stats': distribution_stats,
        'axis_info': axis_info
    }


def create_animation_frame(ax, animation_data, frame_idx, blackscholes_distribution=None):
    """Create a single animation frame."""
    ax.clear()
    
    epochs_to_animate = animation_data['epochs_to_animate']
    full_distributions = animation_data['full_distributions']
    distribution_stats = animation_data['distribution_stats']
    axis_info = animation_data['axis_info']
    
    epoch_idx = epochs_to_animate[frame_idx]
    distribution = full_distributions[epoch_idx]
    stats = distribution_stats[epoch_idx]
    bins = axis_info['bins']
    
    # Plot Black-Scholes distribution first (if provided) with transparency
    if blackscholes_distribution is not None:
        ax.hist(blackscholes_distribution, bins=bins, alpha=0.4, density=True, 
                color='orange', edgecolor='darkorange', linewidth=0.5, 
                label='Black-Scholes')
        
        # Add Black-Scholes mean line
        bs_mean = np.mean(blackscholes_distribution)
        ax.axvline(bs_mean, color='darkorange', linestyle=':', linewidth=2, 
                   label=f'BS Mean: {bs_mean:.4f}', alpha=0.8)
    
    # Plot MLP histogram on top
    ax.hist(distribution, bins=bins, alpha=0.7, density=True, 
            color='skyblue', edgecolor='darkblue', linewidth=0.5,
            label='Multi-Layer Perceptron')
    
    # Add MLP mean and median lines
    mean_val = np.mean(distribution)
    median_val = np.median(distribution)
    
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
               label=f'MLP Mean: {mean_val:.4f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, 
               label=f'MLP Median: {median_val:.4f}')
    
    # Set consistent axis limits for all frames
    ax.set_xlim(axis_info['x_min'], axis_info['x_max'])
    ax.set_ylim(0, axis_info['y_max'])
    
    # Add labels and title
    ax.set_xlabel('Profit/Loss', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    
    title_text = f'P&L Distribution Comparison - Epoch {epoch_idx}\n'
    if blackscholes_distribution is not None:
        title_text += f'MLP vs Black-Scholes Hedging'
    else:
        title_text += f'Multi-Layer Perceptron Hedging'
    
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add statistics text box for MLP
    stats_text = (f'MLP Statistics:\n'
                 f'Mean: {stats["mean"]:.4f}\n'
                 f'Std: {stats["std"]:.4f}\n'
                 f'5th-95th: [{stats["q05"]:.4f}, {stats["q95"]:.4f}]')
    
    # Add Black-Scholes statistics if available
    if blackscholes_distribution is not None:
        bs_mean = np.mean(blackscholes_distribution)
        bs_std = np.std(blackscholes_distribution)
        bs_q05 = np.quantile(blackscholes_distribution, 0.05)
        bs_q95 = np.quantile(blackscholes_distribution, 0.95)
        stats_text += (f'\n\nBS Statistics:\n'
                      f'Mean: {bs_mean:.4f}\n'
                      f'Std: {bs_std:.4f}\n'
                      f'5th-95th: [{bs_q05:.4f}, {bs_q95:.4f}]')
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def save_animation(anim, save_path, fps):
    """Save animation with fallback to GIF if MP4 fails."""
    if not save_path:
        return
        
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


def create_distribution_animation(results, save_path=None, fps=10, interval_epochs=1, blackscholes_distribution=None):
    """Create an animated video showing histogram evolution during training."""
    
    if 'full_distributions' not in results:
        print("Full distributions not tracked. Set track_full_distributions=True when training.")
        return
    
    # Setup animation data
    animation_data = setup_animation_data(results, interval_epochs, blackscholes_distribution)
    epochs_to_animate = animation_data['epochs_to_animate']
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    def animate(frame_idx):
        create_animation_frame(ax, animation_data, frame_idx, blackscholes_distribution)
    
    # Create animation
    print(f"Creating animation with {len(epochs_to_animate)} frames...")
    anim = animation.FuncAnimation(
        fig, animate, frames=len(epochs_to_animate), 
        interval=1000//fps, blit=False, repeat=True
    )
    
    # Save animation
    save_animation(anim, save_path, fps)
    return anim


# Configuration
class ExperimentConfig:
    """Configuration for the experiment."""
    N_PATHS = 10000
    N_EPOCHS = 200
    OUTPUT_DIR = "output"
    ANIMATION_FPS = 8
    ANIMATION_INTERVAL = 3
    SEED = 42


def setup_experiment():
    """Setup derivative and hedger for experiment."""
    torch.manual_seed(ExperimentConfig.SEED)
    
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
    
    return derivative, hedger


def run_training_experiment(hedger, derivative):
    """Run training and return results."""
    print("\nStarting training with distribution tracking...")
    results = fit_with_distribution_tracking(
        hedger=hedger,
        derivative=derivative,
        n_paths=ExperimentConfig.N_PATHS,
        n_epochs=ExperimentConfig.N_EPOCHS,
        track_full_distributions=True,  # Enable full distribution tracking
        verbose=True
    )

    # Final price
    final_price = hedger.price(derivative, n_paths=ExperimentConfig.N_PATHS)
    print(f"\nFinal hedging price: {final_price:.5e}")
    
    return results


def generate_comparison_analysis(results, derivative):
    """Generate Black-Scholes comparison and print statistics."""
    # Print MLP final statistics
    final_stats = results['distribution_stats'][-1]
    print_distribution_stats(final_stats, "Final MLP Distribution Statistics")

    # Create Black-Scholes hedger for comparison
    print("\nComputing Black-Scholes hedge for comparison...")
    bs_pl_distribution, bs_stats = compute_blackscholes_comparison(derivative, ExperimentConfig.N_PATHS)
    bs_distribution_np = bs_pl_distribution.detach().cpu().numpy()
    
    # Print Black-Scholes statistics
    print_distribution_stats(bs_stats, "Black-Scholes Distribution Statistics")
    
    return bs_distribution_np


def create_all_visualizations(results, bs_distribution, output_dir):
    """Generate all plots and animations."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create visualizations
    print("\nGenerating plots...")
    plot_distribution_evolution(results, save_path=f"{output_dir}/mlp_distribution_evolution.png")
    plot_distribution_snapshots(results, save_path=f"{output_dir}/mlp_distribution_snapshots.png")
    
    # Create animated video with consistent axis scaling and Black-Scholes comparison
    print("\nGenerating animated video with Black-Scholes comparison...")
    animation_obj = create_distribution_animation(
        results, 
        save_path=f"{output_dir}/mlp_vs_blackscholes_animation.mp4",
        fps=ExperimentConfig.ANIMATION_FPS,
        interval_epochs=ExperimentConfig.ANIMATION_INTERVAL,
        blackscholes_distribution=bs_distribution
    )
    
    return animation_obj


def print_completion_summary(output_dir):
    """Print completion summary."""
    print("\nAnalysis complete!")
    print(f"Plots and animation saved in '{output_dir}' directory")
    print("Files created:")
    print("  - mlp_distribution_evolution.png (static evolution plots)")
    print("  - mlp_distribution_snapshots.png (histogram snapshots)")
    print("  - mlp_vs_blackscholes_animation.mp4 (animated comparison: MLP vs Black-Scholes)")


def main():
    """Main execution function."""
    # Setup experiment
    derivative, hedger = setup_experiment()
    
    # Run training
    results = run_training_experiment(hedger, derivative)
    
    # Generate comparison analysis
    bs_distribution = generate_comparison_analysis(results, derivative)
    
    # Create visualizations
    create_all_visualizations(results, bs_distribution, ExperimentConfig.OUTPUT_DIR)
    
    # Print completion summary
    print_completion_summary(ExperimentConfig.OUTPUT_DIR)


if __name__ == "__main__":
    main()
