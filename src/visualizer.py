"""
Visualization Module
Generates publication-quality plots for DDoS detection experiment results.

Plots include:
- Bar chart comparison of accuracy, precision, recall, F1 across methods
- ROC curve plot for all methods
- Confusion matrix heatmap
- K-value vs accuracy line plot
- Degree-of-attack distribution plot
- All plots saved to results/plots/ directory
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Any, Optional


class ExperimentVisualizer:
    """
    Generates publication-quality visualization for DDoS detection experiments.
    
    Uses consistent styling, color schemes, and formatting appropriate for
    academic papers and reports.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the visualizer.
        
        Parameters
        ----------
        config : dict
            Visualization configuration:
            - 'figure_dpi': int, resolution in dots per inch
            - 'plot_style': str, matplotlib/seaborn style name
            - 'color_palette': list of hex color codes
            - 'save_figures': bool
            - 'output_dir': base output directory
        """
        if config is None:
            config = {
                'figure_dpi': 300,
                'plot_style': 'seaborn-v0_8',
                'color_palette': ['#2c7bb0', '#f03b20', '#91cf60'],
                'save_figures': True,
                'output_dir': 'results/plots'
            }
        self.config = config
        self.output_dir = config.get('output_dir', 'results/plots')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set style
        plt.style.use(config.get('plot_style', 'seaborn-v0_8'))
        sns.set_palette(config.get('color_palette', ['#2c7bb0', '#f03b20', '#91cf60']))
    
    def plot_method_comparison_bar(self, 
                                   comparison_results: Dict[str, Dict[str, float]],
                                   metrics: Optional[List[str]] = None,
                                   title: str = "KNN Method Comparison",
                                   save_name: str = "method_comparison.png") -> None:
        """
        Bar chart comparing KNN methods across specified metrics.
        
        Parameters
        ----------
        comparison_results : dict
            Dictionary mapping method names to their metric dicts
        metrics : list, optional
            Which metrics to plot. If None, uses default set
        title : str
            Plot title
        save_name : str
            Filename to save (within output_dir)
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
        
        # Extract data
        methods = list(comparison_results.keys())
        n_metrics = len(metrics)
        n_methods = len(methods)
        
        # Build data matrix: methods x metrics
        data_matrix = np.zeros((n_methods, n_metrics))
        metric_names = []
        
        for i, metric in enumerate(metrics):
            metric_names.append(metric.upper())
            for j, method in enumerate(methods):
                data_matrix[j, i] = comparison_results[method].get(metric, 0.0)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot bars
        x = np.arange(n_methods)
        width = 0.15  # Reduced width for multiple metrics per method
        
        for i, metric in enumerate(metrics):
            offsets = x + (i - n_metrics / 2 + 0.5) * width
            bars = ax.bar(offsets, data_matrix[:, i], width, 
                         label=metric, alpha=0.85)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                if height > 0.05:  # Only label if sufficiently high
                    ax.annotate(f'{height:.3f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=8)
        
        # Formatting
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=11)
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim(0, 1.15)  # Scores are in [0, 1]
        ax.grid(axis='y', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=self.config.get('figure_dpi', 300), 
                   bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"  Saved bar comparison: {save_path}")
    
    def plot_roc_curve(self, 
                       all_fpr: List[np.ndarray],
                       all_tpr: List[np.ndarray],
                       all_labels: List[str],
                       auc_scores: Optional[List[float]] = None,
                       title: str = "ROC Curve Comparison",
                       save_name: str = "roc_curve.png") -> None:
        """
        Plot ROC curves for multiple methods.
        
        Parameters
        ----------
        all_fpr : list of np.ndarray
            False Positive Rates for each method
        all_tpr : list of np.ndarray
            True Positive Rates for each method
        all_labels : list of str
            Method labels for legend
        auc_scores : list, optional
            AUC-ROC scores for each method
        title : str
            Plot title
        save_name : str
            Filename to save
        """
        if auc_scores is None:
            auc_scores = [np.trapz(tpr, fpr) for fpr, tpr in zip(all_fpr, all_tpr)]
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot chance line
        ax.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.5)', alpha=0.5)
        
        # Plot each method's ROC curve
        for i, (fpr, tpr, label, auc_score) in enumerate(zip(all_fpr, all_tpr, all_labels, auc_scores)):
            ax.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {auc_score:.3f})',
                   color=self.config.get('color_palette', ['#2c7bb0', '#f03b20', '#91cf60'])[i % 3])
        
        # Formatting
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=self.config.get('figure_dpi', 300),
                   bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"  Saved ROC curve: {save_path}")
    
    def plot_confusion_matrix(self, 
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              method_name: str = "Method",
                              title: str = "Confusion Matrix",
                              save_name: str = "confusion_matrix.png") -> None:
        """
        Plot confusion matrix heatmap.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        method_name : str
            Method name for title
        title : str
            Plot title
        save_name : str
            Filename to save
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'],
                   ax=ax, cbar_kws={'label': 'Count'})
        
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_title(f'{title}\n({method_name})', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=self.config.get('figure_dpi', 300),
                   bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"  Saved confusion matrix: {save_path}")
    
    def plot_k_vs_accuracy(self, 
                           k_values: List[int],
                           accuracies: List[float],
                           f1_scores: Optional[List[float]] = None,
                           title: str = "K-value vs Detection Performance",
                           save_name: str = "k_vs_accuracy.png") -> None:
        """
        Line plot showing K value vs. detection performance.
        
        Parameters
        ----------
        k_values : list
            List of K values tested
        accuracies : list
            Accuracy for each K value
        f1_scores : list, optional
            F1-score for each K value
        title : str
            Plot title
        save_name : str
            Filename to save
        """
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot accuracy
        line1 = ax1.plot(k_values, accuracies, 'o-', 
                        color=self.config.get('color_palette', ['#2c7bb0'])[0],
                        label='Accuracy', linewidth=2, markersize=8)
        ax1.set_xlabel('K value', fontsize=12)
        ax1.set_ylabel('Accuracy', color=line1[0].get_color(), fontsize=12)
        ax1.set_ylim(0, 1.0)
        ax1.tick_params(axis='y', labelcolor=line1[0].get_color())
        ax1.grid(alpha=0.3, linestyle='--')
        
        # Optional: F1 score on secondary axis
        if f1_scores is not None:
            ax2 = ax1.twinx()
            line2 = ax2.plot(k_values, f1_scores, 's-',
                            color=self.config.get('color_palette', ['#f03b20'])[1],
                            label='F1-score', linewidth=2, markersize=8)
            ax2.set_ylabel('F1-score', color=line2[0].get_color(), fontsize=12)
            ax2.set_ylim(0, 1.0)
            ax2.tick_params(axis='y', labelcolor=line2[0].get_color())
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='lower right', fontsize=10)
        else:
            ax1.legend(['Accuracy'], loc='lower right', fontsize=10)
        
        ax1.set_xticks(k_values)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=self.config.get('figure_dpi', 300),
                   bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"  Saved K-vs-accuracy plot: {save_path}")
    
    def plot_degree_distribution(self, 
                                  normal_degrees: np.ndarray,
                                  attack_degrees: np.ndarray,
                                  title: str = "Degree-of-Attack Distribution",
                                  save_name: str = "degree_distribution.png") -> None:
        """
        Plot degree-of-attack distribution for normal vs attack flows.
        
        Parameters
        ----------
        normal_degrees : np.ndarray
            Degree values for normal flows
        attack_degrees : np.ndarray
            Degree values for attack flows
        title : str
            Plot title
        save_name : str
            Filename to save
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histograms
        n_bins = 50
        ax.hist(normal_degrees, bins=n_bins, alpha=0.6, 
               label=f'Normal ({len(normal_degrees)} flows)', 
               color=self.config.get('color_palette', ['#2c7bb0'])[0],
               density=True)
        ax.hist(attack_degrees, bins=n_bins, alpha=0.6,
               label=f'Attack ({len(attack_degrees)} flows)', 
               color=self.config.get('color_palette', ['#f03b20'])[1],
               density=True)
        
        # Add vertical lines at key thresholds
        for threshold in [0.3, 0.5, 0.7]:
            ax.axvline(x=threshold, color='gray', linestyle='--', alpha=0.5,
                      label=f'Threshold {threshold}')
            ax.axvline(x=np.percentile(normal_degrees, 75), color='blue', 
                      linestyle=':', alpha=0.7, label='Q3 normal')
            ax.axvline(x=np.percentile(attack_degrees, 25), color='red',
                      linestyle=':', alpha=0.7, label='Q1 attack')
        
        # Formatting
        ax.set_xlabel('Degree of DDoS Attack', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, linestyle='--')
        
        # Add statistics box
        normal_mean = np.mean(normal_degrees)
        attack_mean = np.mean(attack_degrees)
        gap = attack_mean - normal_mean
        stats_text = f'Normal mean: {normal_mean:.3f}\nAttack mean: {attack_mean:.3f}\nGap: {gap:.3f}'
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=self.config.get('figure_dpi', 300),
                   bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"  Saved degree distribution: {save_path}")
    
    def plot_experiment_results_summary(self, 
                                        trial_results: List[Dict[str, Any]],
                                        metric: str = 'accuracy',
                                        title: str = "Experiment Results Summary",
                                        save_name: str = "experiment_results_summary.png") -> None:
        """
        Plot summary of experiment results across multiple trials.
        
        Parameters
        ----------
        trial_results : list
            List of trial result dictionaries
        metric : str
            Metric to plot (accuracy, f1, auc_roc, etc.)
        title : str
            Plot title
        save_name : str
            Filename to save
        """
        # Extract mean and std across trials
        means = []
        stds = []
        seeds = []
        
        for result in trial_results:
            # Get the metric value (try different key names)
            val = result.get(metric, result.get(f'{metric}_mean', 0))
            se = result.get(f'{metric}_std', result.get(f'{metric}_std', 0))
            
            means.append(float(val) if isinstance(val, (int, float)) else 0.0)
            stds.append(float(se) if isinstance(se, (int, float)) else 0.0)
            # Extract seed if available
            seeds.append(result.get('trial_seed', result.get('random_seed', 0)))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot individual trial results (semi-transparent lines)
        for i, (mean, std, seed) in enumerate(zip(means, stds, seeds)):
            # Generate x-axis (trial numbers around this value)
            ax.plot([seed-0.5, seed+0.5], [mean, mean], 'o-', 
                   alpha=0.3, color='#aaaaaa', linewidth=1)
        
        # Plot mean line with error bars
        x_centers = np.arange(1, len(means) + 1)
        ax.errorbar(x_centers, means, yerr=stds, fmt='o-', 
                   color=self.config.get('color_palette', ['#2c7bb0'])[0],
                   linewidth=2, markersize=8, capsize=5,
                   label=f'Mean {metric.upper()}')
        
        # Formatting
        ax.set_xlabel('Trial (different random seed)', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x_centers)
        ax.set_ylim(0, 1.0 if metric in ['accuracy', 'f1', 'auc_roc'] else None)
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(fontsize=10)
        
        # Add summary stats
        overall_mean = np.mean(means)
        overall_std = np.mean(stds)
        ax.text(0.95, 0.95, f'Mean: {overall_mean:.3f} +/- {overall_std:.3f}',
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=self.config.get('figure_dpi', 300),
                   bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"  Saved experiment summary: {save_path}")
    
    def plot_table(self, 
                   data: Dict[str, Any], 
                   title: str = "Results Table",
                   save_name: str = "results_table.png") -> None:
        """
        Plot a simple table image from dictionary data.
        
        Parameters
        ----------
        data : dict
            Key-value pairs to display in table
        title : str
            Table title
        save_name : str
            Filename to save
        """
        # Create figure with table
        fig, ax = plt.subplots(figsize=(8, max(3, len(data) * 0.3 + 1)))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = [[k, f"{v:.4f}" if isinstance(v, float) else str(v)] 
                     for k, v in data.items()]
        
        # Add title
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add table
        table = ax.table(cellText=table_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0.1, 1, 0.9])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=self.config.get('figure_dpi', 300),
                   bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"  Saved results table: {save_path}")


# Convenience functions

def create_visualizer(config: Dict[str, Any] = None) -> ExperimentVisualizer:
    """Create an ExperimentVisualizer from configuration."""
    return ExperimentVisualizer(config=config)


def quick_plot_comparison(comparison_results: Dict[str, Dict[str, float]],
                         save_path: str = "method_comparison.png") -> None:
    """
    Quick bar chart comparison of methods.
    
    Parameters
    ----------
    comparison_results : dict
        Method -> metrics mapping
    save_path : str
        Path to save the figure
    """
    visualizer = create_visualizer()
    visualizer.plot_method_comparison_bar(comparison_results, save_name=save_path)


def quick_plot_roc(all_fpr, all_tpr, all_labels, auc_scores=None, save_path="roc_curve.png") -> None:
    """Quick ROC curve plot."""
    visualizer = create_visualizer()
    visualizer.plot_roc_curve(all_fpr, all_tpr, all_labels, auc_scores, save_name=save_path)


# Auto-run test
if __name__ == "__main__":
    print("Testing visualization module...")
    
    visualizer = create_visualizer()
    
    # Test bar comparison
    comparison = {
        'Baseline': {'accuracy': 0.85, 'precision': 0.83, 'recall': 0.80, 'f1': 0.81, 'auc_roc': 0.87},
        'Improved': {'accuracy': 0.88, 'precision': 0.86, 'recall': 0.84, 'f1': 0.85, 'auc_roc': 0.90},
        'Proposed': {'accuracy': 0.91, 'precision': 0.89, 'recall': 0.87, 'f1': 0.88, 'auc_roc': 0.93}
    }
    
    visualizer.plot_method_comparison_bar(comparison, save_name="test_comparison.png")
    
    # Test ROC curve (mock data)
    import numpy as np
    fprs = [np.linspace(0, 1, 100) for _ in range(3)]
    tprs = [np.linspace(0, 1, 100) for _ in range(3)]
    visualizer.plot_roc_curve(fprs, tprs, ['Baseline', 'Improved', 'Proposed'],
                             auc_scores=[0.87, 0.90, 0.93], save_name="test_roc.png")
    
    # Test confusion matrix
    y_true = np.array([0]*140 + [1]*60)
    y_pred = np.array([0]*120 + [1]*20 + [1]*40 + [0]*20)  # 85% accuracy
    visualizer.plot_confusion_matrix(y_true, y_pred, method_name="Test", 
                                    save_name="test_cm.png")
    
    # Test K-vs-accuracy
    k_vals = [1, 3, 5, 7, 10, 15, 20]
    accs = [0.78, 0.82, 0.85, 0.84, 0.83, 0.81, 0.79]
    visualizer.plot_k_vs_accuracy(k_vals, accs, save_name="test_kva.png")
    
    # Test degree distribution
    normal_deg = np.random.beta(5, 2, 100)  # Normal: skewed low
    attack_deg = np.random.beta(2, 5, 60) + 0.3  # Attack: skewed high
    visualizer.plot_degree_distribution(normal_deg, attack_deg, 
                                       save_name="test_ddist.png")
    
    print("\n[PASS] All visualization tests passed!")