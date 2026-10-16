"""
Statistical Testing Module
Paired statistical tests with effect sizes and multiple comparison corrections
for rigorous comparison of ML methods across multiple trials.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from scipy.stats import (ttest_rel, wilcoxon, friedmanchisquare, 
                         kruskal, mannwhitneyu, norm)
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')


def cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cohen's d for paired samples.
    
    d = mean(x - y) / std(x - y)
    """
    diff = x - y
    return np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)


def hedges_g_paired(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Hedges' g for paired samples (bias-corrected Cohen's d).
    """
    n = len(x)
    d = cohens_d_paired(x, y)
    # Correction factor
    J = 1 - 3 / (4 * n - 9)
    return J * d


def glass_delta_paired(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Glass's delta (using control group SD).
    """
    diff = x - y
    return np.mean(diff) / (np.std(y, ddof=1) + 1e-10)


def cliff_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cliff's delta (non-parametric effect size).
    
    delta = P(X > Y) - P(X < Y)
    """
    n1, n2 = len(x), len(y)
    greater = sum(1 for xi in x for yj in y if xi > yj)
    less = sum(1 for xi in x for yj in y if xi < yj)
    return (greater - less) / (n1 * n2)


def vargha_delaney_a12(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Vargha-Delaney A12 (probability that X > Y).
    
    A12 = 0.5 means no difference
    A12 > 0.5 means X tends to be larger
    """
    n1, n2 = len(x), len(y)
    greater = sum(1 for xi in x for yj in y if xi > yj)
    equal = sum(1 for xi in x for yj in y if xi == yj)
    return (greater + 0.5 * equal) / (n1 * n2)


def paired_ttest(x: np.ndarray, y: np.ndarray, alternative: str = 'two-sided') -> Dict[str, Any]:
    """
    Paired t-test with effect sizes.
    
    Parameters
    ----------
    x, y : np.ndarray
        Paired samples
    alternative : str
        'two-sided', 'greater', 'less'
        
    Returns
    -------
    dict
        Test results including p-value, effect sizes
    """
    # Remove NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    
    if len(x) < 2:
        return {'error': 'Insufficient data'}
    
    # Paired t-test
    stat, p_val = ttest_rel(x, y, alternative=alternative)
    
    # Effect sizes
    d = cohens_d_paired(x, y)
    g = hedges_g_paired(x, y)
    delta = glass_delta_paired(x, y)
    
    # Confidence interval for mean difference
    diff = x - y
    mean_diff = np.mean(diff)
    se_diff = np.std(diff, ddof=1) / np.sqrt(len(diff))
    ci_low = mean_diff - 1.96 * se_diff
    ci_high = mean_diff + 1.96 * se_diff
    
    return {
        'test': 'paired_ttest',
        'statistic': float(stat),
        'p_value': float(p_val),
        'mean_diff': float(mean_diff),
        'ci_95': (float(ci_low), float(ci_high)),
        'cohens_d': float(d),
        'hedges_g': float(g),
        'glass_delta': float(delta),
        'n': len(x),
        'alternative': alternative
    }


def paired_wilcoxon(x: np.ndarray, y: np.ndarray, alternative: str = 'two-sided') -> Dict[str, Any]:
    """
    Wilcoxon signed-rank test (non-parametric paired test).
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    
    if len(x) < 2:
        return {'error': 'Insufficient data'}
    
    try:
        stat, p_val = wilcoxon(x, y, alternative=alternative, zero_method='wilcox')
    except ValueError:
        # All differences zero
        return {
            'test': 'wilcoxon',
            'statistic': 0,
            'p_value': 1.0,
            'cliff_delta': 0.0,
            'vargha_delaney_a12': 0.5,
            'n': len(x)
        }
    
    # Non-parametric effect sizes
    cliff_d = cliff_delta(x, y)
    vd_a12 = vargha_delaney_a12(x, y)
    
    return {
        'test': 'wilcoxon',
        'statistic': float(stat),
        'p_value': float(p_val),
        'cliff_delta': float(cliff_d),
        'vargha_delaney_a12': float(vd_a12),
        'n': len(x),
        'alternative': alternative
    }


def friedman_test(*arrays: np.ndarray) -> Dict[str, Any]:
    """
    Friedman test for multiple related samples (non-parametric ANOVA).
    """
    # Check all arrays have same length
    lengths = [len(arr) for arr in arrays]
    if len(set(lengths)) > 1:
        return {'error': 'All arrays must have same length'}
    
    # Remove any rows with NaN
    stacked = np.column_stack(arrays)
    mask = ~np.any(np.isnan(stacked), axis=1)
    stacked = stacked[mask]
    
    if stacked.shape[0] < 2:
        return {'error': 'Insufficient data after NaN removal'}
    
    try:
        stat, p_val = friedmanchisquare(*[stacked[:, i] for i in range(stacked.shape[1])])
    except Exception as e:
        return {'error': str(e)}
    
    return {
        'test': 'friedman',
        'statistic': float(stat),
        'p_value': float(p_val),
        'n_samples': stacked.shape[0],
        'n_groups': stacked.shape[1]
    }


def nemenyi_posthoc(ranks: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Nemenyi post-hoc test after Friedman.
    
    Parameters
    ----------
    ranks : np.ndarray
        Average ranks per group (n_groups,)
    alpha : float
        Significance level
        
    Returns
    -------
    dict
        Pairwise comparisons with critical difference
    """
    k = len(ranks)  # number of groups
    n = 1  # This would need the actual N (number of blocks)
    # For proper Nemenyi, we need N from Friedman test
    # This is a simplified version
    
    from scipy.stats import studentized_range
    
    # Critical difference (approximate)
    q_alpha = studentized_range.ppf(1 - alpha, k, float('inf'))
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))  # n should be number of datasets/trials
    
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(ranks[i] - ranks[j])
            pairs.append({
                'group_i': i,
                'group_j': j,
                'rank_diff': diff,
                'significant': diff > cd
            })
    
    return {
        'critical_difference': float(cd),
        'pairs': pairs
    }


def multiple_comparison_correction(p_values: List[float], 
                                    method: str = 'holm',
                                    alpha: float = 0.05) -> Dict[str, Any]:
    """
    Apply multiple comparison correction.
    
    Parameters
    ----------
    p_values : list
        List of p-values
    method : str
        'holm', 'bonferroni', 'fdr_bh' (Benjamini-Hochberg), 'fdr_by'
    alpha : float
        Family-wise error rate
        
    Returns
    -------
    dict
        Corrected p-values and significance
    """
    if not p_values:
        return {'error': 'No p-values provided'}
    
    reject, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method=method)
    
    return {
        'method': method,
        'alpha': alpha,
        'original_p': p_values,
        'corrected_p': p_corrected.tolist(),
        'reject': reject.tolist(),
        'n_comparisons': len(p_values)
    }


def compare_methods_pairwise(results_dict: Dict[str, Dict],
                              metric: str = 'f1',
                              test: str = 'wilcoxon',
                              correction: str = 'holm') -> pd.DataFrame:
    """
    Pairwise comparison of all methods on a specific metric.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping method names to trial results
        Each value should have 'trials' list with metric values
    metric : str
        Metric to compare ('accuracy', 'f1', 'auc_roc', etc.)
    test : str
        'ttest' or 'wilcoxon'
    correction : str
        Multiple comparison correction method
        
    Returns
    -------
    pd.DataFrame
        Pairwise comparison results
    """
    methods = list(results_dict.keys())
    n_methods = len(methods)
    
    # Extract metric values per trial for each method
    trial_data = {}
    for method, result in results_dict.items():
        if 'trials' in result:
            values = [t.get(metric, 0) for t in result['trials']]
            trial_data[method] = np.array(values)
        elif f'{metric}_mean' in result:
            # Single trial case - use mean/std to simulate
            mean = result.get(f'{metric}_mean', 0)
            std = result.get(f'{metric}_std', 0)
            n = result.get('num_trials', 10)
            trial_data[method] = np.random.normal(mean, std, n)
    
    # Pairwise comparisons
    comparisons = []
    p_values = []
    
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            m1, m2 = methods[i], methods[j]
            x, y = trial_data[m1], trial_data[m2]
            
            if test == 'ttest':
                res = paired_ttest(x, y)
            else:
                res = paired_wilcoxon(x, y)
            
            if 'error' not in res:
                comp = {
                    'method_1': m1,
                    'method_2': m2,
                    'test': test,
                    'statistic': res.get('statistic', 0),
                    'p_value': res.get('p_value', 1),
                    'mean_diff': np.mean(x) - np.mean(y),
                    'cohens_d': res.get('cohens_d', res.get('cliff_delta', 0)),
                    'hedges_g': res.get('hedges_g', 0),
                    'cliff_delta': res.get('cliff_delta', 0),
                    'vargha_delaney_a12': res.get('vargha_delaney_a12', 0.5)
                }
                comparisons.append(comp)
                p_values.append(res.get('p_value', 1))
    
    # Apply multiple comparison correction
    if p_values:
        corr_result = multiple_comparison_correction(p_values, method=correction)
        for i, comp in enumerate(comparisons):
            comp['p_corrected'] = corr_result['corrected_p'][i]
            comp['significant'] = corr_result['reject'][i]
    
    df = pd.DataFrame(comparisons)
    if len(df) > 0:
        df = df.sort_values('p_corrected' if 'p_corrected' in df.columns else 'p_value')
    
    return df


def compare_all_metrics(results_dict: Dict[str, Dict],
                         metrics: List[str] = None,
                         test: str = 'wilcoxon',
                         correction: str = 'holm') -> Dict[str, pd.DataFrame]:
    """
    Run pairwise comparisons for all metrics.
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    
    all_comparisons = {}
    for metric in metrics:
        try:
            df = compare_methods_pairwise(results_dict, metric, test, correction)
            if len(df) > 0:
                all_comparisons[metric] = df
        except Exception as e:
            print(f"Warning: Failed to compare {metric}: {e}")
    
    return all_comparisons


def generate_statistical_report(results_dict: Dict[str, Dict],
                                 baseline_method: str = None) -> str:
    """
    Generate a comprehensive statistical report.
    
    Parameters
    ----------
    results_dict : dict
        Method -> trial results
    baseline_method : str, optional
        Method to compare all others against (if None, pairwise all)
        
    Returns
    -------
    str
        Formatted statistical report
    """
    lines = []
    lines.append("=" * 80)
    lines.append("STATISTICAL COMPARISON REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    methods = list(results_dict.keys())
    lines.append(f"Methods compared: {', '.join(methods)}")
    lines.append("")
    
    # Descriptive statistics
    lines.append("-" * 80)
    lines.append("DESCRIPTIVE STATISTICS (mean ± std across trials)")
    lines.append("-" * 80)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
        lines.append(f"\n{metric.upper()}:")
        for method in methods:
            result = results_dict[method]
            mean = result.get(f'{metric}_mean', 0)
            std = result.get(f'{metric}_std', 0)
            lines.append(f"  {method:20s}: {mean:.4f} ± {std:.4f}")
    
    # Pairwise comparisons
    lines.append("\n" + "-" * 80)
    lines.append("PAIRWISE STATISTICAL TESTS (Wilcoxon signed-rank, Holm-corrected)")
    lines.append("-" * 80)
    
    for metric in ['accuracy', 'f1', 'auc_roc']:
        df = compare_methods_pairwise(results_dict, metric, 'wilcoxon', 'holm')
        if len(df) > 0:
            lines.append(f"\n{metric.upper()}:")
            for _, row in df.iterrows():
                sig = "***" if row.get('p_corrected', 1) < 0.001 else \
                      "**" if row.get('p_corrected', 1) < 0.01 else \
                      "*" if row.get('p_corrected', 1) < 0.05 else "ns"
                lines.append(f"  {row['method_1']} vs {row['method_2']}: "
                           f"p={row.get('p_corrected', row['p_value']):.4f} {sig}, "
                           f"Δ={row['mean_diff']:+.4f}, "
                           f"d={row['cohens_d']:.3f}")
    
    # Friedman test (if 3+ methods)
    if len(methods) >= 3:
        lines.append("\n" + "-" * 80)
        lines.append("FRIEDMAN TEST (Omnibus test for multiple methods)")
        lines.append("-" * 80)
        
        for metric in ['accuracy', 'f1', 'auc_roc']:
            trial_arrays = []
            for method in methods:
                result = results_dict[method]
                if 'trials' in result:
                    values = [t.get(metric, 0) for t in result['trials']]
                    trial_arrays.append(np.array(values))
            
            if len(trial_arrays) >= 3 and all(len(arr) == len(trial_arrays[0]) for arr in trial_arrays):
                friedman = friedman_test(*trial_arrays)
                if 'error' not in friedman:
                    lines.append(f"  {metric}: χ²={friedman['statistic']:.3f}, "
                               f"p={friedman['p_value']:.4f}, "
                               f"n={friedman['n_samples']}, k={friedman['n_groups']}")
    
    lines.append("\n" + "=" * 80)
    lines.append("INTERPRETATION GUIDE:")
    lines.append("  p < 0.05: Significant difference (after correction)")
    lines.append("  Cohen's d: 0.2=small, 0.5=medium, 0.8=large")
    lines.append("  Cliff's δ: 0.147=small, 0.33=medium, 0.474=large")
    lines.append("  Vargha-Delaney A12: 0.56=small, 0.64=medium, 0.71=large")
    lines.append("=" * 80)
    
    return "\n".lines()


def bootstrap_confidence_interval(data: np.ndarray, 
                                   n_bootstrap: int = 10000,
                                   confidence: float = 0.95,
                                   statistic: callable = np.mean) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for any statistic.
    """
    if len(data) == 0:
        return (0, 0)
    
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_stats.append(statistic(sample))
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(boot_stats, alpha * 100)
    upper = np.percentile(boot_stats, (1 - alpha) * 100)
    
    return (float(lower), float(upper))


def permutation_test(x: np.ndarray, y: np.ndarray,
                      n_permutations: int = 10000,
                      statistic: callable = lambda a, b: np.mean(a) - np.mean(b),
                      alternative: str = 'two-sided') -> Dict[str, Any]:
    """
    Permutation test for paired samples.
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    
    observed = statistic(x, y)
    
    # Permute signs of differences
    diffs = x - y
    perm_stats = []
    
    for _ in range(n_permutations):
        signs = np.random.choice([-1, 1], size=len(diffs))
        perm_diff = diffs * signs
        perm_stats.append(statistic(perm_diff + y, y))  # Reconstruct
    
    perm_stats = np.array(perm_stats)
    
    if alternative == 'two-sided':
        p_val = np.mean(np.abs(perm_stats) >= np.abs(observed))
    elif alternative == 'greater':
        p_val = np.mean(perm_stats >= observed)
    else:
        p_val = np.mean(perm_stats <= observed)
    
    return {
        'test': 'permutation',
        'observed_statistic': float(observed),
        'p_value': float(p_val),
        'n_permutations': n_permutations
    }


# =============================================================================
# Trial-Level Analysis
# =============================================================================

def analyze_trial_stability(results_dict: Dict[str, Dict],
                             metric: str = 'f1') -> pd.DataFrame:
    """
    Analyze stability of results across trials.
    """
    rows = []
    
    for method, result in results_dict.items():
        if 'trials' not in result:
            continue
        
        values = np.array([t.get(metric, 0) for t in result['trials']])
        
        row = {
            'method': method,
            'mean': np.mean(values),
            'std': np.std(values, ddof=1),
            'cv': np.std(values, ddof=1) / (np.mean(values) + 1e-10),  # Coefficient of variation
            'min': np.min(values),
            'max': np.max(values),
            'range': np.max(values) - np.min(values),
            'iqr': np.percentile(values, 75) - np.percentile(values, 25),
            'n_trials': len(values)
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values('cv')  # Most stable first
    
    return df


def compute_win_tie_loss(results_dict: Dict[str, Dict],
                          metric: str = 'f1') -> pd.DataFrame:
    """
    Compute win/tie/loss matrix between all method pairs.
    """
    methods = list(results_dict.keys())
    n = len(methods)
    
    # Get trial data
    trial_data = {}
    for method, result in results_dict.items():
        if 'trials' in result:
            trial_data[method] = np.array([t.get(metric, 0) for t in result['trials']])
    
    # Pairwise win/tie/loss
    wtl = np.zeros((n, n, 3))  # [win, tie, loss]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x = trial_data[methods[i]]
            y = trial_data[methods[j]]
            
            wins = np.sum(x > y + 1e-10)
            ties = np.sum(np.abs(x - y) <= 1e-10)
            losses = np.sum(x < y - 1e-10)
            
            wtl[i, j] = [wins, ties, losses]
    
    # Create DataFrame
    df = pd.DataFrame(index=methods, columns=methods, dtype=object)
    for i in range(n):
        for j in range(n):
            if i == j:
                df.iloc[i, j] = "—"
            else:
                w, t, l = wtl[i, j]
                df.iloc[i, j] = f"{int(w)}-{int(t)}-{int(l)}"
    
    return df


if __name__ == "__main__":
    print("Testing statistical testing module...")
    
    # Create synthetic trial data for 3 methods
    np.random.seed(42)
    n_trials = 20
    
    # Method A: Best
    results = {
        'Proposed': {
            'trials': [{'accuracy': 0.91 + np.random.normal(0, 0.01),
                       'f1': 0.90 + np.random.normal(0, 0.015),
                       'auc_roc': 0.93 + np.random.normal(0, 0.01)}
                      for _ in range(n_trials)]
        },
        'Improved': {
            'trials': [{'accuracy': 0.88 + np.random.normal(0, 0.012),
                       'f1': 0.87 + np.random.normal(0, 0.015),
                       'auc_roc': 0.90 + np.random.normal(0, 0.012)}
                      for _ in range(n_trials)]
        },
        'Baseline': {
            'trials': [{'accuracy': 0.85 + np.random.normal(0, 0.015),
                       'f1': 0.84 + np.random.normal(0, 0.02),
                       'auc_roc': 0.87 + np.random.normal(0, 0.015)}
                      for _ in range(n_trials)]
        }
    }
    
    # Compute means/stds
    for method, result in results.items():
        for metric in ['accuracy', 'f1', 'auc_roc']:
            values = [t[metric] for t in result['trials']]
            result[f'{metric}_mean'] = np.mean(values)
            result[f'{metric}_std'] = np.std(values, ddof=1)
    
    # Test pairwise comparison
    print("\n1. Pairwise comparison (F1):")
    df = compare_methods_pairwise(results, 'f1', 'wilcoxon', 'holm')
    print(df[['method_1', 'method_2', 'p_value', 'p_corrected', 'mean_diff', 'cohens_d', 'significant']].to_string(index=False))
    
    # Test all metrics
    print("\n2. All metrics comparison:")
    all_comps = compare_all_metrics(results)
    for metric, df in all_comps.items():
        print(f"\n{metric}:")
        print(df[['method_1', 'method_2', 'p_corrected', 'mean_diff', 'cohens_d']].to_string(index=False))
    
    # Test Friedman
    print("\n3. Friedman test:")
    friedman = friedman_test(
        np.array([t['f1'] for t in results['Proposed']['trials']]),
        np.array([t['f1'] for t in results['Improved']['trials']]),
        np.array([t['f1'] for t in results['Baseline']['trials']])
    )
    print(f"  χ²={friedman['statistic']:.3f}, p={friedman['p_value']:.4f}")
    
    # Test stability analysis
    print("\n4. Trial stability:")
    stability = analyze_trial_stability(results, 'f1')
    print(stability.to_string(index=False))
    
    # Test win/tie/loss
    print("\n5. Win/Tie/Loss matrix:")
    wtl = compute_win_tie_loss(results, 'f1')
    print(wtl.to_string())
    
    # Test effect sizes
    print("\n6. Effect sizes (Proposed vs Baseline):")
    x = np.array([t['f1'] for t in results['Proposed']['trials']])
    y = np.array([t['f1'] for t in results['Baseline']['trials']])
    
    d = cohens_d_paired(x, y)
    g = hedges_g_paired(x, y)
    cliff = cliff_delta(x, y)
    vd = vargha_delaney_a12(x, y)
    
    print(f"  Cohen's d: {d:.3f}")
    print(f"  Hedges' g: {g:.3f}")
    print(f"  Cliff's δ: {cliff:.3f}")
    print(f"  VD A12: {vd:.3f}")
    
    print("\n✓ Statistical testing complete!")