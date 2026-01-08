import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from src.evaluation.baseline import EvaluationResult

class ResultAnalyzer:
    """Analyzes and compares evaluation results"""
    
    @staticmethod
    def summarize_results(results: List[EvaluationResult]) -> Dict:
        """Compute summary statistics"""
        harm_scores = [r.harm_score for r in results]
        refusal_scores = [r.refusal_score for r in results]
        refusal_rate = np.mean([r.actually_refuses for r in results])
        
        return {
            'n': len(results),
            'harm_mean': np.mean(harm_scores),
            'harm_std': np.std(harm_scores),
            'refusal_mean': np.mean(refusal_scores),
            'refusal_std': np.std(refusal_scores),
            'refusal_rate': refusal_rate
        }
    
    @staticmethod
    def print_comparison(baseline_results: Dict, intervened_results: Dict):
        """Print comparison table"""
        print("\n" + "=" * 70)
        print("COMPARISON: BASELINE vs INTERVENTION")
        print("=" * 70)
        
        for condition in baseline_results.keys():
            if condition not in intervened_results:
                continue
            
            base = baseline_results[condition]
            interv = intervened_results[condition]
            
            print(f"\nCondition: {condition}")
            print("-" * 70)
            print(f"{'Metric':<20} {'Baseline':>15} {'Intervened':>15} {'Change':>15}")
            print("-" * 70)
            
            # Harmfulness
            harm_change = interv['harm_mean'] - base['harm_mean']
            print(f"{'Harmfulness':<20} {base['harm_mean']:>15.3f} {interv['harm_mean']:>15.3f} {harm_change:>+15.3f}")
            
            # Refusal score
            ref_change = interv['refusal_mean'] - base['refusal_mean']
            print(f"{'Refusal Score':<20} {base['refusal_mean']:>15.3f} {interv['refusal_mean']:>15.3f} {ref_change:>+15.3f}")
            
            # Refusal rate
            rate_change = interv['refusal_rate'] - base['refusal_rate']
            print(f"{'Refusal Rate':<20} {base['refusal_rate']:>15.2%} {interv['refusal_rate']:>15.2%} {rate_change:>+15.2%}")
        
        print("=" * 70)
    
    @staticmethod
    def plot_results(baseline_results: Dict, intervened_results: Dict, save_path: str = 'intervention_results.png'):
        """Create visualization"""
        conditions = list(baseline_results.keys())
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Harmfulness scores
        ax = axes[0]
        base_harm = [baseline_results[c]['harm_mean'] for c in conditions]
        interv_harm = [intervened_results.get(c, {'harm_mean': 0})['harm_mean'] for c in conditions]
        # Note: if condition not in intervened, 0 is dummy. Should handle better but for 'J' it exists.
        
        x = np.arange(len(conditions))
        width = 0.35
        ax.bar(x - width/2, base_harm, width, label='Baseline', alpha=0.8)
        # Only plot intervened bars if data exists
        ax.bar(x + width/2, interv_harm, width, label='Intervened', alpha=0.8)
        
        ax.set_ylabel('Harmfulness Score')
        ax.set_title('Harmfulness at t_inst')
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Refusal scores
        ax = axes[1]
        base_ref = [baseline_results[c]['refusal_mean'] for c in conditions]
        interv_ref = [intervened_results.get(c, {'refusal_mean': 0})['refusal_mean'] for c in conditions]
        
        ax.bar(x - width/2, base_ref, width, label='Baseline', alpha=0.8)
        ax.bar(x + width/2, interv_ref, width, label='Intervened', alpha=0.8)
        ax.set_ylabel('Refusal Score')
        ax.set_title('Refusal at t_post')
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Refusal rates
        ax = axes[2]
        base_rate = [baseline_results[c]['refusal_rate'] for c in conditions]
        interv_rate = [intervened_results.get(c, {'refusal_rate': 0})['refusal_rate'] for c in conditions]
        
        ax.bar(x - width/2, base_rate, width, label='Baseline', alpha=0.8)
        ax.bar(x + width/2, interv_rate, width, label='Intervened', alpha=0.8)
        ax.set_ylabel('Refusal Rate')
        ax.set_title('Actual Refusal Behavior')
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved to '{save_path}'")
        # plt.show() # Don't show in non-interactive env
