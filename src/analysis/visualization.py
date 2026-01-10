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

    @staticmethod
    def plot_attention_contributions(results: Dict[str, List[float]], save_path: str = 'attention_analysis.png'):
        """
        Plot distribution of attention mass (Instruction vs Suffix vs System)
        
        Args:
            results: Dictionary with keys 'instruction_attn', 'suffix_attn', 'system_attn'
                     containing lists of float values (one per sample)
            save_path: Path to save plot
        """
        # Calculate means and stds
        means = {k: np.mean(v) for k, v in results.items()}
        stds = {k: np.std(v) for k, v in results.items()}
        
        labels = ['Instruction', 'Adv Suffix', 'System/Other']
        keys = ['instruction_attn', 'suffix_attn', 'system_attn']
        
        vals = [means[k] for k in keys]
        errs = [stds[k] for k in keys]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x = np.arange(len(labels))
        bars = ax.bar(x, vals, yerr=errs, capsize=5, alpha=0.8, color=['#3498db', '#e74c3c', '#95a5a6'])
        
        ax.set_ylabel('Average Attention Mass at t_post')
        ax.set_title('Attention Distribution (Jailbreak Condition)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}',
                    ha='center', va='bottom')
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"\n✓ Attention Plot saved to '{save_path}'")

    @staticmethod
    def plot_attention_breakdown_comparison(
        refused_results: Dict[str, List[float]], 
        complied_results: Dict[str, List[float]], 
        save_path: str = 'attention_breakdown_comparison.png'
    ):
        """
        Plot side-by-side comparison of attention patterns for Refused vs Complied groups.
        """
        labels = ['Instruction', 'Adv Suffix', 'System/Other']
        keys = ['instruction_attn', 'suffix_attn', 'system_attn']
        
        # Prepare data
        ref_means = [np.mean(refused_results[k]) for k in keys]
        ref_stds = [np.std(refused_results[k]) for k in keys]
        
        comp_means = [np.mean(complied_results[k]) for k in keys]
        comp_stds = [np.std(complied_results[k]) for k in keys]
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Bars
        rects1 = ax.bar(x - width/2, ref_means, width, yerr=ref_stds, label='Refused (Success)', capsize=5, alpha=0.8, color='#2ecc71')
        rects2 = ax.bar(x + width/2, comp_means, width, yerr=comp_stds, label='Complied (Failed)', capsize=5, alpha=0.8, color='#e74c3c')
        
        ax.set_ylabel('Attention Mass at t_post')
        ax.set_title('Attention Distribution: Refused vs Complied (Top 3)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add values on top
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., height,
                        f'{height:.1%}',
                        ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)
        
        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"\n✓ Comparative Attention Plot saved to '{save_path}'")

    @staticmethod
    def plot_individual_samples(
        results: Dict[str, List[float]], 
        group_name: str,
        save_path: str
    ):
        """
        Plot attention distribution for individual samples (e.g., Top 3).
        Creates a figure with N subplots (where N is number of samples in results).
        """
        n_samples = len(results['instruction_attn'])
        labels = ['Instruction', 'Adv Suffix', 'System']
        keys = ['instruction_attn', 'suffix_attn', 'system_attn']
        colors = ['#3498db', '#e74c3c', '#95a5a6']
        
        fig, axes = plt.subplots(1, n_samples, figsize=(5 * n_samples, 5))
        if n_samples == 1:
            axes = [axes]
            
        for i in range(n_samples):
            ax = axes[i]
            vals = [results[k][i] for k in keys]
            
            x = np.arange(len(labels))
            bars = ax.bar(x, vals, color=colors, alpha=0.8)
            
            ax.set_title(f"{group_name} Sample {i+1}")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylim(0, 1.0) # Attention sums to 1
            ax.grid(True, axis='y', alpha=0.3)
            
            # Label values
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1%}',
                        ha='center', va='bottom')
                
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"\n✓ Individual Samples Plot saved to '{save_path}'")

    @staticmethod
    def plot_token_heatmap(
        token_data_list: List[Dict],
        group_name: str,
        save_path: str
    ):
        """
        Plot token-level attention heatmap (1D strip) for each sample.
        """
        import seaborn as sns
        
        n_samples = len(token_data_list)
        # Create a figure with N subplots (stacked vertically)
        fig, axes = plt.subplots(n_samples, 1, figsize=(15, 3 * n_samples))
        if n_samples == 1:
            axes = [axes]
            
        for i, data in enumerate(token_data_list):
            ax = axes[i]
            tokens = data['tokens']
            attn = data['attention']
            
            # Clean tokens for display (replace Ġ with space, etc)
            labels = [t.replace('Ġ', ' ').replace('Ċ', '\\n') for t in tokens]
            
            # Create a 1D heatmap (reshape to [1, seq_len])
            sns.heatmap(
                attn.reshape(1, -1), 
                annot=False, # Too messy with values
                cmap="viridis",
                xticklabels=labels,
                yticklabels=False,
                ax=ax,
                cbar_kws={"orientation": "horizontal", "pad": 0.2}
            )
            
            ax.set_title(f"{group_name} Sample {i+1}")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"\n✓ Token Heatmap saved to '{save_path}'")
        return detailed_results

    @staticmethod
    def plot_token_attention_grid(
        detailed_attention: Dict,
        title: str,
        save_path: str
    ):
        """
        Plot a full token-vs-token attention grid heatmap.
        """
        import seaborn as sns
        
        tokens = detailed_attention['tokens']
        attn = detailed_attention['attention']
        
        # Clean tokens
        labels = [t.replace('Ġ', ' ').replace('Ċ', '\\n') for t in tokens]
        
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(
            attn,
            xticklabels=labels,
            yticklabels=labels,
            cmap="viridis",
            annot=False
        )
        
        if 'pos_info' in detailed_attention:
            pos = detailed_attention['pos_info']
            # Draw vertical/horizontal lines
            # t_inst: End of instruction
            if pos['t_inst']:
                idx = pos['t_inst']
                plt.axvline(x=idx + 0.5, color='white', linestyle='--', alpha=0.5, linewidth=1)
                plt.axhline(y=idx + 0.5, color='white', linestyle='--', alpha=0.5, linewidth=1)
                
            # Adv Suffix Region
            if pos.get('adv_start') and pos.get('adv_end'):
                start = pos['adv_start']
                end = pos['adv_end']
                
                # Highlight suffix area on axes
                # We can draw a box or just lines
                plt.axvline(x=start, color='red', linestyle=':', alpha=0.5)
                plt.axvline(x=end + 1, color='red', linestyle=':', alpha=0.5)
                plt.axhline(y=start, color='red', linestyle=':', alpha=0.5)
                plt.axhline(y=end + 1, color='red', linestyle=':', alpha=0.5)
                
        plt.title(title)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"✓ Saved attention grid to {save_path}")
        plt.close()

    @staticmethod
    def plot_probe_attention(
        tokens: List[str],
        attention_weights: np.ndarray,
        title: str,
        save_path: str,
        pos_info: Dict = None
    ):
        """
        Plot learned attention weights from AttentionProbe as a bar chart/heatmap.
        """
        import seaborn as sns
        
        # Clean tokens
        labels = [t.replace('Ġ', ' ').replace('Ċ', '\\n') for t in tokens]
        
        plt.figure(figsize=(15, 3))
        
        # Create a 1D heatmap
        sns.heatmap(
            attention_weights.reshape(1, -1),
            xticklabels=labels,
            yticklabels=False,
            cmap="Reds",
            annot=False,
            cbar_kws={"label": "Attention Weight"}
        )
        
        # Add region markers
        if pos_info:
            # t_inst
            if pos_info.get('t_inst'):
                plt.axvline(x=pos_info['t_inst'] + 0.5, color='blue', linestyle='--', label='Inst End')
                
            # Adv Suffix
            if pos_info.get('adv_start') and pos_info.get('adv_end'):
                # Draw shaded region
                plt.axvspan(
                    pos_info['adv_start'], 
                    pos_info['adv_end'] + 1, 
                    color='red', alpha=0.1, label='Suffix'
                )
            
            # t_post (last token)
            # usually implied by end of plot, but let's check
            # if t_post < len(tokens)-1: mark it?
            pass
            
            plt.legend(loc='upper right', fontsize='small')

        plt.title(title)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"✓ Saved probe attention plot to {save_path}")
        plt.close()
