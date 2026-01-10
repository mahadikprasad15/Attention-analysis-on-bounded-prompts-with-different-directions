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
            results: Dictionary with keys 'instruction_attn', 'suffix_attn', 'system_attn', 'user_prefix_attn'
                     containing lists of float values (one per sample)
            save_path: Path to save plot
        """
        # Calculate means and stds
        means = {k: np.mean(v) for k, v in results.items()}
        stds = {k: np.std(v) for k, v in results.items()}

        # Combine user_prefix_attn and system_attn into a single "System/Template" category
        if 'user_prefix_attn' in results:
            combined_system = [u + s for u, s in zip(results['user_prefix_attn'], results['system_attn'])]
            means['combined_system'] = np.mean(combined_system)
            stds['combined_system'] = np.std(combined_system)
        else:
            combined_system = results['system_attn']
            means['combined_system'] = means['system_attn']
            stds['combined_system'] = stds['system_attn']

        labels = ['Instruction', 'Adv Suffix', 'System/Template']
        keys = ['instruction_attn', 'suffix_attn', 'combined_system']

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
        labels = ['Instruction', 'Adv Suffix', 'System/Template']

        # Combine user_prefix and system for both groups
        def combine_system(results):
            if 'user_prefix_attn' in results:
                return [u + s for u, s in zip(results['user_prefix_attn'], results['system_attn'])]
            return results['system_attn']

        ref_combined = {
            'instruction_attn': refused_results['instruction_attn'],
            'suffix_attn': refused_results['suffix_attn'],
            'combined_system': combine_system(refused_results)
        }

        comp_combined = {
            'instruction_attn': complied_results['instruction_attn'],
            'suffix_attn': complied_results['suffix_attn'],
            'combined_system': combine_system(complied_results)
        }

        keys = ['instruction_attn', 'suffix_attn', 'combined_system']

        # Prepare data
        ref_means = [np.mean(ref_combined[k]) for k in keys]
        ref_stds = [np.std(ref_combined[k]) for k in keys]

        comp_means = [np.mean(comp_combined[k]) for k in keys]
        comp_stds = [np.std(comp_combined[k]) for k in keys]
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Bars
        rects1 = ax.bar(x - width/2, ref_means, width, yerr=ref_stds, label='Clean Refused (No Suffix)', capsize=5, alpha=0.8, color='#2ecc71')
        rects2 = ax.bar(x + width/2, comp_means, width, yerr=comp_stds, label='Jailbreak Complied (With Suffix)', capsize=5, alpha=0.8, color='#e74c3c')

        ax.set_ylabel('Attention Mass at t_post')
        ax.set_title('Attention Distribution: Clean Refused vs Jailbreak Complied (Top 3)')
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
        labels = ['Instruction', 'Adv Suffix', 'System/Template']
        colors = ['#3498db', '#e74c3c', '#95a5a6']

        # Combine user_prefix and system for each sample
        if 'user_prefix_attn' in results:
            combined_system = [u + s for u, s in zip(results['user_prefix_attn'], results['system_attn'])]
        else:
            combined_system = results['system_attn']

        keys = ['instruction_attn', 'suffix_attn', 'combined_system']
        combined_results = {
            'instruction_attn': results['instruction_attn'],
            'suffix_attn': results['suffix_attn'],
            'combined_system': combined_system
        }

        fig, axes = plt.subplots(1, n_samples, figsize=(5 * n_samples, 5))
        if n_samples == 1:
            axes = [axes]

        for i in range(n_samples):
            ax = axes[i]
            vals = [combined_results[k][i] for k in keys]
            
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

    @staticmethod
    def plot_all_heads_heatmap(
        sweep_data: Dict,
        metric: str = 'suffix_attn',
        save_path: str = 'heads_heatmap.png'
    ):
        """
        Plot heatmap of attention metric across all heads.

        Args:
            sweep_data: Output from get_all_heads_attention_breakdown()
            metric: Which metric to plot ('suffix_attn', 'instruction_attn', 'system_attn')
            save_path: Where to save the plot
        """
        import seaborn as sns

        # Get data: [n_layers, n_heads, n_prompts]
        # Average across prompts to get [n_layers, n_heads]
        data = sweep_data[metric].mean(axis=2)

        n_layers = sweep_data['n_layers']
        n_heads = sweep_data['n_heads']
        group_name = sweep_data['group_name']

        # Create figure
        fig, ax = plt.subplots(figsize=(16, 8))

        # Plot heatmap
        sns.heatmap(
            data,
            cmap='YlOrRd',
            annot=False,
            fmt='.1%',
            cbar_kws={'label': f'Attention Mass'},
            ax=ax,
            vmin=0,
            vmax=data.max()
        )

        metric_names = {
            'suffix_attn': 'Adversarial Suffix',
            'instruction_attn': 'Instruction',
            'system_attn': 'System/Template',
            'user_prefix_attn': 'User Prefix'
        }

        ax.set_title(f'Attention to {metric_names.get(metric, metric)} - {group_name} ({sweep_data["n_prompts"]} prompts)',
                     fontsize=14, pad=20)
        ax.set_xlabel('Head Index', fontsize=12)
        ax.set_ylabel('Layer Index', fontsize=12)
        ax.set_xticks(np.arange(0, n_heads, 4))
        ax.set_xticklabels(np.arange(0, n_heads, 4))
        ax.set_yticks(np.arange(0, n_layers, 2))
        ax.set_yticklabels(np.arange(0, n_layers, 2))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Heatmap saved to '{save_path}'")
        plt.close()

    @staticmethod
    def plot_top_heads_bar(
        sweep_data: Dict,
        metric: str = 'suffix_attn',
        top_n: int = 20,
        save_path: str = 'top_heads_bar.png'
    ):
        """
        Plot bar chart of top N heads by attention metric.

        Args:
            sweep_data: Output from get_all_heads_attention_breakdown()
            metric: Which metric to rank by
            top_n: How many top heads to show
            save_path: Where to save the plot
        """
        # Get data and average across prompts
        data = sweep_data[metric].mean(axis=2)  # [n_layers, n_heads]

        n_layers = sweep_data['n_layers']
        n_heads = sweep_data['n_heads']

        # Flatten and find top heads
        flat_data = data.flatten()
        flat_indices = np.argsort(flat_data)[::-1][:top_n]

        # Convert flat indices to (layer, head) tuples
        top_heads = []
        top_values = []
        for idx in flat_indices:
            layer = idx // n_heads
            head = idx % n_heads
            value = data[layer, head]
            top_heads.append(f"L{layer}H{head}")
            top_values.append(value)

        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, top_n))
        bars = ax.barh(range(top_n), top_values, color=colors)

        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_heads)
        ax.set_xlabel('Attention Mass', fontsize=12)
        ax.set_title(f'Top {top_n} Heads by {metric} - {sweep_data["group_name"]}', fontsize=14)
        ax.invert_yaxis()
        ax.grid(True, axis='x', alpha=0.3)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_values)):
            ax.text(val, i, f' {val:.1%}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Top heads bar chart saved to '{save_path}'")
        plt.close()

    @staticmethod
    def plot_layer_summary(
        sweep_data: Dict,
        save_path: str = 'layer_summary.png'
    ):
        """
        Plot layer-wise summary (average across all heads in each layer).

        Args:
            sweep_data: Output from get_all_heads_attention_breakdown()
            save_path: Where to save the plot
        """
        # Average across heads and prompts: [n_layers, n_heads, n_prompts] -> [n_layers]
        instruction_by_layer = sweep_data['instruction_attn'].mean(axis=(1, 2))
        suffix_by_layer = sweep_data['suffix_attn'].mean(axis=(1, 2))
        system_by_layer = sweep_data['system_attn'].mean(axis=(1, 2))
        user_prefix_by_layer = sweep_data['user_prefix_attn'].mean(axis=(1, 2))

        # Combine user_prefix and system
        combined_system = user_prefix_by_layer + system_by_layer

        n_layers = sweep_data['n_layers']
        layers = np.arange(n_layers)

        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(14, 6))

        width = 0.8
        ax.bar(layers, instruction_by_layer, width, label='Instruction', color='#3498db', alpha=0.8)
        ax.bar(layers, suffix_by_layer, width, bottom=instruction_by_layer,
               label='Adv Suffix', color='#e74c3c', alpha=0.8)
        ax.bar(layers, combined_system, width,
               bottom=instruction_by_layer + suffix_by_layer,
               label='System/Template', color='#95a5a6', alpha=0.8)

        ax.set_xlabel('Layer Index', fontsize=12)
        ax.set_ylabel('Average Attention Mass', fontsize=12)
        ax.set_title(f'Layer-wise Attention Distribution - {sweep_data["group_name"]}', fontsize=14)
        ax.legend()
        ax.set_xticks(np.arange(0, n_layers, 2))
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Layer summary saved to '{save_path}'")
        plt.close()

    @staticmethod
    def plot_refused_vs_complied_heads(
        refused_data: Dict,
        complied_data: Dict,
        metric: str = 'suffix_attn',
        save_path: str = 'refused_vs_complied_heads.png'
    ):
        """
        Plot side-by-side heatmaps comparing refused vs complied groups.

        Args:
            refused_data: Sweep data for refused prompts
            complied_data: Sweep data for complied prompts
            metric: Which metric to compare
            save_path: Where to save the plot
        """
        import seaborn as sns

        # Get averaged data
        refused_avg = refused_data[metric].mean(axis=2)
        complied_avg = complied_data[metric].mean(axis=2)

        # Create side-by-side heatmaps
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Determine shared colorbar range
        vmax = max(refused_avg.max(), complied_avg.max())

        # Refused heatmap
        sns.heatmap(
            refused_avg,
            cmap='YlOrRd',
            annot=False,
            cbar_kws={'label': 'Attention Mass'},
            ax=ax1,
            vmin=0,
            vmax=vmax
        )
        ax1.set_title(f'{refused_data["group_name"]} ({refused_data["n_prompts"]} prompts)', fontsize=12)
        ax1.set_xlabel('Head Index')
        ax1.set_ylabel('Layer Index')

        # Complied heatmap
        sns.heatmap(
            complied_avg,
            cmap='YlOrRd',
            annot=False,
            cbar_kws={'label': 'Attention Mass'},
            ax=ax2,
            vmin=0,
            vmax=vmax
        )
        ax2.set_title(f'{complied_data["group_name"]} ({complied_data["n_prompts"]} prompts)', fontsize=12)
        ax2.set_xlabel('Head Index')
        ax2.set_ylabel('Layer Index')

        metric_names = {
            'suffix_attn': 'Adversarial Suffix Attention',
            'instruction_attn': 'Instruction Attention',
            'system_attn': 'System Attention'
        }

        fig.suptitle(f'{metric_names.get(metric, metric)}: Comparison', fontsize=16, y=0.98)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison heatmap saved to '{save_path}'")
        plt.close()

    @staticmethod
    def create_head_sweep_report(
        refused_data: Dict,
        complied_data: Dict,
        save_dir: str
    ):
        """
        Create a comprehensive report of all head-level visualizations.

        Args:
            refused_data: Sweep data for refused prompts
            complied_data: Sweep data for complied prompts
            save_dir: Directory to save all plots
        """
        import os

        print("\n" + "="*80)
        print("GENERATING HEAD SWEEP VISUALIZATIONS")
        print("="*80)

        # 1. Suffix attention heatmaps
        ResultAnalyzer.plot_all_heads_heatmap(
            complied_data,
            'suffix_attn',
            os.path.join(save_dir, 'heads_suffix_heatmap_complied.png')
        )

        ResultAnalyzer.plot_all_heads_heatmap(
            refused_data,
            'suffix_attn',
            os.path.join(save_dir, 'heads_suffix_heatmap_refused.png')
        )

        # 2. Instruction attention heatmaps
        ResultAnalyzer.plot_all_heads_heatmap(
            complied_data,
            'instruction_attn',
            os.path.join(save_dir, 'heads_instruction_heatmap_complied.png')
        )

        ResultAnalyzer.plot_all_heads_heatmap(
            refused_data,
            'instruction_attn',
            os.path.join(save_dir, 'heads_instruction_heatmap_refused.png')
        )

        # 3. Top heads by suffix attention
        ResultAnalyzer.plot_top_heads_bar(
            complied_data,
            'suffix_attn',
            top_n=20,
            save_path=os.path.join(save_dir, 'top20_suffix_heads_complied.png')
        )

        # 4. Layer-wise summaries
        ResultAnalyzer.plot_layer_summary(
            refused_data,
            os.path.join(save_dir, 'layer_summary_refused.png')
        )

        ResultAnalyzer.plot_layer_summary(
            complied_data,
            os.path.join(save_dir, 'layer_summary_complied.png')
        )

        # 5. Comparison heatmaps
        ResultAnalyzer.plot_refused_vs_complied_heads(
            refused_data,
            complied_data,
            'suffix_attn',
            os.path.join(save_dir, 'comparison_suffix_heads.png')
        )

        ResultAnalyzer.plot_refused_vs_complied_heads(
            refused_data,
            complied_data,
            'instruction_attn',
            os.path.join(save_dir, 'comparison_instruction_heads.png')
        )

        print("\n✓ All head sweep visualizations generated!")
        print("="*80)
