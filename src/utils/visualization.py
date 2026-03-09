import numpy as np
import matplotlib.pyplot as plt


def plot_ga_evolution(generation_data, baseline_values, panels, title, save_path):
    n_panels = len(panels)
    rows = (n_panels + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    gens = generation_data['gen']

    for i, panel in enumerate(panels):
        ax = axes[i // 2, i % 2]
        data_key = panel['data_key']
        label = panel.get('ylabel', data_key)
        color = panel.get('color', 'r')
        scale = panel.get('scale', 1.0)

        values = np.array([v * scale for v in generation_data[data_key]])
        if panel.get('cummax', False):
            ax.plot(gens, values, linestyle='-', color=color, linewidth=1, alpha=0.35,
                    label='Per-generation best')
            values = np.maximum.accumulate(values)
            ax.plot(gens, values, linestyle='-', color=color, linewidth=2, label='Global best')
            ax.legend(fontsize=8)
        else:
            ax.plot(gens, values, linestyle='-', color=color, linewidth=2)

        baseline_key = panel.get('baseline_key')
        if baseline_key and baseline_key in baseline_values:
            ax.axhline(baseline_values[baseline_key] * scale, color='gray', linestyle='--')

        ax.set_ylabel(label)
        ax.set_title(panel.get('title', label))
        ax.grid(True, alpha=0.3)
        if i >= n_panels - 2:
            ax.set_xlabel('Generation')

    # Hide unused axes
    for i in range(n_panels, rows * 2):
        axes[i // 2, i % 2].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved: {save_path}")
    plt.close()


def plot_pgd_results(step_data, base_stats, adv_stats, metric_pairs, title, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(step_data['step'], step_data['loss'], 'b-', alpha=0.5, label='Current')
    axes[0].plot(step_data['step'], step_data['best_loss'], 'r-', linewidth=2, label='Best')
    axes[0].set_xlabel('PGD Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    labels = [m[0] for m in metric_pairs]
    baseline_vals = [m[1] for m in metric_pairs]
    adv_vals = [m[2] for m in metric_pairs]

    x = np.arange(len(labels))
    width = 0.35

    bars1 = axes[1].bar(x - width / 2, baseline_vals, width, label='Baseline', color='steelblue')
    bars2 = axes[1].bar(x + width / 2, adv_vals, width, label='Adversarial', color='coral')

    axes[1].set_ylabel('Value')
    axes[1].set_title('Baseline vs Adversarial')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars1, baseline_vals):
        axes[1].annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, adv_vals):
        axes[1].annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved: {save_path}")
    plt.close()


def plot_attribution_heatmap(data, feature_names, title, ax, cmap='RdBu_r'):
    from matplotlib.colors import TwoSlopeNorm
    vmax = np.abs(data).max()
    if vmax == 0:
        vmax = 1
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(data.T, aspect='auto', cmap=cmap, norm=norm)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Feature')
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    return im


def plot_feature_importance_barh(importance, feature_names, title, ax,
                                 color='steelblue', sort=True):
    if sort:
        sorted_idx = np.argsort(importance)[::-1]
        y_pos = np.arange(len(feature_names))
        ax.barh(y_pos, importance[sorted_idx], color=color)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=8)
    else:
        y_pos = np.arange(len(feature_names))
        ax.barh(y_pos, importance, color=color)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
