"""
evaluation.py
-------------
Evaluate trained models and generate visualizations.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)


# ── Style Setup ────────────────────────────────────────────────────────────

plt.style.use('seaborn-v0_8-darkgrid')
COLORS = ['#6C5CE7', '#00B894', '#E17055', '#0984E3', '#FDCB6E']


# ── Metrics ────────────────────────────────────────────────────────────────

def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """Evaluate a single model and return metrics."""
    y_pred = model.predict(X_test)

    # Get probability scores for ROC
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = y_pred.astype(float)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'y_pred': y_pred,
        'y_proba': y_proba,
    }

    print(f"\n📈 {name}:")
    print(f"   Accuracy  : {metrics['accuracy']:.4f}")
    print(f"   Precision : {metrics['precision']:.4f}")
    print(f"   Recall    : {metrics['recall']:.4f}")
    print(f"   F1 Score  : {metrics['f1_score']:.4f}")

    return metrics


def evaluate_all_models(trained_models: dict, X_test, y_test) -> dict:
    """Evaluate all trained models."""
    print("\n" + "=" * 60)
    print("  📊  MODEL EVALUATION")
    print("=" * 60)

    results = {}
    for name, info in trained_models.items():
        metrics = evaluate_model(name, info['model'], X_test, y_test)
        metrics['time'] = info['time']
        results[name] = metrics

    return results


# ── Visualizations ─────────────────────────────────────────────────────────

def plot_confusion_matrices(results: dict, y_test, save_dir: str = 'outputs'):
    """Generate confusion matrix heatmaps for all models."""
    os.makedirs(save_dir, exist_ok=True)

    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold', y=1.02)

    for ax, (name, metrics) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, metrics['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax,
                    xticklabels=['Not Toxic', 'Toxic'],
                    yticklabels=['Not Toxic', 'Toxic'],
                    cbar_kws={'shrink': 0.8})
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

    plt.tight_layout()
    path = os.path.join(save_dir, 'confusion_matrices.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n📸 Confusion matrices saved → {path}")


def plot_roc_curves(results: dict, y_test, save_dir: str = 'outputs'):
    """Generate ROC curves for all models."""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (name, metrics) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, metrics['y_proba'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[i % len(COLORS)], linewidth=2.5,
                label=f'{name} (AUC = {roc_auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC = 0.5)')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'roc_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📸 ROC curves saved → {path}")


def plot_model_comparison(results: dict, save_dir: str = 'outputs'):
    """Bar chart comparing all models on key metrics."""
    os.makedirs(save_dir, exist_ok=True)

    model_names = list(results.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
    display_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    x = np.arange(len(model_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (metric, display) in enumerate(zip(metrics_names, display_names)):
        values = [results[m][metric] for m in model_names]
        bars = ax.bar(x + i * width, values, width, label=display,
                      color=COLORS[i], edgecolor='white', linewidth=0.5)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'model_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📸 Model comparison saved → {path}")


def plot_training_time(results: dict, save_dir: str = 'outputs'):
    """Bar chart of training times."""
    os.makedirs(save_dir, exist_ok=True)

    names = list(results.keys())
    times = [results[n]['time'] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names, times, color=COLORS[:len(names)], edgecolor='white', height=0.5)
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{t:.2f}s', va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('Training Time (seconds)', fontsize=12)
    ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'training_time.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"📸 Training time chart saved → {path}")


def generate_all_plots(results: dict, y_test, save_dir: str = 'outputs'):
    """Generate all evaluation plots."""
    print("\n" + "=" * 60)
    print("  🎨  GENERATING VISUALIZATIONS")
    print("=" * 60)

    plot_confusion_matrices(results, y_test, save_dir)
    plot_roc_curves(results, y_test, save_dir)
    plot_model_comparison(results, save_dir)
    plot_training_time(results, save_dir)

    print(f"\n✅ All plots saved to '{save_dir}/' folder")


def print_summary_table(results: dict):
    """Print a formatted summary table of results."""
    print("\n" + "=" * 75)
    print("  📋  RESULTS SUMMARY")
    print("=" * 75)
    print(f"  {'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Time':>8}")
    print("  " + "-" * 70)
    for name, m in results.items():
        print(f"  {name:<22} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>10.4f} {m['f1_score']:>10.4f} {m['time']:>7.2f}s")
    print("=" * 75)

    # Find best model
    best = max(results.items(), key=lambda x: x[1]['f1_score'])
    print(f"\n  🏆 Best Model: {best[0]} (F1 = {best[1]['f1_score']:.4f})")
