import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the two CSV files
df_R = pd.read_csv('books_knowmem_f_npo_gdr_R_llama3.csv')
df_PS = pd.read_csv('books_knowmem_f_npo_gdr_PS_llama3.csv')

# Extract including_ratio from the model name (after the last underscore)
def extract_ratio(name):
    if 'Meta-Llama' in name or name == 'meta-llama/Meta-Llama-3-8B':
        return 1.0
    parts = name.split('_')
    try:
        return float(parts[-1])
    except ValueError:
        return 1.0

df_R['including_ratio'] = df_R['name'].apply(extract_ratio)
df_PS['including_ratio'] = df_PS['name'].apply(extract_ratio)

print(df_R[['name', 'including_ratio']].head())

# Extract the including_ratio for x-axis (excluding the base model at 1.0)
# Filter to only include ratios up to 0.25
x_R = df_R[(df_R['including_ratio'] != 1.0) & (df_R['including_ratio'] <= 0.25)]['including_ratio'].values
x_PS = df_PS[(df_PS['including_ratio'] != 1.0) & (df_PS['including_ratio'] <= 0.25)]['including_ratio'].values

# Get all numeric columns to plot (excluding metadata columns)
exclude_cols = ['name', 'epoch', 'indices_seed', 'including_ratio']
plot_columns = [col for col in df_R.columns if col not in exclude_cols and df_R[col].dtype in [np.float64, np.int64, float, int]]

# Create a figure with subplots
n_cols = len(plot_columns)
n_rows = (n_cols + 2) // 3  # 3 columns per row
fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
axes = axes.flatten()

# Plot each metric
for idx, col in enumerate(plot_columns):
    ax = axes[idx]
    
    # Get values for both datasets (excluding base model, up to 0.25)
    y_R = df_R[(df_R['including_ratio'] != 1.0) & (df_R['including_ratio'] <= 0.25)][col].values
    y_PS = df_PS[(df_PS['including_ratio'] != 1.0) & (df_PS['including_ratio'] <= 0.25)][col].values
    
    # Plot both lines
    ax.plot(x_R, y_R, marker='o', linewidth=2, markersize=8, label='Random (R)', color='#2E86AB')
    ax.plot(x_PS, y_PS, marker='s', linewidth=2, markersize=8, label='PS-based', color='#A23B72')
    
    # Formatting
    ax.set_xlabel('Including Ratio', fontsize=11, fontweight='bold')
    ax.set_ylabel(col.replace('_', ' ').title(), fontsize=11, fontweight='bold')
    ax.set_title(col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-0.05, 0.3)
    
    # Format x-axis
    ax.set_xticks([0.05, 0.1, 0.25])

# Hide any unused subplots
for idx in range(len(plot_columns), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('comparison_plot_all_metrics.png', dpi=300, bbox_inches='tight')
print(f"Saved plot with {len(plot_columns)} metrics to 'comparison_plot_all_metrics_randonly.png'")

# Create a second figure with key metrics only
key_metrics = ['knowmem_f', 'fluency_lambada_acc', '']
key_metrics = [m for m in key_metrics if m in plot_columns]

# Baseline values at including_ratio = 0 (pretrained model)
baseline_values = {
    'knowmem_f': 56.88,
    'fluency_lambada_acc': 39.6
}

if key_metrics:
    fig2, axes2 = plt.subplots(1, len(key_metrics), figsize=(5 * len(key_metrics), 4))
    if len(key_metrics) == 1:
        axes2 = [axes2]
    
    for idx, col in enumerate(key_metrics):
        ax = axes2[idx]
        
        # Get values for unlearned models (excluding base model, up to 0.25)
        y_R = df_R[(df_R['including_ratio'] != 1.0) & (df_R['including_ratio'] <= 0.25)][col].values
        y_PS = df_PS[(df_PS['including_ratio'] != 1.0) & (df_PS['including_ratio'] <= 0.25)][col].values
        
        # Add baseline value at x=0
        x_R_with_baseline = np.concatenate([[0], x_R])
        x_PS_with_baseline = np.concatenate([[0], x_PS])
        y_R_with_baseline = np.concatenate([[baseline_values[col]], y_R])
        y_PS_with_baseline = np.concatenate([[baseline_values[col]], y_PS])
        
        ax.plot(x_R_with_baseline, y_R_with_baseline, marker='o', linewidth=2.5, markersize=10, label='Random (R)', color='#2E86AB')
        ax.plot(x_PS_with_baseline, y_PS_with_baseline, marker='s', linewidth=2.5, markersize=10, label='PS-based', color='#A23B72')
        
        ax.set_xlabel('Including Ratio', fontsize=12, fontweight='bold')
        ax.set_ylabel(col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_title(col.replace('_', ' ').title(), fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(-0.02, 0.27)
        ax.set_xticks([0, 0.05, 0.1, 0.25, 0.5, 1.0])
    
    plt.tight_layout()
    plt.savefig('comparison_plot_key_metrics.png', dpi=300, bbox_inches='tight')
    print(f"Saved key metrics plot to 'comparison_plot_key_metrics.png'")

plt.show()
