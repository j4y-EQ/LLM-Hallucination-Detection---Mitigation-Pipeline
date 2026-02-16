"""Cross-Model Comparison Visualization Script.

Generates comparative visualizations of hallucination reduction across multiple models
(Llama, Gemma, Qwen) and datasets (Primary, MMLU, HellaSwag). Creates bar charts and
scatter plots showing baseline vs steered performance, with annotations for absolute
and relative improvements or degradations.

Output Plots:
- primary_comparison.png: Hallucination reduction on primary dataset
- mmlu_comparison.png: General abilities impact on MMLU
- hellaswag_comparison.png: General abilities impact on HellaSwag
- general_abilities_changes.png: Performance changes across datasets
- all_metrics_comparison.png: Comprehensive multi-metric view

Usage:
    python -m steer.across_model_comparison

Note: Hardcoded data for specific experimental results. Update `data` dict
      with new results before running.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory if it doesn't exist
output_dir = r"C:\Users\yenqi\Documents\Steering Experiments\graphs_across_all_results"
os.makedirs(output_dir, exist_ok=True)

# Data for all three models
data = {
    'Llama': {
        'config': 'k=15, α=3.50',
        'Primary': {'baseline': 58.82, 'steered': 52.94, 'reduction': 5.88, 'relative': 10.00},
        'MMLU': {'baseline': 39.72, 'steered': 40.60, 'reduction': -0.88, 'relative': -2.21},
        'HellaSwag': {'baseline': 29.33, 'steered': 29.86, 'reduction': -0.53, 'relative': -1.81}
    },
    'Gemma': {
        'config': 'k=7, α=28.20',
        'Primary': {'baseline': 35.18, 'steered': 31.88, 'reduction': 3.3, 'relative': 9.37},
        'MMLU': {'baseline': 32.28, 'steered': 32.80, 'reduction': -0.52, 'relative': -1.64},
        'HellaSwag': {'baseline': 27.61, 'steered': 23.01, 'reduction': 4.60, 'relative': 16.67}
    },
    'Qwen': {
        'config': 'k=418, α=3.25',
        'Primary': {'baseline': 75.48, 'steered': 56.23, 'reduction':  19.25 , 'relative': 25.50},
        'MMLU': {'baseline': 37.43, 'steered': 38.84, 'reduction': -1.41, 'relative': -3.76},
        'HellaSwag': {'baseline': 15.22, 'steered': 17.88, 'reduction':  -2.66 , 'relative': -17.44}
    }
}

models = ['Llama', 'Gemma', 'Qwen']
model_labels = ['Llama 3 8b Instruct', 'Gemma 2 9b Instruct', 'Qwen 2.5 7b Instruct']
datasets = ['Primary', 'MMLU', 'HellaSwag']
dataset_titles = {
    'Primary': 'Hallucination Reduction',
    'MMLU': 'General Abilities (MMLU)',
    'HellaSwag': 'General Abilities (HellaSwag)'
}

# Create a graph for each dataset
for dataset in datasets:
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(models))
    width = 0.35
    
    baseline_values = [data[model][dataset]['baseline'] for model in models]
    steered_values = [data[model][dataset]['steered'] for model in models]
    
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, steered_values, width, label='Steered', color='#4ECDC4', alpha=0.8)
    
    # Calculate consistent spacing for annotations (as percentage of max value)
    max_value = max(max(baseline_values), max(steered_values))
    annotation_offset = max_value * 0.07  # 4% of max value for consistent spacing
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%',
                   ha='center', va='bottom', fontsize=18, fontweight='bold')
    
    # Add reduction annotations
    for i, model in enumerate(models):
        reduction = data[model][dataset]['reduction']
        relative = data[model][dataset]['relative']
        baseline = data[model][dataset]['baseline']
        steered = data[model][dataset]['steered']
        
        # Position annotation on top of the steered bar with consistent spacing
        y_pos = steered + annotation_offset
        
        # Color based on whether it's improvement (green) or degradation (red)
        # Positive reduction is good (green), negative is bad (red)
        color = 'green' if reduction > 0 else 'red'
        arrow = '↓' if reduction > 0 else '↑'
        
        ax.text(i + width/2, y_pos, f'{arrow} {abs(reduction):.2f}pp',
               ha='center', va='bottom', fontsize=20, color=color, fontweight='bold')
    
    # Customize the plot
    ax.set_xlabel('Model', fontsize=16, fontweight='bold', labelpad=15)
    ax.set_ylabel('Hallucination Rate (%)', fontsize=16, fontweight='bold')
    
    ax.set_title(f'{dataset_titles[dataset]} - Before vs After Steering', 
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    
    # Add model labels to x-axis
    ax.set_xticklabels(model_labels, fontsize=16, fontweight='bold')
    
    ax.legend(fontsize=14, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.tick_params(axis='y', labelsize=14)
    
    # Set y-axis limits to add more space at the top
    ax.set_ylim(0, max_value * 1.25)
    
    # Add a subtle background
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    
    # Add config info for Primary graph below x-axis labels
    if dataset == 'Primary':
        # Adjust bottom margin first to create space
        plt.subplots_adjust(bottom=0.15)
        for i, model in enumerate(models):
            ax.text(i, -0.055, data[model]['config'],
                   ha='center', va='top', fontsize=12, fontweight='normal',
                   transform=ax.get_xaxis_transform())
    
    # Save the figure
    filename = f'{dataset.lower()}_comparison.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f'Saved: {filepath}')
    
    plt.close()

# Create grouped bar chart for general abilities changes
fig, ax = plt.subplots(figsize=(14, 9))

x = np.arange(len(models))
width = 0.35

# Extract change values for MMLU and HellaSwag (negative = degradation)
mmlu_changes = [data[model]['MMLU']['reduction'] for model in models]
hellaswag_changes = [data[model]['HellaSwag']['reduction'] for model in models]

# Create bars
bars1 = ax.bar(x - width/2, mmlu_changes, width, label='MMLU', color='#E17055', alpha=0.85)
bars2 = ax.bar(x + width/2, hellaswag_changes, width, label='HellaSwag', color='#0984E3', alpha=0.85)

# Add horizontal line at y=0
ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7, zorder=1)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        # Position label above or below bar depending on sign
        va = 'bottom' if height >= 0 else 'top'
        y_pos = height if height >= 0 else height
        
        # Color based on positive (green) or negative (red)
        text_color = 'green' if height >= 0 else 'red'
        
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
               f'{height:.2f}pp',
               ha='center', va=va, fontsize=16, fontweight='bold', color=text_color)

# Customize the plot
ax.set_xlabel('Model', fontsize=18, fontweight='bold')
ax.set_ylabel('Change in Performance (percentage points)', fontsize=18, fontweight='bold')
ax.set_title('General Abilities Impact: Performance Changes After Steering\n(Negative = Degradation, Positive = Improvement)', 
            fontsize=19, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=16, fontweight='bold')

ax.legend(fontsize=15, loc='upper right', framealpha=0.95)
ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
ax.tick_params(axis='y', labelsize=14)

# Set y-axis limits with padding
all_changes = mmlu_changes + hellaswag_changes
y_min = min(all_changes)
y_max = max(all_changes)
y_range = y_max - y_min
ax.set_ylim(y_min - y_range * 0.2, y_max + y_range * 0.3)

# Add subtle background
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

# Add annotation box explaining negative vs positive
textstr = 'Note: Negative values indicate performance degradation\nPositive values indicate performance improvement'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

plt.tight_layout()

# Save the figure
filepath = os.path.join(output_dir, 'general_abilities_changes.png')
plt.savefig(filepath, dpi=300, bbox_inches='tight')
print(f'Saved: {filepath}')

plt.close()

# Create grouped bar chart for all three metrics changes
fig, ax = plt.subplots(figsize=(16, 9))

x = np.arange(len(models))
width = 0.25

# Extract change values for all three datasets
primary_changes = [data[model]['Primary']['reduction'] for model in models]
mmlu_changes = [data[model]['MMLU']['reduction'] for model in models]
hellaswag_changes = [data[model]['HellaSwag']['reduction'] for model in models]

# Create bars with three groups per model
bars1 = ax.bar(x - width, primary_changes, width, label='Primary (Hallucination)', color='#FDCB6E', alpha=0.85)
bars2 = ax.bar(x, mmlu_changes, width, label='MMLU', color='#E17055', alpha=0.85)
bars3 = ax.bar(x + width, hellaswag_changes, width, label='HellaSwag', color='#0984E3', alpha=0.85)

# Add horizontal line at y=0
ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7, zorder=1)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        # Position label above or below bar depending on sign
        va = 'bottom' if height >= 0 else 'top'
        y_pos = height if height >= 0 else height
        
        # Color based on positive (green) or negative (red)
        text_color = 'green' if height >= 0 else 'red'
        
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
               f'{height:.2f}pp',
               ha='center', va=va, fontsize=14, fontweight='bold', color=text_color)

# Customize the plot
ax.set_xlabel('Model', fontsize=18, fontweight='bold')
ax.set_ylabel('Change in Performance (percentage points)', fontsize=18, fontweight='bold')
ax.set_title('All Metrics Impact: Performance Changes After Steering\n(Positive = Improvement/Reduction, Negative = Degradation)', 
            fontsize=19, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=16, fontweight='bold')

ax.legend(fontsize=15, loc='upper right', framealpha=0.95)
ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
ax.tick_params(axis='y', labelsize=14)

# Set y-axis limits with padding
all_changes = primary_changes + mmlu_changes + hellaswag_changes
y_min = min(all_changes)
y_max = max(all_changes)
y_range = y_max - y_min
ax.set_ylim(y_min - y_range * 0.15, y_max + y_range * 0.25)

# Add subtle background
ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('white')

# Add annotation box explaining the metrics
textstr = 'Primary: Positive = hallucination reduction (good)\nMMLU/HellaSwag: Negative = performance degradation (bad)'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

plt.tight_layout()

# Save the figure
filepath = os.path.join(output_dir, 'all_metrics_changes.png')
plt.savefig(filepath, dpi=300, bbox_inches='tight')
print(f'Saved: {filepath}')

plt.close()

print(f'\nAll graphs saved to {output_dir}/')
