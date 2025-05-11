import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Create comparison_results directory if it doesn't exist
os.makedirs('comparison_results', exist_ok=True)

# Function to extract metrics from file
def extract_metrics(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Extract model name from the first line
        model_name = re.search(r'=== (.*?) (?:Model|CNN).*? ===', content).group(1).strip()
        
        # Extract basic metrics
        metrics = {}
        metrics['accuracy'] = float(re.search(r'Accuracy: ([\d.]+)', content).group(1)) if re.search(r'Accuracy: ([\d.]+)', content) else 0.0
        metrics['precision'] = float(re.search(r'Precision: ([\d.]+)', content).group(1)) if re.search(r'Precision: ([\d.]+)', content) else 0.0
        metrics['recall'] = float(re.search(r'Recall: ([\d.]+)', content).group(1)) if re.search(r'Recall: ([\d.]+)', content) else 0.0
        metrics['f1_score'] = float(re.search(r'F1 Score: ([\d.]+)', content).group(1)) if re.search(r'F1 Score: ([\d.]+)', content) else 0.0
        
        # Extract specific accuracies
        real_acc_match = re.search(r'Real images accuracy: ([\d.]+)', content)
        ai_acc_match = re.search(r'(?:AI-generated|Fake) images accuracy: ([\d.]+)', content)
        
        metrics['real_accuracy'] = float(real_acc_match.group(1)) if real_acc_match else 0.0
        metrics['fake_accuracy'] = float(ai_acc_match.group(1)) if ai_acc_match else 0.0
        
        # Try to extract confusion matrix
        confusion_matrix = None
        cm_match = re.search(r'Actual(?:\s+\w+)\s+(\d+)\s+(\d+).*?(?:\s+\w+)\s+(\d+)\s+(\d+)', content, re.DOTALL)
        if cm_match:
            confusion_matrix = np.array([
                [int(cm_match.group(1)), int(cm_match.group(2))],
                [int(cm_match.group(3)), int(cm_match.group(4))]
            ])
            metrics['confusion_matrix'] = confusion_matrix

        return model_name, metrics
    except Exception as e:
        print(f"Error extracting metrics from {file_path}: {e}")
        return None, {}

# Dictionary to store model metrics
models_data = {}

# Extract metrics from each model
metrics_files = [
    ('ourmodel_results/model_metrics_v2.txt', 'Our Model'),
    ('ourmodel_finetunned_results/model_metrics.txt', 'Our Model (Fine-tuned)'),
    ('cnndetection_results/model_metrics.txt', 'CNN Detection'),
    ('faceforensics_results/model_metrics.txt', 'FaceForensics'),
    ('faceforensics_finetunned_results/model_metrics.txt', 'FaceForensics (Fine-tuned)')
]

for file_path, label in metrics_files:
    model_name, metrics = extract_metrics(file_path)
    if model_name:
        models_data[label] = metrics

# Define consistent colors for each model
model_names = list(models_data.keys())
colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
model_colors = {model: colors[i] for i, model in enumerate(model_names)}

# Create bar plots for the main metrics
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

# Create a DataFrame for easier plotting
df_metrics = pd.DataFrame({model: [data.get(metric, 0) for metric in metrics_to_plot] 
                           for model, data in models_data.items()}, 
                         index=metrics_labels)

# Plot overall metrics
plt.figure(figsize=(14, 8))
ax = df_metrics.plot(kind='bar', rot=0, color=[model_colors[model] for model in df_metrics.columns], figsize=(14, 8))
plt.title('Performance Metrics Comparison Across Models', fontsize=16)
plt.ylabel('Score', fontsize=14)
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Models', fontsize=12)

# Add value labels on the bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', fontsize=9)

plt.tight_layout()
plt.savefig('comparison_results/overall_metrics_comparison.png', dpi=300, bbox_inches='tight')

# Plot real vs fake accuracy with proper handling of None values
plt.figure(figsize=(14, 8))
ax = plt.subplot(111)
bar_positions = np.arange(len(model_names))
bar_width = 0.35

# Get values, replacing None with 0
real_values = [models_data[model].get('real_accuracy', 0) or 0 for model in model_names]
fake_values = [models_data[model].get('fake_accuracy', 0) or 0 for model in model_names]

# Plot real accuracy bars
real_bars = ax.bar(
    bar_positions - bar_width/2, 
    real_values,
    bar_width, 
    label='Real Images',
    color='skyblue'
)

# Plot fake accuracy bars
fake_bars = ax.bar(
    bar_positions + bar_width/2, 
    fake_values,
    bar_width, 
    label='AI/Fake Images',
    color='salmon'
)

# Customize plot
plt.title('Real vs. AI/Fake Images Classification Accuracy', fontsize=16)
plt.ylabel('Accuracy', fontsize=14)
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(bar_positions, model_names, rotation=15, ha='right')
plt.legend(fontsize=12)

# Add value labels
ax.bar_label(real_bars, fmt='%.3f', fontsize=9)
ax.bar_label(fake_bars, fmt='%.3f', fontsize=9)

plt.tight_layout()
plt.savefig('comparison_results/real_vs_fake_accuracy.png', dpi=300, bbox_inches='tight')

# Create radar chart for model comparison
metrics_to_radar = ['accuracy', 'precision', 'recall', 'f1_score']
metrics_labels_radar = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

# Set up the radar chart
fig = plt.figure(figsize=(12, 10))
radar = fig.add_subplot(111, polar=True)

# Set ticks for the radar chart
angles = np.linspace(0, 2*np.pi, len(metrics_labels_radar), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle
radar.set_theta_offset(np.pi / 2)
radar.set_theta_direction(-1)
radar.set_thetagrids(np.degrees(angles[:-1]), metrics_labels_radar)

for label, angle in zip(radar.get_xticklabels(), angles):
    if angle in (0, np.pi):
        label.set_horizontalalignment('center')
    elif 0 < angle < np.pi:
        label.set_horizontalalignment('left')
    else:
        label.set_horizontalalignment('right')

# Set y-axis limits
radar.set_ylim(0, 1)
radar.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
radar.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
radar.grid(True)

# Plot each model with its consistent color, handling None values
for model in model_names:
    values = [models_data[model].get(metric, 0) or 0 for metric in metrics_to_radar]
    values += values[:1]  # Complete the circle
    radar.plot(angles, values, 'o-', linewidth=2, color=model_colors[model], label=model, markersize=6)
    radar.fill(angles, values, alpha=0.1, color=model_colors[model])

plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.title('Radar Plot of Model Performance Metrics', size=15, y=1.05)
plt.tight_layout()
plt.savefig('comparison_results/radar_metrics_comparison.png', dpi=300, bbox_inches='tight')

# Create a summary table as an image
plt.figure(figsize=(14, 7))
summary_data = []
for model in models_data:
    row = [model]
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'real_accuracy', 'fake_accuracy']:
        value = models_data[model].get(metric, 'N/A')
        if value is None:
            row.append('N/A')
        elif isinstance(value, (int, float)):
            row.append(f'{value:.4f}')
        else:
            row.append('N/A')
    summary_data.append(row)

columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Real Acc.', 'Fake Acc.']
the_table = plt.table(cellText=summary_data, colLabels=columns, 
                      loc='center', cellLoc='center')
the_table.auto_set_font_size(False)
the_table.set_fontsize(10)
the_table.scale(1, 1.5)
plt.axis('off')
plt.title('Summary of Model Performance Metrics', fontsize=16)
plt.tight_layout()
plt.savefig('comparison_results/metrics_summary_table.png', dpi=300, bbox_inches='tight')

# Write the summary table to a text file
with open('comparison_results/metrics_summary.txt', 'w') as f:
    # Write header
    f.write(f"{columns[0]:<25} {columns[1]:<10} {columns[2]:<10} {columns[3]:<10} {columns[4]:<10} {columns[5]:<10} {columns[6]:<10}\n")
    f.write("-" * 85 + "\n")
    
    # Write data rows
    for row in summary_data:
        f.write(f"{row[0]:<25} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<10} {row[5]:<10} {row[6]:<10}\n")

print("All comparison visualizations have been saved to the 'comparison_results' directory.")
