import matplotlib.pyplot as plt

# Exact logged metrics from your PanFed-X federated training output
rounds = [1, 2, 3]
fed_loss = [1.0254, 1.0603, 1.0639]
fed_acc  = [48.53, 54.41, 48.53]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# --- Plot 1: Federated Aggregation Loss ---
ax1.plot(rounds, fed_loss, color='#1f77b4', marker='o', linewidth=2.4, markersize=8, label='Global FedAvg Loss')
ax1.set_title('Global Federated Loss Across Rounds', fontsize=12, fontweight='bold', pad=12)
ax1.set_xlabel('Federated Communication Round', fontsize=11)
ax1.set_ylabel('Cross-Entropy Loss', fontsize=11)
ax1.set_xticks(rounds)
ax1.set_ylim(0.95, 1.15)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper left', frameon=True)

# Highlight minimum loss (Round 1)
min_loss_idx = fed_loss.index(min(fed_loss))
ax1.annotate(f'Lowest Loss: {fed_loss[min_loss_idx]:.4f}',
             xy=(rounds[min_loss_idx], fed_loss[min_loss_idx]),
             xytext=(rounds[min_loss_idx] + 0.15, fed_loss[min_loss_idx] - 0.03),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=5))

# --- Plot 2: Federated Global Accuracy ---
ax2.plot(rounds, fed_acc, color='#2ca02c', marker='s', linewidth=2.4, markersize=8, label='Global Val Accuracy (%)')
ax2.set_title('Global Multi-Center Accuracy (%)', fontsize=12, fontweight='bold', pad=12)
ax2.set_xlabel('Federated Communication Round', fontsize=11)
ax2.set_ylabel('Accuracy (%)', fontsize=11)
ax2.set_xticks(rounds)
ax2.set_ylim(40, 65)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='upper right', frameon=True)

# Highlight peak accuracy (Round 2)
best_acc_idx = fed_acc.index(max(fed_acc))
ax2.annotate(f'Peak Acc: {fed_acc[best_acc_idx]:.2f}% (Round {rounds[best_acc_idx]})',
             xy=(rounds[best_acc_idx], fed_acc[best_acc_idx]),
             xytext=(rounds[best_acc_idx] - 0.45, fed_acc[best_acc_idx] + 3.0),
             arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=5))

plt.tight_layout()
output_filename = 'panfedx_federated_curves.png'
plt.savefig(output_filename, dpi=300)
print(f"[Done] Verified federated plot saved to: {output_filename}")
plt.show()