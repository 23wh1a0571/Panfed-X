import matplotlib.pyplot as plt

# Replace these values with your actual logged numbers from training
epochs = list(range(1, 16))

train_loss = [1.08, 0.94, 0.81, 0.72, 0.65, 0.58, 0.52, 0.47, 0.43, 0.39, 0.36, 0.33, 0.31, 0.28, 0.26]
val_loss   = [1.12, 0.99, 0.88, 0.79, 0.74, 0.69, 0.65, 0.63, 0.60, 0.58, 0.57, 0.55, 0.56, 0.54, 0.53]

train_acc  = [38.2, 51.5, 62.0, 68.4, 73.1, 77.5, 81.2, 84.0, 86.5, 88.2, 90.1, 91.5, 92.8, 94.0, 95.2]
val_acc    = [36.0, 48.5, 58.0, 64.2, 69.0, 71.5, 74.0, 75.8, 77.2, 78.5, 79.0, 80.2, 80.0, 81.5, 82.1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 1. Loss Curve
ax1.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2)
ax1.plot(epochs, val_loss, 'r--s', label='Validation Loss', linewidth=2)
ax1.set_title('DenseNet-121 Loss Curve', fontsize=13, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Cross-Entropy Loss', fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(fontsize=10)

# 2. Accuracy Curve
ax2.plot(epochs, train_acc, 'b-o', label='Training Accuracy', linewidth=2)
ax2.plot(epochs, val_acc, 'g--^', label='Validation Accuracy', linewidth=2)
ax2.set_title('DenseNet-121 Classification Accuracy', fontsize=13, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Accuracy (%)', fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300)
plt.show()
print("[Success] Loss curve saved as 'training_curves.png'")