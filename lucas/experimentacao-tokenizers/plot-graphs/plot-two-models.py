import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MaxNLocator
import pickle


def plot_compare_losses(m1_data, m2_data, m1_name="Model 1", m2_name="Model 2"):
    """
    Plots training and validation losses for two models side-by-side.
    m1_data and m2_data should be dictionaries containing: 
    'batches', 'train_losses', and 'val_losses'
    """
    # Expanded the figure size slightly to make room for 4 lines and a clear legend
    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    # --- MODEL 1 (Blue Shades) ---
    ax1.plot(m1_data['batches'], m1_data['train_losses'], 
             color='#2E6F40', linestyle='-', linewidth=2, label=f"{m1_name} Train")
    ax1.plot(m1_data['batches'], m1_data['val_losses'], 
             color='#2E6F40', linestyle='--', linewidth=1.5, label=f"{m1_name} Val")

    # --- MODEL 2 (Orange Shades) ---
    ax1.plot(m2_data['batches'], m2_data['train_losses'], 
             color='#FF2C2C', linestyle='-', linewidth=2, label=f"{m2_name} Train")
    ax1.plot(m2_data['batches'], m2_data['val_losses'], 
             color='#FF2C2C', linestyle='--', linewidth=1.5, label=f"{m2_name} Val")

    # Bottom Axis Styling
    ax1.set_xlabel("Batches/Epochs", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Place legend outside or inside depending on space (upper right is standard)
    ax1.legend(loc="upper right", frameon=True, facecolor='white', edgecolor='none')

    # Top Axis (Tokens Seen - 512 multiplier)
    def batch_to_tokens(x):
        return x * 512

    def tokens_to_batch(x):
        return x / 512

    ax2 = ax1.secondary_xaxis('top', functions=(batch_to_tokens, tokens_to_batch))
    ax2.set_xlabel("Tokens seen", fontsize=11)

    fig.tight_layout()
    plt.savefig("model_comparison.pdf", dpi=300)
    plt.show()


def load_data(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)[1:]

train_losses_gpt2 = load_data("train_losses_gpt2_0.pkl") + load_data("train_losses_gpt2.pkl")
val_losses_gpt2 = load_data("val_losses_gpt2_0.pkl") + load_data("val_losses_gpt2.pkl")

pt_data = {
        "batches": torch.linspace(0, 56000, 70), 
        "train_losses": load_data("train_losses_pt.pkl"), #+ [0 for x in range(55)],
        "val_losses": load_data("val_losses_pt.pkl")# + [0 for x in range(55)]
        }


gpt2_data = {
        "batches": torch.linspace(0, 102000, len(train_losses_gpt2)),
        "train_losses": train_losses_gpt2,
        "val_losses": val_losses_gpt2

        }
plot_compare_losses(pt_data, gpt2_data, m1_name="pt", m2_name="gpt2-en")
