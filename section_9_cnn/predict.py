"""
====================================================
 LAB-02 — Prediction avec le CNN entraine
====================================================
Ce script charge le modele sauvegarde et effectue
une prediction sur une image de test choisie.

Usage :
  python section_9_cnn/predict.py

Prerequis :
  Avoir lance train_cnn.py au moins une fois
  pour generer models/cnn_fashion_mnist.pth
"""

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Chemins dynamiques ──────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "cnn_fashion_mnist.pth")
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Noms des classes ─────────────────────────────────────────────────────────
class_names = [
    "T-shirt/top", "Pantalon", "Pull",   "Robe",   "Manteau",
    "Sandale",     "Chemise",  "Basket", "Sac",    "Botte"
]

# ── Definition du modele (meme architecture que train_cnn.py) ────────────────
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1,  16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(32 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def predict_sample(n_images=16):
    """
    Charge le modele entraîne et predit sur n_images aleatoires du jeu de test.
    Affiche les probabilites de chaque classe pour la premiere image.
    """
    # ── Chargement du modele ─────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modele introuvable : {MODEL_PATH}\n"
            "Lance d'abord train_cnn.py"
        )

    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Modele charge depuis : {MODEL_PATH}")

    # ── Chargement du dataset de test ────────────────────────────────────────
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.FashionMNIST(
        root=DATA_DIR, train=False, download=True, transform=transform
    )

    # Selectionner n_images aleatoires
    indices = np.random.choice(len(test_dataset), n_images, replace=False)
    images_list, labels_list = [], []
    for idx in indices:
        img, lbl = test_dataset[idx]
        images_list.append(img)
        labels_list.append(lbl)

    images_tensor = torch.stack(images_list).to(DEVICE)
    labels_tensor = torch.tensor(labels_list).to(DEVICE)

    # ── Predictions ──────────────────────────────────────────────────────────
    with torch.no_grad():
        outputs     = model(images_tensor)
        proba       = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    # ── Figure 1 : grille de predictions ─────────────────────────────────────
    cols = 4
    rows = n_images // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5))
    fig.suptitle("LAB-02 — Predictions sur nouvelles images (jeu de test)",
                 fontsize=13, fontweight="bold")

    for i in range(n_images):
        ax  = axes[i // cols][i % cols]
        img = images_list[i].squeeze().numpy()
        img = (img * 0.5) + 0.5  # denormalisation

        pred   = predicted[i].item()
        reel   = labels_list[i]
        conf   = proba[i][pred].item() * 100
        color  = "green" if pred == reel else "red"

        ax.imshow(img, cmap="gray")
        ax.set_title(
            f"Predit : {class_names[pred]} ({conf:.0f}%)\nReel : {class_names[reel]}",
            fontsize=8, color=color
        )
        ax.axis("off")

    plt.tight_layout()
    out1 = os.path.join(OUTPUT_DIR, "9_4_predict_grille.png")
    plt.savefig(out1, dpi=130)
    plt.close()
    print(f"Grille de predictions sauvegardee : {out1}")

    # ── Figure 2 : probabilites pour la 1re image ───────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle("LAB-02 — Detail d'une prediction (probabilites par classe)",
                  fontsize=12, fontweight="bold")

    img0   = images_list[0].squeeze().numpy()
    img0   = (img0 * 0.5) + 0.5
    prob0  = proba[0].cpu().numpy() * 100
    pred0  = predicted[0].item()
    reel0  = labels_list[0]

    axes2[0].imshow(img0, cmap="gray")
    axes2[0].set_title(f"Image\nReel : {class_names[reel0]}", fontsize=10)
    axes2[0].axis("off")

    colors_bar = ["green" if i == pred0 else "steelblue" for i in range(10)]
    bars = axes2[1].barh(class_names, prob0, color=colors_bar)
    axes2[1].set_xlabel("Probabilite (%)")
    axes2[1].set_xlim([0, 100])
    axes2[1].set_title(f"Probabilites par classe\nPredit : {class_names[pred0]} ({prob0[pred0]:.1f}%)",
                       fontsize=10)
    axes2[1].axvline(50, color="orange", linestyle="--", alpha=0.5, label="seuil 50%")
    for bar, val in zip(bars, prob0):
        if val > 2:
            axes2[1].text(val + 0.5, bar.get_y() + bar.get_height()/2,
                          f"{val:.1f}%", va="center", fontsize=8)
    axes2[1].grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    out2 = os.path.join(OUTPUT_DIR, "9_5_predict_probabilites.png")
    plt.savefig(out2, dpi=130)
    plt.close()
    print(f"Probabilites sauvegardees : {out2}")

    # ── Resume console ────────────────────────────────────────────────────────
    corrects = (predicted == labels_tensor).sum().item()
    print(f"\nResultats sur {n_images} images aleatoires :")
    print(f"  Corrects : {corrects}/{n_images} ({100*corrects/n_images:.0f}%)")


if __name__ == "__main__":
    predict_sample(n_images=16)

