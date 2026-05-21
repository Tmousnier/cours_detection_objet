"""
====================================================
 LAB-02 — Classification d'images avec un CNN
====================================================
Dataset : Fashion-MNIST (10 classes de vetements)

Pipeline CNN :
  Image (28x28)
  -> Convolution (16 filtres 3x3) -> ReLU -> MaxPooling
  -> Convolution (32 filtres 3x3) -> ReLU -> MaxPooling
  -> Flatten
  -> Fully Connected (128 neurones) -> ReLU
  -> Sortie (10 classes)

Source du cours :
  https://github.com/yugmerabtene/nexa-computer-vision/blob/main/LAB-02.md
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Chemins dynamiques ──────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
MODELS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ── 1. Configuration generale ───────────────────────────────────────────────
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE    = 64
EPOCHS        = 5
LEARNING_RATE = 0.001

print(f"Device utilise : {DEVICE}")

# ── 2. Pretraitement des images ──────────────────────────────────────────────
# Fashion-MNIST : images en niveaux de gris de taille 28 x 28
# ToTensor   : transforme l'image en tenseur PyTorch (valeurs entre 0 et 1)
# Normalize  : centre et reduit (-1 a 1) pour stabiliser l'apprentissage
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ── 3. Chargement des datasets ───────────────────────────────────────────────
print("Telechargement de Fashion-MNIST...")
train_dataset = datasets.FashionMNIST(
    root=DATA_DIR, train=True,  download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root=DATA_DIR, train=False, download=True, transform=transform
)
print(f"  Train : {len(train_dataset)} images")
print(f"  Test  : {len(test_dataset)} images")

# ── 4. DataLoaders ───────────────────────────────────────────────────────────
# Le DataLoader envoie les images par paquets (batch) au modele
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ── 5. Noms des classes Fashion-MNIST ────────────────────────────────────────
class_names = [
    "T-shirt/top", "Pantalon", "Pull",    "Robe",   "Manteau",
    "Sandale",     "Chemise",  "Basket",  "Sac",    "Botte"
]

# ── 6. Architecture CNN ──────────────────────────────────────────────────────
class SimpleCNN(nn.Module):
    """
    CNN simple a 2 blocs convolutionnels.

    Flux des donnees :
      Entree  : (batch, 1, 28, 28)
      conv1   : (batch, 16, 28, 28) -> pool : (batch, 16, 14, 14)
      conv2   : (batch, 32, 14, 14) -> pool : (batch, 32,  7,  7)
      Flatten : (batch, 32*7*7) = (batch, 1568)
      fc1     : (batch, 128)
      fc2     : (batch, 10)   <- logits (une valeur par classe)
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Bloc 1 — Convolution 1 canal -> 16 cartes, kernel 3x3
        self.conv1 = nn.Conv2d(in_channels=1,  out_channels=16, kernel_size=3, padding=1)

        # Bloc 2 — Convolution 16 cartes -> 32 cartes, kernel 3x3
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # ReLU : garde les valeurs positives, met a 0 les negatives
        self.relu = nn.ReLU()

        # MaxPooling 2x2 : divise la resolution par 2 a chaque appel
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Couche fully connected : 32 * 7 * 7 = 1568 -> 128 neurones
        self.fc1 = nn.Linear(32 * 7 * 7, 128)

        # Couche de sortie : 128 -> 10 classes
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Bloc convolutionnel 1
        x = self.conv1(x)     # convolution
        x = self.relu(x)      # activation non-lineaire
        x = self.pool(x)      # reduction spatiale

        # Bloc convolutionnel 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten : cartes -> vecteur 1D pour les couches denses
        x = x.view(x.size(0), -1)

        # Couches fully connected
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x  # logits (CrossEntropyLoss les convertit en probabilites)


# ── 7. Creation du modele ────────────────────────────────────────────────────
model     = SimpleCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()   # adapte a la classification multi-classes
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\nArchitecture CNN :")
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Nombre total de parametres : {total_params:,}")


# ── 8. Boucle d'entrainement ─────────────────────────────────────────────────
def train():
    model.train()
    history = {"loss": [], "accuracy": []}

    for epoch in range(EPOCHS):
        running_loss        = 0.0
        correct_predictions = 0
        total_images        = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()          # reinitialisation des gradients
            outputs = model(images)        # passage avant (prediction)
            loss    = criterion(outputs, labels)  # calcul de l'erreur
            loss.backward()                # retropropagation
            optimizer.step()              # mise a jour des poids

            running_loss        += loss.item()
            _, predicted         = torch.max(outputs, 1)
            total_images        += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        accuracy = 100 * correct_predictions / total_images
        history["loss"].append(avg_loss)
        history["accuracy"].append(accuracy)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    return history


# ── 9. Evaluation sur le jeu de test ────────────────────────────────────────
def evaluate():
    model.eval()
    correct_predictions = 0
    total_images        = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs     = model(images)
            _, predicted = torch.max(outputs, 1)
            total_images        += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = 100 * correct_predictions / total_images
    print(f"\nAccuracy sur le jeu de test : {accuracy:.2f}%")
    return accuracy


# ── 10. Visualisation des courbes d'apprentissage ───────────────────────────
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("LAB-02 — Courbes d'apprentissage CNN (Fashion-MNIST)", fontsize=13, fontweight="bold")

    epochs_range = range(1, EPOCHS + 1)

    axes[0].plot(epochs_range, history["loss"], "b-o", linewidth=2, markersize=6)
    axes[0].set_title("Loss (erreur)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CrossEntropy Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(epochs_range)

    axes[1].plot(epochs_range, history["accuracy"], "g-o", linewidth=2, markersize=6)
    axes[1].set_title("Accuracy (precision)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim([0, 100])
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(epochs_range)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "9_1_courbes_apprentissage.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"Courbes sauvegardees : {out}")


# ── 11. Affichage de predictions (8 images) ──────────────────────────────────
def show_predictions():
    model.eval()
    images, labels = next(iter(test_loader))
    images_dev = images.to(DEVICE)

    with torch.no_grad():
        outputs   = model(images_dev)
        _, predicted = torch.max(outputs, 1)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle("LAB-02 — Predictions du CNN sur Fashion-MNIST", fontsize=13, fontweight="bold")

    for idx in range(8):
        ax  = axes[idx // 4][idx % 4]
        img = images[idx].squeeze().numpy()
        img = (img * 0.5) + 0.5  # denormalisation -> [0, 1]

        pred   = predicted[idx].item()
        reel   = labels[idx].item()
        couleur = "green" if pred == reel else "red"

        ax.imshow(img, cmap="gray")
        ax.set_title(
            f"Predit : {class_names[pred]}\nReel   : {class_names[reel]}",
            fontsize=9, color=couleur
        )
        ax.axis("off")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "9_2_predictions.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"Predictions sauvegardees : {out}")


# ── 12. Visualisation des filtres de la 1re couche ───────────────────────────
def show_filters():
    filtres = model.conv1.weight.data.cpu().numpy()  # (16, 1, 3, 3)
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    fig.suptitle("LAB-02 — Les 16 filtres de conv1 (appris)", fontsize=12, fontweight="bold")

    for i in range(16):
        ax = axes[i // 8][i % 8]
        f  = filtres[i, 0]  # filtre 3x3
        vmax = max(abs(f.min()), abs(f.max()))
        ax.imshow(f, cmap="RdBu", vmin=-vmax, vmax=vmax)
        ax.set_title(f"F{i+1}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "9_3_filtres_conv1.png")
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"Filtres sauvegardes : {out}")


# ── Lancement ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  ENTRAINEMENT DU CNN — Fashion-MNIST")
    print("="*55)

    history  = train()
    accuracy = evaluate()

    print("\nGeneration des figures...")
    plot_history(history)
    show_predictions()
    show_filters()

    # Sauvegarde du modele
    model_path = os.path.join(MODELS_DIR, "cnn_fashion_mnist.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nModele sauvegarde : {model_path}")
    print("\n" + "="*55)
    print(f"  TERMINE | Accuracy finale : {accuracy:.2f}%")
    print("="*55)

