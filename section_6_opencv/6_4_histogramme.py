"""
====================================================
 Section 6.4 — Histogramme d'une image
====================================================
L'histogramme représente la distribution des intensités
des pixels (0 = noir, 255 = blanc).
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ── 1. Lecture de l'image ───────────────────────────────────────────────────
img = cv2.imread("../Image/Harry_bebe.jpeg")

if img is None:
    raise FileNotFoundError("Image non trouvée. Vérifiez le chemin.")

gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ── 2. Calcul de l'histogramme (niveaux de gris) ───────────────────────────
# calcHist(images, channels, mask, histSize, ranges)
hist_gris = cv2.calcHist([gris], [0], None, [256], [0, 256])

# ── 3. Histogramme par canal BGR ────────────────────────────────────────────
couleurs = ("b", "g", "r")
noms_canaux = ("Bleu (B)", "Vert (G)", "Rouge (R)")

# ── 4. Affichage ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Image originale
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Image originale")
axes[0].axis("off")

# Histogramme niveaux de gris
axes[1].plot(hist_gris, color="black")
axes[1].fill_between(np.arange(256), hist_gris.flatten(), alpha=0.3, color="gray")
axes[1].set_title("Histogramme — Niveaux de gris")
axes[1].set_xlabel("Intensité (0-255)")
axes[1].set_ylabel("Nombre de pixels")
axes[1].set_xlim([0, 256])

# Histogramme couleurs BGR
for i, (couleur, nom) in enumerate(zip(couleurs, noms_canaux)):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    axes[2].plot(hist, color=couleur, label=nom, alpha=0.8)

axes[2].set_title("Histogramme — Canaux BGR")
axes[2].set_xlabel("Intensité (0-255)")
axes[2].set_ylabel("Nombre de pixels")
axes[2].set_xlim([0, 256])
axes[2].legend()

plt.suptitle("Analyse d'histogramme", fontsize=14)
plt.tight_layout()
plt.show()

