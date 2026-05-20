"""
====================================================
 Section 6.4 — Histogramme et égalisation
====================================================
L'histogramme représente la distribution des intensités
des pixels (0 = noir, 255 = blanc).

cv2.equalizeHist redistribue les intensités pour améliorer
le contraste d'une image sous-exposée.
"""

import cv2
import matplotlib
matplotlib.use("Agg")   # Mode sans affichage (serveur / SSH)
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

# ── 3. Égalisation d'histogramme ────────────────────────────────────────────
# cv2.equalizeHist applique une transformation cumulative sur l'histogramme.
# Les pixels sombres deviennent plus variés ; les détails cachés apparaissent.
gris_equalise = cv2.equalizeHist(gris)
hist_equalise  = cv2.calcHist([gris_equalise], [0], None, [256], [0, 256])

print("Avant égalisation  : min =", gris.min(),         "max =", gris.max())
print("Après égalisation  : min =", gris_equalise.min(), "max =", gris_equalise.max())

# ── 4. Histogramme par canal BGR ────────────────────────────────────────────
couleurs      = ("b", "g", "r")
noms_canaux   = ("Bleu (B)", "Vert (G)", "Rouge (R)")

# ── 5. Affichage ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# Ligne 1 — Histogramme de base
axes[0][0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0][0].set_title("Image originale")
axes[0][0].axis("off")

axes[0][1].plot(hist_gris, color="black")
axes[0][1].fill_between(np.arange(256), hist_gris.flatten(), alpha=0.3, color="gray")
axes[0][1].set_title("Histogramme — Niveaux de gris")
axes[0][1].set_xlabel("Intensité (0-255)")
axes[0][1].set_ylabel("Nombre de pixels")
axes[0][1].set_xlim([0, 256])

for i, (couleur, nom) in enumerate(zip(couleurs, noms_canaux)):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    axes[0][2].plot(hist, color=couleur, label=nom, alpha=0.8)
axes[0][2].set_title("Histogramme — Canaux BGR")
axes[0][2].set_xlabel("Intensité (0-255)")
axes[0][2].set_ylabel("Nombre de pixels")
axes[0][2].set_xlim([0, 256])
axes[0][2].legend()

# Ligne 2 — Égalisation
axes[1][0].imshow(gris, cmap="gray")
axes[1][0].set_title("Avant égalisation")
axes[1][0].axis("off")

axes[1][1].imshow(gris_equalise, cmap="gray")
axes[1][1].set_title("Après égalisation")
axes[1][1].axis("off")

axes[1][2].plot(hist_gris,    color="gray",  label="Avant",  alpha=0.7)
axes[1][2].plot(hist_equalise, color="blue", label="Après",  alpha=0.7)
axes[1][2].set_title("Histogramme avant / après égalisation")
axes[1][2].set_xlabel("Intensité (0-255)")
axes[1][2].set_ylabel("Nombre de pixels")
axes[1][2].set_xlim([0, 256])
axes[1][2].legend()

plt.suptitle("Section 6.4 — Histogramme et égalisation", fontsize=14)
plt.tight_layout()
plt.savefig("../outputs/6_4_histogramme.png", dpi=130)
plt.close()
print("Figure sauvegardée : outputs/6_4_histogramme.png")

