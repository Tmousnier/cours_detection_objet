"""
====================================================
 Section 6.2 — Conversion en niveaux de gris
====================================================
cv2.cvtColor applique une combinaison linéaire pondérée :
    Gris = 0.299*R + 0.587*V + 0.114*B
Ces poids reflètent la sensibilité de l'œil humain aux couleurs.
"""

import cv2
import matplotlib.pyplot as plt

# ── 1. Lecture de l'image ───────────────────────────────────────────────────
img = cv2.imread("../Image/Harry_bebe.jpeg")

if img is None:
    raise FileNotFoundError("Image non trouvée. Vérifiez le chemin.")

# ── 2. Conversion BGR → Niveaux de gris ────────────────────────────────────
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("Forme image couleur :", img.shape)   # (hauteur, largeur, 3)
print("Forme niveaux gris  :", gris.shape)  # (hauteur, largeur)  — 1 seul canal

# ── 3. Affichage comparatif ─────────────────────────────────────────────────
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(img_rgb)
axes[0].set_title("Image couleur (RGB)")
axes[0].axis("off")

axes[1].imshow(gris, cmap="gray")
axes[1].set_title("Niveaux de gris")
axes[1].axis("off")

plt.suptitle("Comparaison couleur vs niveaux de gris", fontsize=14)
plt.tight_layout()
plt.show()

