"""
====================================================
 Section 8.1 — HOG (Histogram of Oriented Gradients)
====================================================
HOG résume la structure globale des contours d'une image
sous forme d'histogramme d'orientations de gradients.

Pipeline HOG :
  1) Calcul des gradients (Sobel X et Y)
  2) Division en cellules (8×8 px)
  3) Histogramme d'orientations sur 9 bins (0°–180°)
  4) Normalisation par blocs de cellules

Basé sur : https://github.com/yugmerabtene/nexa-computer-vision/blob/main/JOUR-01.md
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Chemins dynamiques ──────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_PATH = os.path.join(BASE_DIR, "Image", "Harry_bebe.jpeg")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT ET PRÉPARATION
# ═══════════════════════════════════════════════════════════════════════════
img_real = cv2.imread(IMAGE_PATH)
if img_real is None:
    raise FileNotFoundError(f"Image non trouvée : {IMAGE_PATH}")

# HOG exige une fenêtre de taille fixe — on redimensionne à (128, 64)
# ATTENTION : cv2.resize prend (largeur, hauteur) ≠ forme NumPy (hauteur, largeur)
img_resize = cv2.resize(img_real, (128, 64), interpolation=cv2.INTER_AREA)
gris       = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

print(f"Image originale  : {img_real.shape}")
print(f"Image HOG (gris) : {gris.shape}  ← fenêtre 128×64 obligatoire")

# ═══════════════════════════════════════════════════════════════════════════
# 2. CALCUL DU DESCRIPTEUR HOG
# ═══════════════════════════════════════════════════════════════════════════
hog = cv2.HOGDescriptor(
    _winSize   =(128, 64),   # Taille de la fenêtre
    _blockSize =(16, 16),    # Taille d'un bloc (en pixels)
    _blockStride=(8,  8),    # Décalage entre blocs (chevauchement 50%)
    _cellSize  =(8,   8),    # Taille d'une cellule
    _nbins     =9,           # Nombre de bins d'orientation (0°–180°)
)

descripteur = hog.compute(gris)
print(f"\nDimension du descripteur HOG : {descripteur.shape[0]}")
print(f"  → (128-16)/8+1 = 15 blocs en x")
print(f"  → ( 64-16)/8+1 =  7 blocs en y")
print(f"  → 15 × 7 × 4 cellules/bloc × 9 bins = {15*7*4*9}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. CALCUL DES GRADIENTS (visualisation manuelle)
# ═══════════════════════════════════════════════════════════════════════════
grad_x = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=3)

magnitude  = np.sqrt(grad_x**2 + grad_y**2)
orientation = np.arctan2(np.abs(grad_y), np.abs(grad_x)) * 180 / np.pi

mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

print(f"\nGradient magnitude : min={magnitude.min():.1f}, max={magnitude.max():.1f}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. COMPARAISON HOG — forme similaire vs forme différente
# ═══════════════════════════════════════════════════════════════════════════
# Image synthétique : rectangle blanc sur fond noir
img_rect = np.zeros((200, 300, 3), dtype=np.uint8)
cv2.rectangle(img_rect, (50, 40), (250, 160), (255, 255, 255), -1)

# Variante décalée (légèrement différente)
img_shifted = np.zeros((200, 300, 3), dtype=np.uint8)
cv2.rectangle(img_shifted, (55, 42), (255, 162), (255, 255, 255), -1)

# Forme différente : cercle
img_cercle = np.zeros((200, 300, 3), dtype=np.uint8)
cv2.circle(img_cercle, (150, 100), 70, (255, 255, 255), -1)

def hog_descriptor(image_bgr):
    """Calcule le descripteur HOG d'une image BGR."""
    g = cv2.cvtColor(cv2.resize(image_bgr, (128, 64), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY)
    return hog.compute(g).flatten()

desc_rect    = hog_descriptor(img_rect)
desc_shifted = hog_descriptor(img_shifted)
desc_cercle  = hog_descriptor(img_cercle)

dist_shifted  = np.linalg.norm(desc_rect - desc_shifted)
dist_different = np.linalg.norm(desc_rect - desc_cercle)

print(f"\nDistance HOG (rect ↔ rect décalé) : {dist_shifted:.2f}")
print(f"Distance HOG (rect ↔ cercle)       : {dist_different:.2f}")
print(f"Séparation OK : {dist_different > dist_shifted}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. AFFICHAGE — figure complète
# ═══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 12))
fig.suptitle("Section 8.1 — HOG (Histogram of Oriented Gradients)",
             fontsize=15, fontweight="bold")

# Ligne 1 : pipeline sur image réelle
ax1 = fig.add_subplot(2, 4, 1)
ax1.imshow(cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB))
ax1.set_title("1. Image redim. (128×64)")
ax1.axis("off")

ax2 = fig.add_subplot(2, 4, 2)
ax2.imshow(gris, cmap="gray")
ax2.set_title("2. Niveaux de gris")
ax2.axis("off")

ax3 = fig.add_subplot(2, 4, 3)
ax3.imshow(mag_norm, cmap="hot")
ax3.set_title("3. Magnitude des gradients\n(zones de bords = blanc)")
ax3.axis("off")

ax4 = fig.add_subplot(2, 4, 4)
ax4.plot(descripteur[:100], color="steelblue", linewidth=1)
ax4.set_title(f"4. Descripteur HOG\n({descripteur.shape[0]} dimensions)")
ax4.set_xlabel("Dimension")
ax4.set_ylabel("Valeur")
ax4.grid(True, alpha=0.3)

# Ligne 2 : comparaison formes
ax5 = fig.add_subplot(2, 4, 5)
ax5.imshow(cv2.cvtColor(img_rect, cv2.COLOR_BGR2RGB))
ax5.set_title("5. Rectangle (référence)")
ax5.axis("off")

ax6 = fig.add_subplot(2, 4, 6)
ax6.imshow(cv2.cvtColor(img_shifted, cv2.COLOR_BGR2RGB))
ax6.set_title(f"6. Rectangle décalé\nDist HOG = {dist_shifted:.2f}")
ax6.axis("off")

ax7 = fig.add_subplot(2, 4, 7)
ax7.imshow(cv2.cvtColor(img_cercle, cv2.COLOR_BGR2RGB))
ax7.set_title(f"7. Cercle (forme différente)\nDist HOG = {dist_different:.2f}")
ax7.axis("off")

ax8 = fig.add_subplot(2, 4, 8)
categories = ["Rect ↔ Rect\n(décalé)", "Rect ↔\nCercle"]
distances  = [dist_shifted, dist_different]
colors     = ["#27ae60", "#e74c3c"]
bars = ax8.bar(categories, distances, color=colors, width=0.5, edgecolor="black")
ax8.set_title("8. Distances L2 entre HOG\n(plus grand = plus différent)")
ax8.set_ylabel("Distance L2")
ax8.grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars, distances):
    ax8.text(bar.get_x() + bar.get_width()/2, val + 0.1,
             f"{val:.2f}", ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "8_1_hog.png")
plt.savefig(output_path, dpi=130)
plt.close()
print(f"\nFigure sauvegardee : {output_path}")

