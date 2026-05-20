# -*- coding: utf-8 -*-
"""
====================================================
 Section 8.1 - HOG (Histogram of Oriented Gradients)
====================================================
HOG resume la structure globale des contours d'une image
sous forme d'histogramme d'orientations de gradients.

Pipeline HOG :
  1) Calcul des gradients (Sobel X et Y)
  2) Division en cellules (8x8 px)
  3) Histogramme d'orientations sur 9 bins (0 a 180 degres)
  4) Normalisation par blocs de cellules

Utilise les 2 vraies photos du chien :
  - Photo 1 : Harry_bebe.jpeg    (reference)
  - Photo 2 : Harry_bebe (1).png (similaire - meme chien)

Base sur : https://github.com/yugmerabtene/nexa-computer-vision/blob/main/JOUR-01.md
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- Chemins dynamiques -------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_PATH1 = os.path.join(BASE_DIR, "Image", "Harry_bebe.jpeg")
IMAGE_PATH2 = os.path.join(BASE_DIR, "Image", "Harry_bebe (1).png")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# UTILITAIRES
# =============================================================================
def imread_safe(path):
    """Lit une image meme si le chemin contient des espaces (Windows)."""
    arr = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# =============================================================================
# 1. CHARGEMENT ET PREPARATION
# =============================================================================
print("=" * 55)
print("  Chargement des images")
print("=" * 55)

img1 = imread_safe(IMAGE_PATH1)
img2 = imread_safe(IMAGE_PATH2)

if img1 is None: raise FileNotFoundError(f"Image introuvable : {IMAGE_PATH1}")
if img2 is None: raise FileNotFoundError(f"Image introuvable : {IMAGE_PATH2}")

# Aligner les tailles
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)

print(f"  Photo 1 : {img1.shape}  -- {os.path.basename(IMAGE_PATH1)}")
print(f"  Photo 2 : {img2.shape}  -- {os.path.basename(IMAGE_PATH2)}")

# HOG impose une fenetre de taille fixe (128 x 64)
# ATTENTION : cv2.resize prend (largeur, hauteur) != forme NumPy (hauteur, largeur)
hog_w, hog_h = 128, 64
gris1 = cv2.cvtColor(cv2.resize(img1, (hog_w, hog_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
gris2 = cv2.cvtColor(cv2.resize(img2, (hog_w, hog_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)

print(f"\n  Redim. HOG : {hog_w}x{hog_h} px  (fenetre obligatoire)")


# =============================================================================
# 2. DESCRIPTEUR HOG
# =============================================================================
hog = cv2.HOGDescriptor(
    _winSize   =(hog_w, hog_h),
    _blockSize =(16, 16),
    _blockStride=(8,  8),
    _cellSize  =(8,   8),
    _nbins     =9,
)

desc1 = hog.compute(gris1).flatten()
desc2 = hog.compute(gris2).flatten()

print("\n" + "=" * 55)
print("  Descripteur HOG")
print("=" * 55)
print(f"  Dimension : {desc1.shape[0]}")
print(f"    (128-16)/8+1 = 15 blocs en x")
print(f"    ( 64-16)/8+1 =  7 blocs en y")
print(f"    15 x 7 x 4 cellules x 9 bins = {15*7*4*9}")


# =============================================================================
# 3. GRADIENTS (visualisation manuelle)
# =============================================================================
def compute_gradients(gris):
    gx  = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
    gy  = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return mag, mag_norm

mag1, mag_norm1 = compute_gradients(gris1)
mag2, mag_norm2 = compute_gradients(gris2)


# =============================================================================
# 4. COMPARAISON HOG - photos similaires vs image differente
# =============================================================================
# Image synthetique tres differente (cercle blanc) pour le controle
img_cercle = np.zeros((200, 300, 3), dtype=np.uint8)
cv2.circle(img_cercle, (150, 100), 70, (255, 255, 255), -1)
gris_cercle = cv2.cvtColor(cv2.resize(img_cercle, (hog_w, hog_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
desc_cercle = hog.compute(gris_cercle).flatten()

dist_similar   = float(np.linalg.norm(desc1 - desc2))
dist_different = float(np.linalg.norm(desc1 - desc_cercle))

print("\n" + "=" * 55)
print("  Comparaison distances L2")
print("=" * 55)
print(f"  Photo 1 <-> Photo 2 (meme chien) : {dist_similar:.2f}")
print(f"  Photo 1 <-> Cercle  (different)  : {dist_different:.2f}")
print(f"  Separation OK : {dist_different > dist_similar}")


# =============================================================================
# 5. FIGURE COMPLETE - 8 panneaux
# =============================================================================
fig = plt.figure(figsize=(22, 12))
fig.suptitle("Section 8.1 - HOG sur 2 vraies photos du meme chien", fontsize=15, fontweight="bold")

# -- Ligne 1 : pipeline photo 1 -----------------------------------------------
ax1 = fig.add_subplot(2, 4, 1)
ax1.imshow(cv2.cvtColor(cv2.resize(img1, (hog_w, hog_h)), cv2.COLOR_BGR2RGB))
ax1.set_title(f"1. Photo 1 (redim. {hog_w}x{hog_h})\n{os.path.basename(IMAGE_PATH1)}")
ax1.axis("off")

ax2 = fig.add_subplot(2, 4, 2)
ax2.imshow(gris1, cmap="gray")
ax2.set_title("2. Photo 1 - Niveaux de gris")
ax2.axis("off")

ax3 = fig.add_subplot(2, 4, 3)
ax3.imshow(mag_norm1, cmap="hot")
ax3.set_title("3. Photo 1 - Magnitude gradients\n(bords = zones claires)")
ax3.axis("off")

ax4 = fig.add_subplot(2, 4, 4)
ax4.plot(desc1[:120], color="steelblue", linewidth=1.2, label="Photo 1", alpha=0.9)
ax4.plot(desc2[:120], color="green",     linewidth=1.2, label="Photo 2", alpha=0.7)
ax4.set_title(f"4. Descripteurs HOG\n({desc1.shape[0]} dims  - 120 premieres)")
ax4.set_xlabel("Dimension")
ax4.set_ylabel("Valeur")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# -- Ligne 2 : pipeline photo 2 + comparaison ---------------------------------
ax5 = fig.add_subplot(2, 4, 5)
ax5.imshow(cv2.cvtColor(cv2.resize(img2, (hog_w, hog_h)), cv2.COLOR_BGR2RGB))
ax5.set_title(f"5. Photo 2 (redim. {hog_w}x{hog_h})\n{os.path.basename(IMAGE_PATH2)}")
ax5.axis("off")

ax6 = fig.add_subplot(2, 4, 6)
ax6.imshow(gris2, cmap="gray")
ax6.set_title("6. Photo 2 - Niveaux de gris")
ax6.axis("off")

ax7 = fig.add_subplot(2, 4, 7)
ax7.imshow(mag_norm2, cmap="hot")
ax7.set_title("7. Photo 2 - Magnitude gradients")
ax7.axis("off")

ax8 = fig.add_subplot(2, 4, 8)
categories = ["Photo1 <-> Photo2\n(meme chien)", "Photo1 <-> Cercle\n(different)"]
distances  = [dist_similar, dist_different]
colors     = ["#27ae60", "#e74c3c"]
bars = ax8.bar(categories, distances, color=colors, edgecolor="black", width=0.5)
for bar, val in zip(bars, distances):
    ax8.text(bar.get_x() + bar.get_width()/2, val + 0.3,
             f"{val:.2f}", ha="center", fontweight="bold", fontsize=11)
ax8.set_title(f"8. Distances HOG L2\n(similaire {dist_similar:.2f} < different {dist_different:.2f})")
ax8.set_ylabel("Distance L2")
ax8.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "8_1_hog.png")
plt.savefig(output_path, dpi=130)
plt.close()
print(f"\nFigure sauvegardee : {output_path}")

# =============================================================================
# 6. CONCLUSION / INTERPRETATION
# =============================================================================
ratio = dist_different / dist_similar if dist_similar > 0 else float("inf")

print("\n" + "=" * 55)
print("  CONCLUSION - Ce que HOG nous apprend")
print("=" * 55)
print(f"""
  Distances L2 entre descripteurs HOG (3780 dimensions) :
  --------------------------------------------------------
  Photo 1 <-> Photo 2 (meme chien)  : {dist_similar:.2f}   [FAIBLE]
  Photo 1 <-> Cercle  (different)   : {dist_different:.2f}  [ELEVE]
  --------------------------------------------------------
  Ratio diferent / similaire        : x{ratio:.1f}

  Ce qu'on en deduit :
  1) HOG encode la STRUCTURE DES CONTOURS d'une image.
     Les 2 photos du chien ont des gradients similaires
     (fourrure, silhouette, oreilles) -> distance faible.

  2) Un cercle synthetique a une structure completement
     differente -> distance {ratio:.1f}x plus grande.

  3) HOG peut donc distinguer des formes differentes,
     mais il N'EST PAS invariant au changement d'echelle
     ni a la rotation (contrairement a SIFT).

  4) Utilisation classique : detecter des pietons (forme
     verticale caracteristique) avec un classifieur SVM.

  Checkpoint : dist_different ({dist_different:.2f}) > dist_similar ({dist_similar:.2f}) ?
  -> {"OUI - HOG separe correctement les sujets" if dist_different > dist_similar else "NON - probleme de separation"}
""")


