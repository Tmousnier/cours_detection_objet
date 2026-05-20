"""
====================================================
 Section 6 — Script complet (pipeline complet)
====================================================
Ce script exécute toutes les étapes de la section 6
dans l'ordre : lecture → gris → resize → histogramme → seuillage
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

IMAGE_PATH = "../Image/Harry_bebe.jpeg"

# ───────────────────────────────────────────────────────────────────────────
# 1. Lecture
# ───────────────────────────────────────────────────────────────────────────
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Image introuvable : {IMAGE_PATH}")

print(f"[1] Image chargée  — forme : {img.shape}, dtype : {img.dtype}")

# ───────────────────────────────────────────────────────────────────────────
# 2. Niveaux de gris
# ───────────────────────────────────────────────────────────────────────────
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"[2] Niveaux de gris — forme : {gris.shape}")

# ───────────────────────────────────────────────────────────────────────────
# 3. Redimensionnement
# ───────────────────────────────────────────────────────────────────────────
img_resize = cv2.resize(img, (300, 300))
print(f"[3] Redimensionné   — forme : {img_resize.shape}")

# ───────────────────────────────────────────────────────────────────────────
# 4. Histogramme
# ───────────────────────────────────────────────────────────────────────────
hist = cv2.calcHist([gris], [0], None, [256], [0, 256])

# ───────────────────────────────────────────────────────────────────────────
# 5. Seuillage (Otsu)
# ───────────────────────────────────────────────────────────────────────────
seuil_otsu, thresh = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"[5] Seuil Otsu       : {seuil_otsu:.1f}")

# ───────────────────────────────────────────────────────────────────────────
# Affichage récapitulatif
# ───────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Pipeline complet — Section 6 OpenCV", fontsize=15, fontweight="bold")

ax1 = fig.add_subplot(2, 3, 1)
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title("1. Image originale")
ax1.axis("off")

ax2 = fig.add_subplot(2, 3, 2)
ax2.imshow(gris, cmap="gray")
ax2.set_title("2. Niveaux de gris")
ax2.axis("off")

ax3 = fig.add_subplot(2, 3, 3)
ax3.imshow(cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB))
ax3.set_title("3. Redimensionné (300×300)")
ax3.axis("off")

ax4 = fig.add_subplot(2, 3, 4)
ax4.plot(hist, color="black")
ax4.fill_between(np.arange(256), hist.flatten(), alpha=0.3, color="gray")
ax4.axvline(seuil_otsu, color="red", linestyle="--", label=f"Otsu={seuil_otsu:.0f}")
ax4.set_title("4. Histogramme")
ax4.set_xlabel("Intensité")
ax4.set_ylabel("Pixels")
ax4.legend()
ax4.set_xlim([0, 256])

ax5 = fig.add_subplot(2, 3, 5)
ax5.imshow(thresh, cmap="gray")
ax5.set_title(f"5. Seuillage Otsu ({seuil_otsu:.0f})")
ax5.axis("off")

plt.tight_layout()
plt.show()

