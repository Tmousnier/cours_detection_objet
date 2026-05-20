"""
====================================================
 Section 6.6 — Contours et extraction de boîtes
====================================================
Après le seuillage, on extrait les contours pour isoler
les objets. C'est le lien direct avec la détection (Jour 2).

Fonctions clés :
  - cv2.findContours  : détecte les contours dans une image binaire
  - cv2.boundingRect  : retourne la boîte englobante d'un contour
  - cv2.Canny         : détecteur de contours par gradient (Canny)
  - cv2.grabCut       : segmentation avant-plan / fond (plus précis)

cv2.RETR_EXTERNAL      → récupère uniquement les contours externes
cv2.CHAIN_APPROX_SIMPLE → compresse les segments redondants
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Mode sans affichage (serveur / SSH)
import matplotlib.pyplot as plt

# ── Chemins dynamiques ──────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_PATH = os.path.join(BASE_DIR, "Image", "Harry_bebe.jpeg")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Lecture de l'image ───────────────────────────────────────────────────
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Image non trouvée : {IMAGE_PATH}")

gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ── 2. Seuillage binaire (prérequis pour findContours basique) ──────────────
_, binaire = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)

# ── 3. Détection Canny ───────────────────────────────────────────────────────
# Canny détecte les bords par gradient (seuil_bas=100, seuil_haut=200)
canny = cv2.Canny(gris, 100, 200)
n_pixels_contour = cv2.countNonZero(canny)
print(f"Pixels de contour (Canny) : {n_pixels_contour}")

# ── 4. GrabCut — boîte englobante précise autour de l'animal ────────────────
# GrabCut sépare le sujet du fond en utilisant les couleurs (plus fiable
# que le seuillage simple sur des images à fond complexe).
h_img, w_img = img.shape[:2]
margin_x = int(w_img * 0.04)
margin_y = int(h_img * 0.04)

# Rectangle initial (x, y, largeur, hauteur) = zone probable de l'avant-plan
rect_gc   = (margin_x, margin_y, w_img - 2 * margin_x, h_img - 2 * margin_y)
mask_gc   = np.zeros((h_img, w_img), np.uint8)
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

cv2.grabCut(img, mask_gc, rect_gc, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# Masque binaire : 0/2 = fond → 0, 1/3 = avant-plan → 255
masque_fg = np.where((mask_gc == 2) | (mask_gc == 0), 0, 255).astype(np.uint8)

# Trouver les contours sur le masque GrabCut
contours_gc, _ = cv2.findContours(masque_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Nombre de régions GrabCut : {len(contours_gc)}")

# Garder les 5 plus grandes régions et fusionner en une seule boîte
contours_significatifs = sorted(contours_gc, key=cv2.contourArea, reverse=True)[:5]

img_boites = img.copy()
if contours_significatifs:
    x_min = min(cv2.boundingRect(c)[0] for c in contours_significatifs)
    y_min = min(cv2.boundingRect(c)[1] for c in contours_significatifs)
    x_max = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours_significatifs)
    y_max = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours_significatifs)
    cv2.rectangle(img_boites, (x_min, y_min), (x_max, y_max), (0, 255, 0), 4)
    print(f"Boîte englobante : x={x_min}, y={y_min}, w={x_max - x_min}, h={y_max - y_min}")

# ── 5. Affichage comparatif ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 6))

axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Image originale")
axes[0].axis("off")

axes[1].imshow(binaire, cmap="gray")
axes[1].set_title("Seuillage binaire\n(threshold=127)")
axes[1].axis("off")

axes[2].imshow(canny, cmap="gray")
axes[2].set_title(f"Contours Canny\n({n_pixels_contour} pixels de bord)")
axes[2].axis("off")

axes[3].imshow(cv2.cvtColor(img_boites, cv2.COLOR_BGR2RGB))
axes[3].set_title(f"Boîte englobante (GrabCut)\n({len(contours_significatifs)} régions fusionnées)")
axes[3].axis("off")

plt.suptitle("Section 6.6 — Contours et extraction de boîtes", fontsize=14, fontweight="bold")
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "6_6_contours_boites.png")
plt.savefig(output_path, dpi=130)
plt.close()
print(f"Figure sauvegardée : {output_path}")
