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

cv2.RETR_EXTERNAL  → récupère uniquement les contours externes
cv2.CHAIN_APPROX_SIMPLE → compresse les segments redondants
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Mode sans affichage (serveur / SSH)
import matplotlib.pyplot as plt

# ── 1. Lecture de l'image ───────────────────────────────────────────────────
img = cv2.imread("../Image/Harry_bebe.jpeg")

if img is None:
    raise FileNotFoundError("Image non trouvée. Vérifiez le chemin.")

gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ── 2. Seuillage binaire (prérequis pour findContours) ─────────────────────
_, binaire = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)

# ── 3. Détection des contours ───────────────────────────────────────────────
# findContours retourne la liste des contours et une hiérarchie
contours, hierarchy = cv2.findContours(
    binaire, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
print(f"Nombre d'objets détectés : {len(contours)}")

# ── 4. Extraction des boîtes englobantes ────────────────────────────────────
img_boites = img.copy()
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    print(f"  Objet {i:02d} : x={x}, y={y}, largeur={w}, hauteur={h}")
    cv2.rectangle(img_boites, (x, y), (x + w, y + h), (0, 255, 0), 2)

# ── 5. Détection Canny ───────────────────────────────────────────────────────
# Canny détecte les bords par gradient (deux seuils : bas et haut)
# seuil_bas=100, seuil_haut=200
canny = cv2.Canny(gris, 100, 200)
n_pixels_contour = cv2.countNonZero(canny)
print(f"Pixels de contour (Canny) : {n_pixels_contour}")

# ── 6. Affichage comparatif ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Image originale")
axes[0].axis("off")

axes[1].imshow(binaire, cmap="gray")
axes[1].set_title("Seuillage binaire")
axes[1].axis("off")

axes[2].imshow(canny, cmap="gray")
axes[2].set_title(f"Contours Canny\n({n_pixels_contour} pixels)")
axes[2].axis("off")

axes[3].imshow(cv2.cvtColor(img_boites, cv2.COLOR_BGR2RGB))
axes[3].set_title(f"Boîtes englobantes\n({len(contours)} objets)")
axes[3].axis("off")

plt.suptitle("Section 6.6 — Contours et extraction de boîtes", fontsize=14)
plt.tight_layout()
plt.savefig("../outputs/6_6_contours_boites.png", dpi=130)
plt.close()
print("Figure sauvegardée : outputs/6_6_contours_boites.png")

