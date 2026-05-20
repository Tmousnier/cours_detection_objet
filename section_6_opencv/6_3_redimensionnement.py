"""
====================================================
 Section 6.3 — Redimensionnement d'une image
====================================================
cv2.resize permet de changer la taille d'une image.
On peut spécifier une taille absolue ou un facteur d'échelle.

Interpolations :
  - INTER_AREA   : recommandée pour la réduction (évite le crénelage)
  - INTER_CUBIC  : recommandée pour l'agrandissement (plus fluide)
  - INTER_LINEAR : par défaut, bon compromis vitesse / qualité

ATTENTION : cv2.resize prend (largeur, hauteur),
            l'inverse de la forme NumPy (hauteur, largeur).
"""

import cv2
import matplotlib
matplotlib.use("Agg")   # Mode sans affichage (serveur / SSH)
import matplotlib.pyplot as plt

# ── 1. Lecture de l'image ───────────────────────────────────────────────────
img = cv2.imread("../Image/Harry_bebe.jpeg")

if img is None:
    raise FileNotFoundError("Image non trouvée. Vérifiez le chemin.")

hauteur, largeur = img.shape[:2]
print(f"Taille originale : {largeur}x{hauteur} px")

# ── 2. Redimensionnement à taille fixe (INTER_AREA pour réduire) ────────────
redimensionnee = cv2.resize(img, (128, 64), interpolation=cv2.INTER_AREA)
print("Taille 128x64 :", redimensionnee.shape)

# ── 3. Redimensionnement par facteur x2 (INTER_CUBIC pour agrandir) ─────────
x2 = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
print("Taille x2 :", x2.shape)

# ── 4. Redimensionnement par facteur x0.5 ───────────────────────────────────
moitie = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
print("Taille x0.5 :", moitie.shape)

# ── 5. Affichage comparatif ─────────────────────────────────────────────────
images = [img, redimensionnee, moitie]
titres  = [
    f"Originale\n{largeur}x{hauteur}",
    f"Réduite 128x64\n(INTER_AREA)",
    f"x0.5\n(INTER_AREA)",
]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, image, titre in zip(axes, images, titres):
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(titre)
    ax.axis("off")

plt.suptitle("Section 6.3 — Redimensionnement d'images", fontsize=14)
plt.tight_layout()
plt.savefig("../outputs/6_3_redimensionnement.png", dpi=130)
plt.close()
print("Figure sauvegardée : outputs/6_3_redimensionnement.png")

