"""
====================================================
 Section 6.3 — Redimensionnement d'une image
====================================================
cv2.resize permet de changer la taille d'une image.
On peut spécifier une taille absolue ou un facteur d'échelle.
"""

import cv2
import matplotlib.pyplot as plt

# ── 1. Lecture de l'image ───────────────────────────────────────────────────
img = cv2.imread("../Image/Harry_bebe.jpeg")

if img is None:
    raise FileNotFoundError("Image non trouvée. Vérifiez le chemin.")

hauteur, largeur = img.shape[:2]
print(f"Taille originale : {largeur}x{hauteur} px")

# ── 2. Redimensionnement à taille fixe ─────────────────────────────────────
img_fixe = cv2.resize(img, (300, 300))
print("Taille fixe :", img_fixe.shape)

# ── 3. Redimensionnement par facteur d'échelle ──────────────────────────────
facteur = 0.5
img_moitie = cv2.resize(img, None, fx=facteur, fy=facteur)
print("Taille x0.5 :", img_moitie.shape)

# ── 4. Interpolation (important pour agrandir une image) ────────────────────
# INTER_LINEAR  : interpolation bilinéaire (défaut, bon équilibre)
# INTER_CUBIC   : bicubique (meilleure qualité, plus lent)
# INTER_NEAREST : plus proche voisin (rapide, pixelisé)
img_grand = cv2.resize(img, (largeur * 2, hauteur * 2), interpolation=cv2.INTER_CUBIC)
print("Taille x2 (cubic) :", img_grand.shape)

# ── 5. Affichage comparatif ─────────────────────────────────────────────────
images = [img, img_fixe, img_moitie]
titres = ["Originale", "Fixe 300x300", "x0.5"]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, image, titre in zip(axes, images, titres):
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{titre}\n{image.shape[1]}x{image.shape[0]}")
    ax.axis("off")

plt.suptitle("Redimensionnement d'images", fontsize=14)
plt.tight_layout()
plt.show()

