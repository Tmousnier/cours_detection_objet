"""
====================================================
 Section 6.5 — Seuillage (Thresholding)
====================================================
Le seuillage binarise une image : chaque pixel devient
blanc (255) ou noir (0) selon un seuil défini.
"""

import os
import cv2
import matplotlib
matplotlib.use("Agg")   # Mode sans affichage (serveur / SSH)
import matplotlib.pyplot as plt

# ── Chemins dynamiques ──────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_PATH = os.path.join(BASE_DIR, "Image", "Harry_bebe.jpeg")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Lecture et conversion en niveaux de gris ─────────────────────────────
img = cv2.imread(IMAGE_PATH)

if img is None:
    raise FileNotFoundError(f"Image non trouvée : {IMAGE_PATH}")

gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ── 2. Seuillage global (manuel) ────────────────────────────────────────────
# retval, dst = cv2.threshold(src, thresh, maxval, type)
seuil = 127
_, thresh_binaire   = cv2.threshold(gris, seuil, 255, cv2.THRESH_BINARY)
_, thresh_inv       = cv2.threshold(gris, seuil, 255, cv2.THRESH_BINARY_INV)
_, thresh_trunc     = cv2.threshold(gris, seuil, 255, cv2.THRESH_TRUNC)
_, thresh_tozero    = cv2.threshold(gris, seuil, 255, cv2.THRESH_TOZERO)

# ── 3. Seuillage automatique (méthode d'Otsu) ───────────────────────────────
# Otsu calcule automatiquement le seuil optimal
seuil_otsu, thresh_otsu = cv2.threshold(
    gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
print(f"Seuil calculé par Otsu : {seuil_otsu:.1f}")

# ── 4. Seuillage adaptatif ───────────────────────────────────────────────────
# Utile pour les images avec des variations d'éclairage
thresh_adapt_mean = cv2.adaptiveThreshold(
    gris, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    blockSize=11,  # Taille du voisinage (doit être impair)
    C=2            # Constante soustraite à la moyenne
)

thresh_adapt_gauss = cv2.adaptiveThreshold(
    gris, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    blockSize=11,
    C=2
)

# ── 5. Affichage ─────────────────────────────────────────────────────────────
images = [gris, thresh_binaire, thresh_inv, thresh_otsu,
          thresh_adapt_mean, thresh_adapt_gauss]
titres = [
    "Niveaux de gris",
    f"Binaire (seuil={seuil})",
    "Binaire inversé",
    f"Otsu (seuil={seuil_otsu:.0f})",
    "Adaptatif Mean",
    "Adaptatif Gaussien"
]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, image, titre in zip(axes.ravel(), images, titres):
    ax.imshow(image, cmap="gray")
    ax.set_title(titre)
    ax.axis("off")

plt.suptitle("Comparaison des methodes de seuillage", fontsize=14)
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "6_5_seuillage.png")
plt.savefig(output_path, dpi=130)
plt.close()
print(f"Figure sauvegardee : {output_path}")
