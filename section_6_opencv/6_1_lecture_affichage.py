"""
====================================================
 Section 6.1 — Lecture et affichage d'une image
====================================================
OpenCV lit les images sous forme de tableaux NumPy.
L'ordre des canaux est BGR (et non RGB).
"""

import cv2
import matplotlib.pyplot as plt

# ── 1. Lecture de l'image ───────────────────────────────────────────────────
img = cv2.imread("../Image/Harry_bebe.jpeg")

# Vérification : cv2.imread retourne None si le fichier est introuvable
if img is None:
    raise FileNotFoundError("Image non trouvée. Vérifiez le chemin.")

# ── 2. Informations de base ─────────────────────────────────────────────────
print("Type         :", type(img))        # numpy.ndarray
print("Forme        :", img.shape)        # (hauteur, largeur, canaux)
print("Type données :", img.dtype)        # uint8
print("Valeur min   :", img.min())
print("Valeur max   :", img.max())

# ── 3. Affichage avec Matplotlib (conversion BGR → RGB obligatoire) ──────────
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6, 5))
plt.imshow(img_rgb)
plt.title("Image originale (RGB)")
plt.axis("off")
plt.tight_layout()
plt.show()

# ── 4. Affichage avec OpenCV ─────────────────────────────────────────────────
cv2.imshow("Image originale (BGR)", img)
cv2.waitKey(0)   # Attendre une touche
cv2.destroyAllWindows()

