"""
====================================================
 Section 6.7 — Exemple complet OpenCV (pipeline)
====================================================
Script : openCV_bases.py
Démontre la chaîne complète :
  lecture → conversion → resize → histogramme → seuillage → boîte détectée

Basé sur l'exemple du cours (image synthétique : rectangle blanc / fond noir).
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")   # Mode sans affichage (serveur / SSH)
import matplotlib.pyplot as plt

# ── 1. Création d'une image synthétique ─────────────────────────────────────
# Rectangle blanc sur fond noir (scène simple reproductible)
img = np.zeros((200, 300, 3), dtype=np.uint8)
cv2.rectangle(img, (50, 40), (250, 160), (255, 255, 255), -1)

# ── 2. Conversion en niveaux de gris ────────────────────────────────────────
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ── 3. Redimensionnement (INTER_AREA pour réduire) ──────────────────────────
petite = cv2.resize(gris, (64, 32), interpolation=cv2.INTER_AREA)

# ── 4. Histogramme ───────────────────────────────────────────────────────────
hist = cv2.calcHist([gris], [0], None, [256], [0, 256])
print(f"Pixels noirs  (valeur   0) : {int(hist[0][0])}")
print(f"Pixels blancs (valeur 255) : {int(hist[255][0])}")

# ── 5. Seuillage binaire → extraction de la boîte ───────────────────────────
_, binaire = cv2.threshold(gris, 127, 255, cv2.THRESH_BINARY)
points = cv2.findNonZero(binaire)
x, y, w, h = cv2.boundingRect(points)
print(f"Boîte détectée : x={x}, y={y}, w={w}, h={h}")

# Dessiner la boîte sur l'image
img_resultat = img.copy()
cv2.rectangle(img_resultat, (x, y), (x + w, y + h), (0, 255, 0), 2)

# ── 6. Affichage récapitulatif ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("1. Image synthétique")
axes[0].axis("off")

axes[1].imshow(gris, cmap="gray")
axes[1].set_title("2. Niveaux de gris")
axes[1].axis("off")

axes[2].plot(hist, color="black")
axes[2].fill_between(np.arange(256), hist.flatten(), alpha=0.3, color="gray")
axes[2].set_title("4. Histogramme")
axes[2].set_xlabel("Intensité")
axes[2].set_xlim([0, 256])

axes[3].imshow(cv2.cvtColor(img_resultat, cv2.COLOR_BGR2RGB))
axes[3].set_title(f"5. Boîte détectée\nx={x}, y={y}, w={w}, h={h}")
axes[3].axis("off")

plt.suptitle("Section 6.7 — Pipeline complet OpenCV", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("../outputs/6_7_pipeline_complet.png", dpi=130)
plt.close()
print("Figure sauvegardée : outputs/6_7_pipeline_complet.png")

