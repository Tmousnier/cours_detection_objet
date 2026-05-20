"""
====================================================
 Section 6 — Script complet (pipeline complet)
====================================================
Ce script exécute toutes les étapes de la section 6
dans l'ordre : lecture → gris → resize → histogramme → seuillage

La figure est sauvegardée dans outputs/main_section6.png
(matplotlib.use("Agg") évite les problèmes d'affichage en SSH / serveur)
"""

import os
import cv2
import matplotlib
matplotlib.use("Agg")   # Pas d'affichage graphique → sauvegarde fichier
import matplotlib.pyplot as plt
import numpy as np

# ── Chemins dynamiques (fonctionne peu importe le répertoire de lancement) ──
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_PATH  = os.path.join(BASE_DIR, "Image", "Harry_bebe.jpeg")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "main_section6.png")

# Créer le dossier outputs/ s'il n'existe pas
os.makedirs(OUTPUT_DIR, exist_ok=True)


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

# ── Ajout : contours + boîtes ────────────────────────────────────────────────
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_boites = img.copy()
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if w * h > 500:   # Filtrer les très petits contours
        cv2.rectangle(img_boites, (x, y), (x + w, y + h), (0, 255, 0), 2)

ax6 = fig.add_subplot(2, 3, 6)
ax6.imshow(cv2.cvtColor(img_boites, cv2.COLOR_BGR2RGB))
ax6.set_title(f"6. Contours ({len(contours)} détectés)")
ax6.axis("off")

plt.tight_layout()

# ── Sauvegarde ───────────────────────────────────────────────────────────────
output_path = OUTPUT_PATH
plt.savefig(output_path, dpi=130)
plt.close()
print(f"[6] Contours détectés : {len(contours)}")
print(f"\n✅ Figure sauvegardée : {output_path}")

