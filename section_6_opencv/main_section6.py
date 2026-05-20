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
# 4. Histogramme niveaux de gris + courbes R, G, B
# ───────────────────────────────────────────────────────────────────────────
hist = cv2.calcHist([gris], [0], None, [256], [0, 256])

# OpenCV stocke en BGR → index 2=R, 1=G, 0=B
canaux_rgb = [
    {"nom": "Rouge (R)", "index": 2, "color": "red",   "fill": "#ffaaaa"},
    {"nom": "Vert  (G)", "index": 1, "color": "green", "fill": "#aaffaa"},
    {"nom": "Bleu  (B)", "index": 0, "color": "blue",  "fill": "#aaaaff"},
]

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
x = np.arange(256)

# Courbe niveaux de gris (fond gris)
ax4.plot(x, hist.flatten(), color="black", linewidth=1.2, alpha=0.5, label="Gris")
ax4.fill_between(x, hist.flatten(), alpha=0.15, color="gray")

# Courbes R, G, B
for c in canaux_rgb:
    h = cv2.calcHist([img], [c["index"]], None, [256], [0, 256]).flatten()
    ax4.plot(x, h, color=c["color"], linewidth=1.8, label=c["nom"], alpha=0.9)
    ax4.fill_between(x, h, color=c["fill"], alpha=0.20)

# Ligne verticale Otsu
ax4.axvline(seuil_otsu, color="red", linestyle="--", linewidth=1.4, label=f"Otsu={seuil_otsu:.0f}")

ax4.set_title("4. Histogramme R – G – B")
ax4.set_xlabel("Intensité (0–255)")
ax4.set_ylabel("Pixels")
ax4.set_xlim([0, 255])
ax4.legend(fontsize=7)
ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(2, 3, 5)
ax5.imshow(thresh, cmap="gray")
ax5.set_title(f"5. Seuillage Otsu ({seuil_otsu:.0f})")
ax5.axis("off")

# ── Contours + boîte englobante autour de l'animal (GrabCut) ────────────────
# GrabCut est conçu pour extraire le sujet du fond.
# On lui donne un rectangle légèrement en retrait des bords → il segmente automatiquement

h_img, w_img = img.shape[:2]
margin_x = int(w_img * 0.04)
margin_y = int(h_img * 0.04)

# Rectangle initial = zone probable de l'avant-plan (x, y, largeur, hauteur)
rect_gc = (margin_x, margin_y, w_img - 2 * margin_x, h_img - 2 * margin_y)

mask_gc   = np.zeros((h_img, w_img), np.uint8)
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

# 5 itérations GrabCut
cv2.grabCut(img, mask_gc, rect_gc, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# Masque binaire : 0/2 = fond, 1/3 = avant-plan
masque_fg = np.where((mask_gc == 2) | (mask_gc == 0), 0, 255).astype(np.uint8)

# Trouver les contours sur le masque GrabCut
contours_gc, _ = cv2.findContours(masque_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_significatifs = sorted(contours_gc, key=cv2.contourArea, reverse=True)[:5]

img_boites = img.copy()
if contours_significatifs:
    # Boîte englobante globale sur les 5 plus grandes régions
    x_min = min(cv2.boundingRect(c)[0] for c in contours_significatifs)
    y_min = min(cv2.boundingRect(c)[1] for c in contours_significatifs)
    x_max = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in contours_significatifs)
    y_max = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in contours_significatifs)

    cv2.rectangle(img_boites, (x_min, y_min), (x_max, y_max), (0, 255, 0), 4)
    print(f"[6] Boîte englobante (GrabCut) : x={x_min}, y={y_min}, w={x_max-x_min}, h={y_max-y_min}")

ax6 = fig.add_subplot(2, 3, 6)
ax6.imshow(cv2.cvtColor(img_boites, cv2.COLOR_BGR2RGB))
ax6.set_title(f"6. Boîte englobante\n({len(contours_significatifs)} régions fusionnées)")
ax6.axis("off")

plt.tight_layout()

# ── Sauvegarde ───────────────────────────────────────────────────────────────
plt.savefig(OUTPUT_PATH, dpi=130)
plt.close()
print(f"\n✅ Figure sauvegardée : {OUTPUT_PATH}")

