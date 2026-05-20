# -*- coding: utf-8 -*-
"""
====================================================
 Section 7 - IoU (Intersection over Union)
====================================================
L'IoU mesure la qualite de localisation d'une boite predite
par rapport a la boite de reference (ground truth).

Formule :
    IoU = Aire(intersection) / Aire(union)

Valeur entre 0 (aucun overlap) et 1 (superposition parfaite).

Utilise les 2 vraies photos du chien :
  - Photo 1 : Harry_bebe.jpeg    -> boite de reference (GT)
  - Photo 2 : Harry_bebe (1).png -> boite predite

Base sur : https://github.com/yugmerabtene/nexa-computer-vision/blob/main/JOUR-01.md
"""

import os
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -- Chemins dynamiques -------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_PATH1 = os.path.join(BASE_DIR, "Image", "Harry_bebe.jpeg")
IMAGE_PATH2 = os.path.join(BASE_DIR, "Image", "Harry_bebe (1).png")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# UTILITAIRES
# =============================================================================
def imread_safe(path):
    """Lit une image meme si le chemin contient des espaces (Windows)."""
    arr = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def iou(box_a, box_b):
    """
    Calcule le score IoU entre deux boites.
    Format : (x1, y1, x2, y2)

    Etapes :
      1) Coordonnees de l'intersection
      2) Verification de la validite
      3) Aires et formule IoU
    """
    x_gauche = max(box_a[0], box_b[0])
    y_haut   = max(box_a[1], box_b[1])
    x_droite = min(box_a[2], box_b[2])
    y_bas    = min(box_a[3], box_b[3])

    if x_droite <= x_gauche or y_bas <= y_haut:
        return 0.0

    aire_inter = (x_droite - x_gauche) * (y_bas - y_haut)
    aire_a     = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    aire_b     = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    aire_union = aire_a + aire_b - aire_inter
    return aire_inter / aire_union


def grabcut_bbox(img, scale=0.25, iterations=3):
    """
    Extrait la boite englobante du sujet via GrabCut.
    Travaille sur une image reduite (scale) pour la vitesse.
    Retourne (x1, y1, x2, y2) a l'echelle originale.
    """
    h_orig, w_orig = img.shape[:2]
    img_small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h_s, w_s  = img_small.shape[:2]

    margin_x = max(1, int(w_s * 0.04))
    margin_y = max(1, int(h_s * 0.04))
    rect_gc  = (margin_x, margin_y, w_s - 2 * margin_x, h_s - 2 * margin_y)

    mask      = np.zeros((h_s, w_s), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(img_small, mask, rect_gc, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)

    mask_fg = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
    # Reprojeter a la taille originale
    mask_fg = cv2.resize(mask_fg, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(mask_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0, w_orig, h_orig)

    top5 = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    x1 = min(cv2.boundingRect(c)[0] for c in top5)
    y1 = min(cv2.boundingRect(c)[1] for c in top5)
    x2 = max(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in top5)
    y2 = max(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in top5)
    return (x1, y1, x2, y2)


# =============================================================================
# 1. CHARGEMENT DES IMAGES
# =============================================================================
print("=" * 55)
print("  Chargement des images")
print("=" * 55)

img1 = imread_safe(IMAGE_PATH1)
img2 = imread_safe(IMAGE_PATH2)

if img1 is None: raise FileNotFoundError(f"Image introuvable : {IMAGE_PATH1}")
if img2 is None: raise FileNotFoundError(f"Image introuvable : {IMAGE_PATH2}")

# Aligner les tailles
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)

print(f"  Photo 1 : {img1.shape}  -- {os.path.basename(IMAGE_PATH1)}")
print(f"  Photo 2 : {img2.shape}  -- {os.path.basename(IMAGE_PATH2)}")


# =============================================================================
# 2. EXTRACTION DES BOITES PAR GRABCUT
# =============================================================================
print("\n" + "=" * 55)
print("  Extraction des boites (GrabCut)")
print("=" * 55)

print("  Traitement photo 1...")
box_gt   = grabcut_bbox(img1)   # boite de reference (GT)
print(f"  Boite GT   (photo 1) : {box_gt}")

print("  Traitement photo 2...")
box_pred = grabcut_bbox(img2)   # boite predite
print(f"  Boite pred (photo 2) : {box_pred}")


# =============================================================================
# 3. CALCUL IoU
# =============================================================================
score = iou(box_pred, box_gt)

print("\n" + "=" * 55)
print("  Resultat IoU")
print("=" * 55)

aire_gt   = (box_gt[2]   - box_gt[0])   * (box_gt[3]   - box_gt[1])
aire_pred = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])
print(f"  Aire GT   : {aire_gt}  px2")
print(f"  Aire pred : {aire_pred}  px2")
print(f"  IoU       : {score:.3f}")

if   score >= 0.8: niveau = "Excellente localisation"
elif score >= 0.5: niveau = "Bonne localisation"
elif score  > 0.0: niveau = "Localisation partielle"
else:              niveau = "Aucun overlap"
print(f"  -> {niveau}")


# =============================================================================
# 4. VISUALISATION IoU (fonction generique)
# =============================================================================
def visualiser_iou(box_gt, box_pred, score, ax, titre=""):
    """Dessine les deux boites et leur intersection."""
    x_min_v = min(box_gt[0], box_pred[0]) - 15
    y_min_v = min(box_gt[1], box_pred[1]) - 15
    x_max_v = max(box_gt[2], box_pred[2]) + 15
    y_max_v = max(box_gt[3], box_pred[3]) + 15

    ax.set_facecolor("#f8f8f8")

    # Zone d'intersection
    xi1 = max(box_gt[0], box_pred[0])
    yi1 = max(box_gt[1], box_pred[1])
    xi2 = min(box_gt[2], box_pred[2])
    yi2 = min(box_gt[3], box_pred[3])
    if xi2 > xi1 and yi2 > yi1:
        ax.add_patch(patches.Rectangle((xi1, yi1), xi2-xi1, yi2-yi1,
                     linewidth=0, facecolor="#9b59b6", alpha=0.45, label="Intersection"))

    # Boite GT
    ax.add_patch(patches.Rectangle((box_gt[0], box_gt[1]),
                 box_gt[2]-box_gt[0], box_gt[3]-box_gt[1],
                 linewidth=2.5, edgecolor="#27ae60", facecolor="none", label="Photo 1 (GT)"))

    # Boite predite
    ax.add_patch(patches.Rectangle((box_pred[0], box_pred[1]),
                 box_pred[2]-box_pred[0], box_pred[3]-box_pred[1],
                 linewidth=2.5, edgecolor="#e67e22", facecolor="none", label="Photo 2 (pred)"))

    ax.set_xlim(x_min_v, x_max_v)
    ax.set_ylim(y_max_v, y_min_v)
    ax.set_title(f"{titre}\nIoU = {score:.3f}", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.grid(True, alpha=0.3)


# =============================================================================
# 5. COURBE SHIFT vs IoU (exemple pedagogique avec images synthetiques)
# =============================================================================
def make_rect(shift=0):
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (50+shift, 40), (250+shift, 160), (255, 255, 255), -1)
    return img

def bbox_synth(img_bgr):
    _, b = cv2.threshold(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
    pts = cv2.findNonZero(b)
    if pts is None: return None
    x, y, w, h = cv2.boundingRect(pts)
    return (x, y, x+w, y+h)

shifts_list = list(range(0, 65, 5))
box_ref     = bbox_synth(make_rect(0))
iou_shifts  = [iou(bbox_synth(make_rect(s)), box_ref) for s in shifts_list]


# =============================================================================
# 6. FIGURE COMPLETE
# =============================================================================
fig = plt.figure(figsize=(22, 14))
fig.suptitle("Section 7 - IoU : 2 vraies photos du meme chien", fontsize=16, fontweight="bold")

# -- Panneau 1 : Photo 1 avec boite GT ----------------------------------------
ax1 = fig.add_subplot(2, 4, 1)
img1_draw = img1.copy()
cv2.rectangle(img1_draw, (box_gt[0], box_gt[1]), (box_gt[2], box_gt[3]), (0, 255, 0), 4)
ax1.imshow(cv2.cvtColor(img1_draw, cv2.COLOR_BGR2RGB))
ax1.set_title(f"1. Photo 1 - Boite GT\n{os.path.basename(IMAGE_PATH1)}")
ax1.axis("off")

# -- Panneau 2 : Photo 2 avec boite pred --------------------------------------
ax2 = fig.add_subplot(2, 4, 2)
img2_draw = img2.copy()
cv2.rectangle(img2_draw, (box_pred[0], box_pred[1]), (box_pred[2], box_pred[3]), (255, 120, 0), 4)
ax2.imshow(cv2.cvtColor(img2_draw, cv2.COLOR_BGR2RGB))
ax2.set_title(f"2. Photo 2 - Boite pred\n{os.path.basename(IMAGE_PATH2)}")
ax2.axis("off")

# -- Panneau 3 : Visualisation IoU --------------------------------------------
ax3 = fig.add_subplot(2, 4, 3)
visualiser_iou(box_gt, box_pred, score, ax3, titre=f"3. Schema IoU\n(score = {score:.3f})")

# -- Panneau 4 : superposition des 2 boites -----------------------------------
ax4 = fig.add_subplot(2, 4, 4)
overlay = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
cv2.rectangle(overlay, (box_gt[0], box_gt[1]),   (box_gt[2],   box_gt[3]),   (0, 255, 0),   3)
cv2.rectangle(overlay, (box_pred[0], box_pred[1]), (box_pred[2], box_pred[3]), (255, 120, 0), 3)
ax4.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
ax4.set_title(f"4. Superposition\nVert=GT  Orange=Pred")
ax4.axis("off")

# -- Panneau 5-7 : courbe shift vs IoU (pedagogique) -------------------------
ax5 = fig.add_subplot(2, 4, (5, 7))
ax5.plot(shifts_list, iou_shifts, marker="o", linewidth=2.5, color="steelblue", label="IoU")
ax5.axhline(0.8, color="#27ae60", linestyle="--", linewidth=1.5, label="Seuil 0.8 (excellent)")
ax5.axhline(0.5, color="#e67e22", linestyle="--", linewidth=1.5, label="Seuil 0.5 (acceptable)")
ax5.fill_between(shifts_list, iou_shifts, alpha=0.15, color="steelblue")
ax5.set_title("5. Courbe : Decalage (shift) vs IoU\n(illustration pedagogique - rectangle synthetique)", fontsize=11)
ax5.set_xlabel("Decalage en pixels")
ax5.set_ylabel("Score IoU")
ax5.set_xlim([0, 60])
ax5.set_ylim([0, 1.05])
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# -- Panneau 8 : tableau recapitulatif ----------------------------------------
ax8 = fig.add_subplot(2, 4, 8)
ax8.axis("off")
headers = ["Photo", "Boite (x1,y1,x2,y2)", "Aire px2"]
rows = [
    ["GT (photo 1)",   str(box_gt),   str(aire_gt)],
    ["Pred (photo 2)", str(box_pred), str(aire_pred)],
    ["IoU",            f"{score:.3f}", niveau],
]
table = ax8.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.1, 2.0)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    elif row == 3:
        cell.set_facecolor("#d5f5e3" if score >= 0.5 else "#fadbd8")

ax8.set_title("Resume", fontweight="bold")

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "7_1_iou.png")
plt.savefig(fig_path, dpi=130)
plt.close()
print(f"\nFigure sauvegardee : {fig_path}")


# =============================================================================
# 7. EXPORT JSON
# =============================================================================
metriques = {
    "images": {
        "photo1": os.path.basename(IMAGE_PATH1),
        "photo2": os.path.basename(IMAGE_PATH2),
    },
    "iou_score":       round(score, 3),
    "bbox_gt":         list(box_gt),
    "bbox_pred":       list(box_pred),
    "aire_gt":         aire_gt,
    "aire_pred":       aire_pred,
    "interpretation":  niveau,
    "courbe_shift_iou": {str(s): round(iou(bbox_synth(make_rect(s)), box_ref), 3) for s in shifts_list},
    "figure_path":     fig_path,
}
json_path = os.path.join(OUTPUT_DIR, "7_1_iou_metrics.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(metriques, f, indent=2, ensure_ascii=False)

print(f"Metriques JSON  : {json_path}")
print(f"\n--- Resume ---")
print(f"IoU photos reelles : {score:.3f}  -> {niveau}")

