# -*- coding: utf-8 -*-
"""
====================================================
 Lab Jour 1 - Pipeline complet
====================================================
Script principal du lab : IoU + HOG + SIFT + metriques JSON

Utilise 2 vraies photos du meme chien pour les tests SIFT/HOG :
  - Image 1 : Harry_bebe.jpeg    (photo principale)
  - Image 2 : Harry_bebe (1).png (photo similaire - meme chien)

Checkpoints :
  A) iou_score entre 0 et 1  (~0.84 pour shift=12)
  B) hog_different_l2 > hog_similar_l2
  C) sift_good_matches_similar >= sift_good_matches_different

Base sur : https://github.com/yugmerabtene/nexa-computer-vision/blob/main/JOUR-01.md
"""

import os
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- Chemins dynamiques -------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_PATH1 = os.path.join(BASE_DIR, "Image", "Harry_bebe.jpeg")
IMAGE_PATH2 = os.path.join(BASE_DIR, "Image", "Harry_bebe (1).png")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- Chargement des 2 images reelles ------------------------------------------
# cv2.imread ne gere pas les espaces dans les chemins sur Windows
# --> on utilise np.fromfile + cv2.imdecode
def imread_safe(path):
    arr = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

img1 = imread_safe(IMAGE_PATH1)
img2 = imread_safe(IMAGE_PATH2)

if img1 is None:
    raise FileNotFoundError(f"Image non trouvee : {IMAGE_PATH1}")
if img2 is None:
    raise FileNotFoundError(f"Image non trouvee : {IMAGE_PATH2}")

print(f"Image 1 : {img1.shape}  -- {os.path.basename(IMAGE_PATH1)}")
print(f"Image 2 : {img2.shape}  -- {os.path.basename(IMAGE_PATH2)}")

# Redimensionner img2 a la meme taille qu'img1 pour HOG
img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)

# =============================================================================
# UTILITAIRES
# =============================================================================
def make_scene(forme="rectangle", shift=0):
    """Cree une image synthetique 200x300 avec une forme geometrique."""
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    if forme == "rectangle":
        cv2.rectangle(img, (50 + shift, 40), (250 + shift, 160), (255, 255, 255), -1)
    elif forme == "cercle":
        cv2.circle(img, (150, 100), 70, (255, 255, 255), -1)
    return img


def bbox_from_threshold(gris, seuil=127):
    """Retourne (x1, y1, x2, y2) via seuillage + boundingRect."""
    _, binaire = cv2.threshold(gris, seuil, 255, cv2.THRESH_BINARY)
    points = cv2.findNonZero(binaire)
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    return (x, y, x + w, y + h)


def iou(box_a, box_b):
    """IoU entre deux boites (x1, y1, x2, y2)."""
    x_left   = max(box_a[0], box_b[0])
    y_top    = max(box_a[1], box_b[1])
    x_right  = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    inter  = (x_right - x_left) * (y_bottom - y_top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


# =============================================================================
# ETAPE 1 -- IoU (sur images synthetiques - reference du cours)
# =============================================================================
print("\n" + "=" * 55)
print("  ETAPE 1 -- IoU")
print("=" * 55)

SHIFT = 12
img_gt_synth   = make_scene("rectangle", shift=0)
img_pred_synth = make_scene("rectangle", shift=SHIFT)

box_gt   = bbox_from_threshold(cv2.cvtColor(img_gt_synth,   cv2.COLOR_BGR2GRAY))
box_pred = bbox_from_threshold(cv2.cvtColor(img_pred_synth, cv2.COLOR_BGR2GRAY))

iou_score = iou(box_pred, box_gt)
print(f"  Boite GT   : {box_gt}")
print(f"  Boite pred : {box_pred}")
print(f"  IoU        : {iou_score:.3f}")
assert 0 < iou_score <= 1, "Checkpoint A FAILED"
print("  [CHECKPOINT A] OK")


# =============================================================================
# ETAPE 2 -- HOG sur les 2 vraies photos
# =============================================================================
print("\n" + "=" * 55)
print("  ETAPE 2 -- HOG (images reelles)")
print("=" * 55)

hog_desc = cv2.HOGDescriptor(
    _winSize=(128, 64), _blockSize=(16, 16),
    _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9,
)

def compute_hog(img_bgr):
    g = cv2.cvtColor(
        cv2.resize(img_bgr, (128, 64), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2GRAY
    )
    return hog_desc.compute(g).flatten()


# Image differente (forme geometrique) pour le controle
img_cercle_synth = make_scene("cercle")

desc_img1       = compute_hog(img1)
desc_img2       = compute_hog(img2_resized)   # photo similaire (meme chien)
desc_different  = compute_hog(img_cercle_synth)

hog_similar_l2   = float(np.linalg.norm(desc_img1 - desc_img2))
hog_different_l2 = float(np.linalg.norm(desc_img1 - desc_different))

print(f"  Dimension HOG              : {desc_img1.shape[0]}")
print(f"  Dist HOG (img1 <-> img2)   : {hog_similar_l2:.2f}  (meme chien)")
print(f"  Dist HOG (img1 <-> cercle) : {hog_different_l2:.2f}  (forme differente)")
assert hog_different_l2 > hog_similar_l2, "Checkpoint B FAILED"
print("  [CHECKPOINT B] OK -- different > similaire")


# =============================================================================
# ETAPE 3 -- SIFT sur les 2 vraies photos
# =============================================================================
print("\n" + "=" * 55)
print("  ETAPE 3 -- SIFT (images reelles)")
print("=" * 55)

sift = cv2.SIFT_create()

def detect_sift(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return sift.detectAndCompute(g, None)

kp1,  desc_s1       = detect_sift(img1)
kp2,  desc_s2       = detect_sift(img2_resized)
kp_c, desc_s_cercle = detect_sift(img_cercle_synth)

def lowe_matches(d1, d2, ratio=0.75):
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return []
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(d1, d2, k=2)
    return [m for m, n in matches if m.distance < ratio * n.distance]

bons_similaires = lowe_matches(desc_s1, desc_s2)
bons_differents = lowe_matches(desc_s1, desc_s_cercle)

print(f"  Points cles img1 (photo 1) : {len(kp1)}")
print(f"  Points cles img2 (photo 2) : {len(kp2)}")
print(f"  Points cles cercle         : {len(kp_c)}")
print(f"  Bons matches (similaire)   : {len(bons_similaires)}")
print(f"  Bons matches (different)   : {len(bons_differents)}")
assert len(bons_similaires) >= len(bons_differents), "Checkpoint C FAILED"
print("  [CHECKPOINT C] OK -- similaire >= different")


# =============================================================================
# FIGURE DE SYNTHESE
# =============================================================================
# Boites IoU
img_iou_viz = img_gt_synth.copy()
if box_gt:   cv2.rectangle(img_iou_viz, box_gt[:2],   box_gt[2:],   (0, 255, 0), 2)
if box_pred: cv2.rectangle(img_iou_viz, box_pred[:2], box_pred[2:], (0, 0, 255), 2)

# Courbe shift vs IoU
shifts     = list(range(0, 65, 5))
iou_scores = [
    iou(bbox_from_threshold(cv2.cvtColor(make_scene("rectangle", s), cv2.COLOR_BGR2GRAY)), box_gt)
    for s in shifts
]

# SIFT matching sur vraies photos (top 15 matches)
gray1 = cv2.cvtColor(img1,         cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
img_sift_draw = cv2.drawMatches(
    gray1, kp1, gray2, kp2,
    bons_similaires[:15], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

fig = plt.figure(figsize=(22, 13))
fig.suptitle(
    "Lab Jour 1 - Pipeline complet : IoU + HOG + SIFT\n(2 vraies photos du meme chien)",
    fontsize=14, fontweight="bold"
)

# -- Ligne 1 ------------------------------------------------------------------
ax1 = fig.add_subplot(2, 4, 1)
ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
ax1.set_title(f"1. Photo 1\n{os.path.basename(IMAGE_PATH1)}")
ax1.axis("off")

ax2 = fig.add_subplot(2, 4, 2)
ax2.imshow(cv2.cvtColor(img2_resized, cv2.COLOR_BGR2RGB))
ax2.set_title(f"2. Photo 2\n{os.path.basename(IMAGE_PATH2)}")
ax2.axis("off")

ax3 = fig.add_subplot(2, 4, 3)
ax3.imshow(cv2.cvtColor(img_iou_viz, cv2.COLOR_BGR2RGB))
ax3.set_title(f"3. IoU = {iou_score:.3f}\nVert=GT  Bleu=Pred (shift={SHIFT}px)")
ax3.axis("off")

ax4 = fig.add_subplot(2, 4, 4)
ax4.plot(shifts, iou_scores, marker="o", color="steelblue", linewidth=2)
ax4.axhline(0.8, color="green",  linestyle="--", label="Seuil 0.8")
ax4.axhline(0.5, color="orange", linestyle="--", label="Seuil 0.5")
ax4.set_title("4. Courbe Shift vs IoU")
ax4.set_xlabel("Shift (px)")
ax4.set_ylabel("IoU")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# -- Ligne 2 ------------------------------------------------------------------
ax5 = fig.add_subplot(2, 4, 5)
cats = ["Photo1 <-> Photo2\n(similaire)", "Photo1 <-> Cercle\n(different)"]
dsts = [hog_similar_l2, hog_different_l2]
cols = ["#27ae60", "#e74c3c"]
bars = ax5.bar(cats, dsts, color=cols, edgecolor="black", width=0.5)
for bar, val in zip(bars, dsts):
    ax5.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
             f"{val:.1f}", ha="center", fontweight="bold")
ax5.set_title("5. Distances HOG L2\n[CHECKPOINT B] OK")
ax5.set_ylabel("Distance L2")
ax5.grid(True, alpha=0.3, axis="y")

ax6 = fig.add_subplot(2, 4, 6)
ax6.imshow(cv2.cvtColor(img_sift_draw, cv2.COLOR_BGR2RGB))
ax6.set_title(f"6. SIFT Matching (photos reelles)\n{len(bons_similaires)} bons matches [CHECKPOINT C]")
ax6.axis("off")

ax7 = fig.add_subplot(2, 4, 7)
noms    = ["KP\nPhoto1", "KP\nPhoto2", "KP\nCercle", "Matches\nSimilaire", "Matches\nDifferent"]
valeurs = [len(kp1), len(kp2), len(kp_c), len(bons_similaires), len(bons_differents)]
cols7   = ["#3498db", "#2ecc71", "#e74c3c", "#27ae60", "#e67e22"]
bars7   = ax7.bar(noms, valeurs, color=cols7, edgecolor="black")
for bar, val in zip(bars7, valeurs):
    ax7.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
             str(val), ha="center", fontweight="bold")
ax7.set_title("7. SIFT - Points cles & Matches")
ax7.set_ylabel("Nombre")
ax7.grid(True, alpha=0.3, axis="y")

ax8 = fig.add_subplot(2, 4, 8)
ax8.axis("off")
checks = [
    ("A", "IoU valide",         f"{iou_score:.3f} dans ]0,1]",                    True),
    ("B", "HOG : diff > sim",   f"{hog_different_l2:.1f} > {hog_similar_l2:.1f}", hog_different_l2 > hog_similar_l2),
    ("C", "SIFT : sim >= diff", f"{len(bons_similaires)} >= {len(bons_differents)}", len(bons_similaires) >= len(bons_differents)),
]
y = 0.85
for lettre, label, valeur, ok in checks:
    couleur = "#27ae60" if ok else "#e74c3c"
    statut  = "  OK  " if ok else "FAILED"
    ax8.text(0.02, y,      f"[{statut}]",            transform=ax8.transAxes, fontsize=11, color=couleur, fontweight="bold")
    ax8.text(0.30, y,      f"CP {lettre} - {label}", transform=ax8.transAxes, fontsize=9)
    ax8.text(0.30, y-0.06, f"  -> {valeur}",         transform=ax8.transAxes, fontsize=8, color="gray")
    y -= 0.22
ax8.set_title("8. Checkpoints", fontweight="bold")

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "lab_jour1.png")
plt.savefig(fig_path, dpi=130)
plt.close()
print(f"\nFigure sauvegardee : {fig_path}")


# -- Export JSON --------------------------------------------------------------
metriques = {
    "images": {
        "photo1": os.path.basename(IMAGE_PATH1),
        "photo2": os.path.basename(IMAGE_PATH2),
    },
    "iou_score":                   round(iou_score, 3),
    "bbox_gt":                     list(box_gt),
    "bbox_pred":                   list(box_pred),
    "hog_dimension":               int(desc_img1.shape[0]),
    "hog_similar_l2":              round(hog_similar_l2, 2),
    "hog_different_l2":            round(hog_different_l2, 2),
    "sift_kp_photo1":              len(kp1),
    "sift_kp_photo2":              len(kp2),
    "sift_kp_cercle":              len(kp_c),
    "sift_good_matches_similar":   len(bons_similaires),
    "sift_good_matches_different": len(bons_differents),
    "checkpoints": {
        "A_iou_valide":      bool(0 < iou_score <= 1),
        "B_hog_separation":  bool(hog_different_l2 > hog_similar_l2),
        "C_sift_separation": bool(len(bons_similaires) >= len(bons_differents)),
    },
    "figure_path": fig_path,
}

json_path = os.path.join(OUTPUT_DIR, "lab_jour1_metrics.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(metriques, f, indent=2, ensure_ascii=False)

print(f"Metriques JSON  : {json_path}")
print(f"\n{'='*55}")
all_ok = all(metriques["checkpoints"].values())
for k, v in metriques["checkpoints"].items():
    print(f"  {'OK  ' if v else 'FAIL'} {k}")
print(f"\n  {'LAB VALIDE' if all_ok else 'CERTAINS CHECKPOINTS ECHOUES'}")

