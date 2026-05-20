"""
====================================================
 Lab Jour 1 — Pipeline complet
====================================================
Script principal du lab : IoU + HOG + SIFT + métriques JSON

Checkpoints :
  A) iou_score entre 0 et 1  (~0.84 pour shift=12)
  B) hog_different_l2 > hog_shifted_l2
  C) sift_good_matches_similar > sift_good_matches_different

Basé sur : https://github.com/yugmerabtene/nexa-computer-vision/blob/main/JOUR-01.md
"""

import os
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Chemins dynamiques ──────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════
def make_scene(forme="rectangle", shift=0):
    """Crée une image synthétique 200×300 avec une forme géométrique."""
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    if forme == "rectangle":
        cv2.rectangle(img, (50 + shift, 40), (250 + shift, 160), (255, 255, 255), -1)
    elif forme == "cercle":
        cv2.circle(img, (150, 100), 70, (255, 255, 255), -1)
    elif forme == "triangle":
        pts = np.array([[150, 30], [50, 170], [250, 170]], np.int32)
        cv2.fillPoly(img, [pts], (255, 255, 255))
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
    """IoU entre deux boîtes (x1, y1, x2, y2)."""
    x_left   = max(box_a[0], box_b[0])
    y_top    = max(box_a[1], box_b[1])
    x_right  = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    inter    = (x_right - x_left) * (y_bottom - y_top)
    area_a   = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b   = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


# ═══════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — IoU
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 55)
print("  ÉTAPE 1 — IoU")
print("=" * 55)

SHIFT = 12
img_gt   = make_scene("rectangle", shift=0)
img_pred = make_scene("rectangle", shift=SHIFT)

gray_gt   = cv2.cvtColor(img_gt,   cv2.COLOR_BGR2GRAY)
gray_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2GRAY)

box_gt   = bbox_from_threshold(gray_gt)
box_pred = bbox_from_threshold(gray_pred)

iou_score = iou(box_pred, box_gt)
print(f"  Boîte GT   : {box_gt}")
print(f"  Boîte pred : {box_pred}")
print(f"  IoU        : {iou_score:.3f}")
assert 0 < iou_score <= 1, "Checkpoint A FAILED"
print("  [CHECKPOINT A] OK")


# ═══════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — HOG
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("  ÉTAPE 2 — HOG")
print("=" * 55)

hog_desc = cv2.HOGDescriptor(
    _winSize   =(128, 64),
    _blockSize =(16, 16),
    _blockStride=(8,  8),
    _cellSize  =(8,   8),
    _nbins     =9,
)

def compute_hog(img_bgr):
    g = cv2.cvtColor(cv2.resize(img_bgr, (128, 64), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY)
    return hog_desc.compute(g).flatten()

img_cercle   = make_scene("cercle")
img_shifted  = make_scene("rectangle", shift=SHIFT)

desc_gt       = compute_hog(img_gt)
desc_shifted  = compute_hog(img_shifted)
desc_different = compute_hog(img_cercle)

hog_shifted_l2   = float(np.linalg.norm(desc_gt - desc_shifted))
hog_different_l2 = float(np.linalg.norm(desc_gt - desc_different))

print(f"  Dimension HOG    : {desc_gt.shape[0]}")
print(f"  Dist (similaire) : {hog_shifted_l2:.2f}")
print(f"  Dist (différent) : {hog_different_l2:.2f}")
assert hog_different_l2 > hog_shifted_l2, "Checkpoint B FAILED"
print("  [CHECKPOINT B] OK — différent > similaire")


# ═══════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — SIFT
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("  ÉTAPE 3 — SIFT")
print("=" * 55)

sift = cv2.SIFT_create()

def detect_sift(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return sift.detectAndCompute(g, None)

kp_gt,   desc_sift_gt   = detect_sift(img_gt)
kp_pred, desc_sift_pred = detect_sift(img_shifted)
kp_other, desc_sift_other = detect_sift(img_cercle)

# Matching avec test de ratio de Lowe (threshold=0.75)
def lowe_matches(d1, d2, ratio=0.75):
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return []
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(d1, d2, k=2)
    return [m for m, n in matches if m.distance < ratio * n.distance]

bons_similaires  = lowe_matches(desc_sift_gt, desc_sift_pred)
bons_differents  = lowe_matches(desc_sift_gt, desc_sift_other)

print(f"  Points clés GT         : {len(kp_gt)}")
print(f"  Points clés similaire  : {len(kp_pred)}")
print(f"  Points clés différent  : {len(kp_other)}")
print(f"  Bons matches (similaire)  : {len(bons_similaires)}")
print(f"  Bons matches (différent)  : {len(bons_differents)}")
assert len(bons_similaires) >= len(bons_differents), "Checkpoint C FAILED"
print("  [CHECKPOINT C] OK — similaire >= différent")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE DE SYNTHÈSE
# ═══════════════════════════════════════════════════════════════════════════
# Dessiner les boîtes IoU
img_iou_viz = img_gt.copy()
if box_gt:
    cv2.rectangle(img_iou_viz, (box_gt[0], box_gt[1]),   (box_gt[2], box_gt[3]),   (0, 255, 0), 2)
if box_pred:
    cv2.rectangle(img_iou_viz, (box_pred[0], box_pred[1]), (box_pred[2], box_pred[3]), (0, 0, 255), 2)

# Courbe shift vs IoU
shifts     = list(range(0, 65, 5))
iou_scores = [iou(bbox_from_threshold(cv2.cvtColor(make_scene("rectangle", s), cv2.COLOR_BGR2GRAY)), box_gt)
              for s in shifts]

# SIFT matching image
img_sift_draw = cv2.drawMatches(
    cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY), kp_gt,
    cv2.cvtColor(img_shifted, cv2.COLOR_BGR2GRAY), kp_pred,
    bons_similaires[:10], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

fig = plt.figure(figsize=(20, 12))
fig.suptitle("Lab Jour 1 — Pipeline complet : IoU + HOG + SIFT",
             fontsize=15, fontweight="bold")

# Ligne 1
ax1 = fig.add_subplot(2, 4, 1)
ax1.imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))
ax1.set_title(f"1. Image GT\nBoîte : {box_gt}")
ax1.axis("off")

ax2 = fig.add_subplot(2, 4, 2)
ax2.imshow(cv2.cvtColor(img_iou_viz, cv2.COLOR_BGR2RGB))
ax2.set_title(f"2. IoU = {iou_score:.3f}\nVert=GT  Bleu=Pred")
ax2.axis("off")

ax3 = fig.add_subplot(2, 4, 3)
ax3.plot(shifts, iou_scores, marker="o", color="steelblue", linewidth=2)
ax3.axhline(0.8, color="green",  linestyle="--", label="0.8")
ax3.axhline(0.5, color="orange", linestyle="--", label="0.5")
ax3.set_title("3. Courbe Shift vs IoU")
ax3.set_xlabel("Shift (px)")
ax3.set_ylabel("IoU")
ax3.set_xlim([0, 60])
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(2, 4, 4)
ax4.plot(desc_gt[:80],        color="blue",   label="GT",       linewidth=1.2, alpha=0.8)
ax4.plot(desc_shifted[:80],   color="green",  label="Décalé",   linewidth=1.2, alpha=0.8)
ax4.plot(desc_different[:80], color="red",    label="Différent",linewidth=1.2, alpha=0.8)
ax4.set_title(f"4. HOG — 80 premières dims\n({desc_gt.shape[0]} total)")
ax4.set_xlabel("Dimension")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# Ligne 2
ax5 = fig.add_subplot(2, 4, 5)
categories = ["GT ↔ Décalé", "GT ↔ Cercle"]
dists = [hog_shifted_l2, hog_different_l2]
colors_bar = ["#27ae60", "#e74c3c"]
bars = ax5.bar(categories, dists, color=colors_bar, edgecolor="black", width=0.5)
for bar, val in zip(bars, dists):
    ax5.text(bar.get_x() + bar.get_width()/2, val + 0.05,
             f"{val:.2f}", ha="center", fontweight="bold")
ax5.set_title("5. Distances HOG L2\n(Checkpoint B)")
ax5.set_ylabel("Distance L2")
ax5.grid(True, alpha=0.3, axis="y")

ax6 = fig.add_subplot(2, 4, 6)
ax6.imshow(cv2.cvtColor(img_sift_draw, cv2.COLOR_BGR2RGB))
ax6.set_title(f"6. SIFT Matching\n{len(bons_similaires)} bons matches (Checkpoint C)")
ax6.axis("off")

ax7 = fig.add_subplot(2, 4, 7)
noms   = ["KP GT", "KP Décalé", "KP Cercle", "Matches\nSimilaire", "Matches\nDifférent"]
valeurs = [len(kp_gt), len(kp_pred), len(kp_other), len(bons_similaires), len(bons_differents)]
cols   = ["#3498db","#2ecc71","#e74c3c","#27ae60","#e67e22"]
bars2  = ax7.bar(noms, valeurs, color=cols, edgecolor="black")
for bar, val in zip(bars2, valeurs):
    ax7.text(bar.get_x() + bar.get_width()/2, val + 0.1,
             str(val), ha="center", fontweight="bold")
ax7.set_title("7. SIFT — Points clés & Matches")
ax7.set_ylabel("Nombre")
ax7.grid(True, alpha=0.3, axis="y")

# Panneau 8 : checkpoints
ax8 = fig.add_subplot(2, 4, 8)
ax8.axis("off")
checks = [
    ("A", "IoU valide",              f"{iou_score:.3f} ∈ ]0,1]", True),
    ("B", "HOG : diff > similaire",  f"{hog_different_l2:.2f} > {hog_shifted_l2:.2f}", hog_different_l2 > hog_shifted_l2),
    ("C", "SIFT : sim >= diff",      f"{len(bons_similaires)} >= {len(bons_differents)}", len(bons_similaires) >= len(bons_differents)),
]
y = 0.85
for lettre, label, valeur, ok in checks:
    couleur = "#27ae60" if ok else "#e74c3c"
    statut  = "OK" if ok else "FAILED"
    ax8.text(0.05, y, f"[{statut}]", transform=ax8.transAxes,
             fontsize=11, color=couleur, fontweight="bold")
    ax8.text(0.25, y, f"Checkpoint {lettre} : {label}", transform=ax8.transAxes, fontsize=9)
    ax8.text(0.25, y - 0.06, f"        → {valeur}", transform=ax8.transAxes,
             fontsize=8, color="gray")
    y -= 0.20
ax8.set_title("8. Checkpoints du lab", fontweight="bold")

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "lab_jour1.png")
plt.savefig(fig_path, dpi=130)
plt.close()
print(f"\nFigure sauvegardee : {fig_path}")


# ═══════════════════════════════════════════════════════════════════════════
# EXPORT JSON
# ═══════════════════════════════════════════════════════════════════════════
metriques = {
    "iou_score":                round(iou_score, 3),
    "bbox_gt":                  list(box_gt),
    "bbox_pred":                list(box_pred),
    "hog_dimension":            int(desc_gt.shape[0]),
    "hog_shifted_l2":           round(hog_shifted_l2, 2),
    "hog_different_l2":         round(hog_different_l2, 2),
    "sift_kp_gt":               len(kp_gt),
    "sift_kp_pred":             len(kp_pred),
    "sift_kp_other":            len(kp_other),
    "sift_good_matches_similar":  len(bons_similaires),
    "sift_good_matches_different": len(bons_differents),
    "checkpoints": {
        "A_iou_valide":           bool(0 < iou_score <= 1),
        "B_hog_separation":       bool(hog_different_l2 > hog_shifted_l2),
        "C_sift_separation":      bool(len(bons_similaires) >= len(bons_differents)),
    },
    "figure_path": fig_path,
}

json_path = os.path.join(OUTPUT_DIR, "lab_jour1_metrics.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(metriques, f, indent=2, ensure_ascii=False)

print(f"Metriques JSON  : {json_path}")
print(f"\n{'='*55}")
print(f"  RÉSUMÉ FINAL")
print(f"{'='*55}")
all_ok = all(metriques["checkpoints"].values())
for k, v in metriques["checkpoints"].items():
    print(f"  {'OK' if v else 'FAIL'}  {k}")
print(f"\n  {'LAB VALIDE' if all_ok else 'CERTAINS CHECKPOINTS ECHOUES'}")

