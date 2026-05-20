"""
====================================================
 Section 7 — IoU (Intersection over Union)
====================================================
L'IoU mesure la qualité de localisation d'une boîte prédite
par rapport à la boîte de référence (ground truth).

Formule :
    IoU = Aire(intersection) / Aire(union)

Valeur entre 0 (aucun overlap) et 1 (superposition parfaite).

Basé sur : https://github.com/yugmerabtene/nexa-computer-vision/blob/main/JOUR-01.md
"""

import os
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Chemins dynamiques ──────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# 1. FONCTION IoU
# ═══════════════════════════════════════════════════════════════════════════
def iou(box_a, box_b):
    """
    Calcule le score IoU entre deux boîtes.
    Format des boîtes : (x1, y1, x2, y2)
      - (x1, y1) = coin supérieur gauche
      - (x2, y2) = coin inférieur droit

    Étapes :
      1) Calculer les coordonnées de l'intersection
      2) Vérifier qu'elle est valide (non nulle)
      3) Calculer les aires et appliquer la formule
    """
    # Étape 1 — Coordonnées de l'intersection
    x_gauche = max(box_a[0], box_b[0])
    y_haut   = max(box_a[1], box_b[1])
    x_droite = min(box_a[2], box_b[2])
    y_bas    = min(box_a[3], box_b[3])

    # Étape 2 — Pas d'intersection si les boîtes ne se chevauchent pas
    if x_droite <= x_gauche or y_bas <= y_haut:
        return 0.0

    # Étape 3 — Aires
    aire_inter = (x_droite - x_gauche) * (y_bas - y_haut)
    aire_a     = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    aire_b     = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    aire_union = aire_a + aire_b - aire_inter

    return aire_inter / aire_union


# ═══════════════════════════════════════════════════════════════════════════
# 2. EXEMPLE NUMÉRIQUE (issu du cours)
# ═══════════════════════════════════════════════════════════════════════════
box_gt   = (40, 60, 181, 191)   # Boîte de référence (ground truth)
box_pred = (52, 60, 192, 191)   # Boîte prédite (décalage de 12 px)

score = iou(box_pred, box_gt)

print("=" * 50)
print("  Calcul IoU — Exemple numérique")
print("=" * 50)
print(f"  Boîte GT   : {box_gt}")
print(f"  Boîte pred : {box_pred}")
print(f"  Aire GT    : {(box_gt[2]-box_gt[0]) * (box_gt[3]-box_gt[1])} px²")
print(f"  Aire pred  : {(box_pred[2]-box_pred[0]) * (box_pred[3]-box_pred[1])} px²")
print(f"  IoU        : {score:.3f}")
print("=" * 50)

# Interprétation
if score >= 0.8:
    niveau = "Excellente localisation"
elif score >= 0.5:
    niveau = "Bonne localisation"
elif score > 0.0:
    niveau = "Localisation partielle"
else:
    niveau = "Aucun overlap"
print(f"  Interpretation : {niveau}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. VISUALISATION DE L'IoU
# ═══════════════════════════════════════════════════════════════════════════
def visualiser_iou(box_gt, box_pred, score, ax, titre=""):
    """Dessine les deux boîtes et leur intersection sur un axe matplotlib."""
    # Zone totale à afficher
    x_min_view = min(box_gt[0], box_pred[0]) - 15
    y_min_view = min(box_gt[1], box_pred[1]) - 15
    x_max_view = max(box_gt[2], box_pred[2]) + 15
    y_max_view = max(box_gt[3], box_pred[3]) + 15

    # Fond blanc
    ax.set_facecolor("#f8f8f8")

    # Zone d'intersection
    x_inter1 = max(box_gt[0], box_pred[0])
    y_inter1 = max(box_gt[1], box_pred[1])
    x_inter2 = min(box_gt[2], box_pred[2])
    y_inter2 = min(box_gt[3], box_pred[3])

    if x_inter2 > x_inter1 and y_inter2 > y_inter1:
        rect_inter = patches.Rectangle(
            (x_inter1, y_inter1),
            x_inter2 - x_inter1, y_inter2 - y_inter1,
            linewidth=0, facecolor="#9b59b6", alpha=0.5, label="Intersection"
        )
        ax.add_patch(rect_inter)

    # Boîte Ground Truth (verte)
    rect_gt = patches.Rectangle(
        (box_gt[0], box_gt[1]),
        box_gt[2] - box_gt[0], box_gt[3] - box_gt[1],
        linewidth=2.5, edgecolor="#27ae60", facecolor="none", label="Ground Truth"
    )
    ax.add_patch(rect_gt)

    # Boîte prédite (orange)
    rect_p = patches.Rectangle(
        (box_pred[0], box_pred[1]),
        box_pred[2] - box_pred[0], box_pred[3] - box_pred[1],
        linewidth=2.5, edgecolor="#e67e22", facecolor="none", label="Prediction"
    )
    ax.add_patch(rect_p)

    ax.set_xlim(x_min_view, x_max_view)
    ax.set_ylim(y_max_view, y_min_view)   # Axe Y inversé (convention image)
    ax.set_title(f"{titre}\nIoU = {score:.3f}", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.grid(True, alpha=0.3)


# ═══════════════════════════════════════════════════════════════════════════
# 4. COURBE SHIFT VS IoU (exercice bonus du cours)
# ═══════════════════════════════════════════════════════════════════════════
# On décale progressivement la boîte prédite et on observe l'impact sur l'IoU

shifts     = list(range(0, 65, 5))
iou_scores = []

for s in shifts:
    box_shifted = (box_gt[0] + s, box_gt[1], box_gt[2] + s, box_gt[3])
    iou_scores.append(iou(box_shifted, box_gt))

# Décalage à 0, 12, 30 et 50 pour les exemples visuels
exemples = [(0, "Shift=0\n(parfait)"), (12, "Shift=12\n(léger)"),
            (30, "Shift=30\n(moyen)"), (50, "Shift=50\n(fort)")]


# ═══════════════════════════════════════════════════════════════════════════
# 5. FIGURE COMPLÈTE
# ═══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 14))
fig.suptitle("Section 7 — IoU (Intersection over Union)", fontsize=16, fontweight="bold")

# Ligne 1 : 4 exemples visuels de décalage
for col, (shift, label) in enumerate(exemples):
    ax = fig.add_subplot(2, 4, col + 1)
    b_shifted = (box_gt[0] + shift, box_gt[1], box_gt[2] + shift, box_gt[3])
    s = iou(b_shifted, box_gt)
    couleur_titre = "#27ae60" if s >= 0.8 else "#e67e22" if s >= 0.5 else "#e74c3c"
    visualiser_iou(box_gt, b_shifted, s, ax, titre=label)
    ax.set_title(f"{label}\nIoU = {s:.3f}", fontsize=10, color=couleur_titre)

# Ligne 2 gauche-centre : courbe shift vs IoU
ax_courbe = fig.add_subplot(2, 4, (5, 7))
ax_courbe.plot(shifts, iou_scores, marker="o", linewidth=2.5,
               color="steelblue", markersize=6, label="IoU")
ax_courbe.axhline(0.8, color="#27ae60", linestyle="--", linewidth=1.5, label="Seuil 0.8 (excellent)")
ax_courbe.axhline(0.5, color="#e67e22", linestyle="--", linewidth=1.5, label="Seuil 0.5 (acceptable)")
ax_courbe.fill_between(shifts, iou_scores, alpha=0.15, color="steelblue")
ax_courbe.set_title("Courbe : Décalage (shift) vs IoU", fontsize=12)
ax_courbe.set_xlabel("Décalage en pixels")
ax_courbe.set_ylabel("Score IoU")
ax_courbe.set_xlim([0, 60])
ax_courbe.set_ylim([0, 1.05])
ax_courbe.legend(fontsize=9)
ax_courbe.grid(True, alpha=0.3)

# Annotate le point shift=12 (exemple du cours)
idx_12 = shifts.index(10) if 10 in shifts else 2
ax_courbe.annotate(f"shift=12\nIoU≈{iou((box_gt[0]+12, box_gt[1], box_gt[2]+12, box_gt[3]), box_gt):.2f}",
                   xy=(12, iou((box_gt[0]+12, box_gt[1], box_gt[2]+12, box_gt[3]), box_gt)),
                   xytext=(20, 0.75),
                   arrowprops=dict(arrowstyle="->", color="gray"),
                   fontsize=9, color="gray")

# Ligne 2 droite : tableau récapitulatif
ax_table = fig.add_subplot(2, 4, 8)
ax_table.axis("off")
headers = ["Shift (px)", "IoU", "Qualité"]
rows = []
for s in [0, 10, 20, 30, 40, 50, 60]:
    b = (box_gt[0] + s, box_gt[1], box_gt[2] + s, box_gt[3])
    v = iou(b, box_gt)
    q = "Excellent" if v >= 0.8 else "Bon" if v >= 0.5 else "Faible" if v > 0 else "Nul"
    rows.append([str(s), f"{v:.3f}", q])

table = ax_table.table(
    cellText=rows, colLabels=headers,
    loc="center", cellLoc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.6)

# Colorier les lignes selon la qualité
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
    elif col == 2 and row > 0:
        val = rows[row - 1][1]
        v = float(val)
        if v >= 0.8:
            cell.set_facecolor("#d5f5e3")
        elif v >= 0.5:
            cell.set_facecolor("#fdebd0")
        else:
            cell.set_facecolor("#fadbd8")

ax_table.set_title("Tableau shift vs IoU", fontsize=10, fontweight="bold")

plt.tight_layout()
fig_path = os.path.join(OUTPUT_DIR, "7_1_iou.png")
plt.savefig(fig_path, dpi=130)
plt.close()
print(f"Figure sauvegardee : {fig_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. EXPORT JSON (métriques — format lab du cours)
# ═══════════════════════════════════════════════════════════════════════════
metriques = {
    "iou_score":  round(score, 3),
    "bbox_gt":    list(box_gt),
    "bbox_pred":  list(box_pred),
    "seuil_bon":  0.5,
    "seuil_excellent": 0.8,
    "interpretation": niveau,
    "courbe_shift_iou": {str(s): round(iou((box_gt[0]+s, box_gt[1], box_gt[2]+s, box_gt[3]), box_gt), 3)
                         for s in shifts},
    "figure_path": fig_path
}

json_path = os.path.join(OUTPUT_DIR, "7_1_iou_metrics.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(metriques, f, indent=2, ensure_ascii=False)

print(f"Metriques JSON  : {json_path}")
print(f"\n--- Résumé ---")
print(f"IoU (shift=12)  : {score:.3f}  → {niveau}")
print(f"IoU (shift=0)   : 1.000  → Superposition parfaite")
print(f"IoU (shift=30)  : {iou((box_gt[0]+30,box_gt[1],box_gt[2]+30,box_gt[3]),box_gt):.3f}  → seuil 0.5 proche")

