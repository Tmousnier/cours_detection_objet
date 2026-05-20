# Section 7 — IoU (Intersection over Union)

Basé sur : [JOUR-01.md — yugmerabtene/nexa-computer-vision](https://github.com/yugmerabtene/nexa-computer-vision/blob/main/JOUR-01.md)

---

## 📐 Formule

```
         Aire(Intersection)
IoU = ────────────────────────
           Aire(Union)
```

| Score | Interprétation |
|-------|---------------|
| `>= 0.8` | Excellente localisation |
| `>= 0.5` | Bonne localisation (seuil standard) |
| `> 0.0`  | Localisation partielle |
| `= 0.0`  | Aucun overlap |

---

## 📁 Fichiers

```
section_7_iou/
├── 7_1_iou.py       # Calcul IoU, visualisation, courbe shift vs IoU
└── README.md

outputs/
├── 7_1_iou.png               # Figure complète
└── 7_1_iou_metrics.json      # Métriques exportées
```

---

## ▶️ Lancer

```bash
python section_7_iou/7_1_iou.py
```

---

## 🛠️ Dépendances

```bash
pip install opencv-python numpy matplotlib
```

