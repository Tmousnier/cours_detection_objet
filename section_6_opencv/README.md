# Section 6 — Manipulation d'images avec OpenCV

Ce dossier contient tous les scripts Python correspondant à la **section 6** du cours  
[JOUR-01.md — yugmerabtene/nexa-computer-vision](https://github.com/yugmerabtene/nexa-computer-vision/blob/main/JOUR-01.md)

---

## 📁 Structure

```
section_6_opencv/
├── 6_1_lecture_affichage.py     # cv2.imread, img.shape, BGR vs RGB
├── 6_2_niveaux_de_gris.py       # cv2.cvtColor — formule pondérée
├── 6_3_redimensionnement.py     # cv2.resize — INTER_AREA / INTER_CUBIC
├── 6_4_histogramme.py           # cv2.calcHist + cv2.equalizeHist
├── 6_5_seuillage.py             # seuillage global, Otsu, adaptatif
├── 6_6_contours_boites.py       # cv2.findContours, Canny, boundingRect
└── README.md                    # cette documentation
```

Les figures générées sont sauvegardées dans :
```
outputs/
├── 6_3_redimensionnement.png
├── 6_4_histogramme.png
└── 6_6_contours_boites.png
```

---

## 📚 Notions couvertes

| Script | Section | Fonctions clés | Notion |
|--------|---------|----------------|--------|
| `6_1` | 6.1 | `cv2.imread` | Lecture, `shape`, `dtype`, canaux BGR |
| `6_2` | 6.2 | `cv2.cvtColor` | Gris = 0.299R + 0.587G + 0.114B |
| `6_3` | 6.3 | `cv2.resize` | Interpolations INTER_AREA / INTER_CUBIC |
| `6_4` | 6.4 | `cv2.calcHist`, `cv2.equalizeHist` | Histogramme + amélioration du contraste |
| `6_5` | 6.5 | `cv2.threshold`, `adaptiveThreshold` | Seuillage fixe, Otsu, adaptatif |
| `6_6` | 6.6 | `cv2.findContours`, `cv2.Canny`, `cv2.boundingRect` | Contours → boîtes candidates |

---

## ▶️ Lancer un script

```bash
# Activer l'environnement virtuel
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / Mac

# Exemple : contours et boîtes
python section_6_opencv/6_6_contours_boites.py
```

> Les scripts utilisent `matplotlib.use("Agg")` et `plt.savefig()` :  
> les figures sont **sauvegardées** dans `outputs/` au lieu d'être affichées.

---

## 🛠️ Dépendances

```bash
pip install opencv-python numpy matplotlib
```

