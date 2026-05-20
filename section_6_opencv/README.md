# Section 6 — Manipulation d'images avec OpenCV

Ce dossier contient tous les scripts Python correspondant à la **section 6** du cours.

---

## 📁 Structure

```
section_6_opencv/
├── 6_1_lecture_affichage.py     # Lire et afficher une image
├── 6_2_niveaux_de_gris.py       # Conversion en niveaux de gris (cvtColor)
├── 6_3_redimensionnement.py     # Redimensionner (resize)
├── 6_4_histogramme.py           # Calcul et affichage d'histogramme
├── 6_5_seuillage.py             # Seuillage global, Otsu, adaptatif
└── main_section6.py             # Pipeline complet — toutes les étapes
```

---

## ▶️ Lancer un script

```bash
# Exemple : lire et afficher une image
python 6_1_lecture_affichage.py

# Pipeline complet
python main_section6.py
```

---

## 📚 Notions couvertes

| Script | Notion |
|--------|--------|
| `6_1` | `cv2.imread`, `img.shape`, canaux BGR |
| `6_2` | `cv2.cvtColor`, formule Gris = 0.299R + 0.587G + 0.114B |
| `6_3` | `cv2.resize`, interpolations (`INTER_LINEAR`, `INTER_CUBIC`) |
| `6_4` | `cv2.calcHist`, histogramme couleur et niveaux de gris |
| `6_5` | `cv2.threshold`, méthode d'Otsu, seuillage adaptatif |

---

## 🛠️ Dépendances

```bash
pip install opencv-python matplotlib numpy
```

