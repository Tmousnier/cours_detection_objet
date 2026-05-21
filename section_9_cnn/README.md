# Section 9 — LAB-02 : Classification CNN sur Fashion-MNIST

Ce dossier contient le LAB-02 du cours :
[LAB-02.md — yugmerabtene/nexa-computer-vision](https://github.com/yugmerabtene/nexa-computer-vision/blob/main/LAB-02.md)

---

## Objectif pedagogique

A la fin du lab, comprendre :

```
comment une image entre dans un CNN
comment une convolution extrait des caracteristiques
a quoi servent ReLU et MaxPooling
comment le modele transforme une image en prediction
comment entrainer un CNN simple
comment lire la loss et l'accuracy
comment tester le modele sur une image inconnue
```

---

## Structure du dossier

```
section_9_cnn/
├── train_cnn.py         # Entrainement du CNN (5 epochs)
├── predict.py           # Prediction sur de nouvelles images
├── requirements.txt     # Dependances Python
├── models/
│   └── cnn_fashion_mnist.pth   # Modele sauvegarde (genere apres train)
└── README.md
```

**Figures generees dans `outputs/` :**
```
9_1_courbes_apprentissage.png   # Loss + Accuracy par epoch
9_2_predictions.png             # 8 predictions sur images de test
9_3_filtres_conv1.png           # Les 16 filtres appris par conv1
9_4_predict_grille.png          # 16 predictions aleatoires (predict.py)
9_5_predict_probabilites.png    # Detail probabilites d'une image
```

---

## Architecture CNN

```
Entree : image 28x28 (niveaux de gris)
         |
    [Conv2D 16 filtres 3x3] -> ReLU -> MaxPool 2x2
         | (16 x 14 x 14)
    [Conv2D 32 filtres 3x3] -> ReLU -> MaxPool 2x2
         | (32 x 7 x 7 = 1568)
    [Flatten]
         |
    [Fully Connected 128] -> ReLU
         |
    [Fully Connected 10]  <- sortie (1 par classe)
         |
    Classe predite (argmax)
```

**Nombre de parametres :** ~21 840

---

## Dataset : Fashion-MNIST

10 classes de vetements :

| Classe | Label |
|--------|-------|
| T-shirt/top | 0 |
| Pantalon    | 1 |
| Pull        | 2 |
| Robe        | 3 |
| Manteau     | 4 |
| Sandale     | 5 |
| Chemise     | 6 |
| Basket      | 7 |
| Sac         | 8 |
| Botte       | 9 |

---

## Installation

```bash
pip install -r section_9_cnn/requirements.txt
```

---

## Utilisation

```bash
# 1. Entrainement (telechargement auto + 5 epochs)
python section_9_cnn/train_cnn.py

# 2. Prediction sur de nouvelles images
python section_9_cnn/predict.py
```

---

## Resultats attendus

```
Epoch 1/5 | Loss: 0.6200 | Accuracy: 77.50%
Epoch 2/5 | Loss: 0.4100 | Accuracy: 85.20%
Epoch 3/5 | Loss: 0.3500 | Accuracy: 87.30%
Epoch 4/5 | Loss: 0.3200 | Accuracy: 88.50%
Epoch 5/5 | Loss: 0.3000 | Accuracy: 89.10%

Accuracy sur le jeu de test : ~88% a 91%
```

---

## Ce qu'on deduit

| Observation | Ce que ca signifie |
|-------------|-------------------|
| Loss diminue a chaque epoch | Le modele apprend (minimise l'erreur) |
| Accuracy augmente | Le modele reconnait mieux les classes |
| Filtres conv1 varies | Chaque filtre detecte un motif different |
| Bonne accuracy test ≈ train | Pas de surapprentissage (overfitting) |

---

## Questions pedagogiques

**Q1 : Pourquoi utilise-t-on une convolution ?**
> La convolution extrait automatiquement des motifs visuels : bords, textures, formes —
> sans avoir a les programmer manuellement.

**Q2 : A quoi sert ReLU ?**
> ReLU remplace les valeurs negatives par 0 et garde les positives.
> Elle ajoute de la non-linearite : sans elle, empiler des couches revient a une seule transformation lineaire.

**Q3 : A quoi sert MaxPooling ?**
> MaxPooling reduit la taille spatiale (divise par 2) en gardant les informations les plus fortes.
> Cela reduit le nombre de parametres et rend le modele plus robuste aux translations.

**Q4 : Pourquoi fait-on un Flatten ?**
> Les couches fully connected attendent un vecteur 1D.
> Flatten transforme les cartes 2D en vecteur pour la classification finale.

**Q5 : Pourquoi CrossEntropyLoss ?**
> Elle est adaptee a la classification multi-classes :
> elle combine Softmax + log-vraisemblance negative.

