# PVC

Résolution du Problème du Voyageur de Commerce (TSP)  
Implémentation en Python d’heuristiques constructives et d’algorithmes exacts pour le TSP, incluant **Plus Proche Points (PPP)**, **PVCPrim** et **Branch and Bound (HDS)**. Génération de statistiques et visualisation graphique des résultats.

---

## Résolution du Problème du Voyageur de Commerce (TSP)

Ce projet implémente plusieurs méthodes pour résoudre le **Problème du Voyageur de Commerce (TSP)** en Python, allant des heuristiques constructives aux algorithmes exacts.

---

## Fonctionnalités

- **PPP (Point le Plus Proche)** : heuristique constructive rapide.
- **OptPPP** : amélioration de PPP via décroisement des arêtes.
- **PVCPrim / OptPrim** : approximation basée sur un arbre couvrant minimum.
- **HDS** : algorithme exact basé sur **Branch and Bound** avec heuristique de demi-somme.
- Analyse statistique des heuristiques sur ensembles de points aléatoires.
- Visualisation des résultats via histogrammes et comparaisons graphiques.

---

## Prérequis

- Python 3.12 ou supérieur
- Modules Python :
  - `numpy`
  - `matplotlib`
  - `heapq` (intégré à Python)

---

## Structure du projet

.
├── src
│ ├── PVC.py # Structures et algorithmes principaux
│ ├── PVC_points.py # Gestion des points et distances
│ ├── heap.py # Structures et algorithmes du tas, changement de priorité optimal
├── data # Données de tests et fichiers txt
├── algorithmes
│ ├── ppp.pdf
│ ├── prim.pdf
│ ├── hds.pdf
├── exemple_utilisation.ipynb # Notebook d'exemples
├── statistiques.ipynb # Notebook pour analyse et visualisation
├── rapport.pdf # Rapport complet
├── requirements.txt # Dépendances Python
└── README.md


---

## Utilisation

### Exemple simple

```python
from src.PVC_points import PVC_points
import numpy as np

# Générer 10 points aléatoires
points = np.random.rand(10, 2)
tsp = PVC_points(points)
tsp.charger_de_liste(points)

# Exécuter les algorithmes
cycle_ppp = tsp.PPP(points[0])
cycle_opt = tsp.OptPPP(points[0])
cycle_prim = tsp.PVCPrim()
best_cycle = tsp.HDS()

# Afficher les résultats
print("PPP :", tsp.longueur(cycle_ppp))
print("OptPPP :", tsp.longueur(cycle_opt))
print("PVCPrim :", tsp.longueur(cycle_prim))
print("HDS :", tsp.longueur(cycle_hds))

---

Auteur

Mohamed Elakef Zenagui
