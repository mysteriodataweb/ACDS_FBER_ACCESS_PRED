# Rapport de présentation

## Prédiction de l'accès FTTH au Togo

- **Projet :** modélisation prédictive de l'accès à la fibre optique des ménages
- **Data scientist :** Alfred Kwami BIAM
- **Programme :** African Citizen Data Scientist (ACDS) – École Centrale Casablanca

Tous les chiffres de ce rapport proviennent de l'exécution du notebook
`notebooks/Fiber_Acces_pred.ipynb` (fichier de référence : `Output/rapport_modele.json`).

---

## Résumé exécutif

À partir des données de recensement de **30 558 ménages** et de features satellites MOSAIKS, un modèle
**Random Forest** estime la probabilité qu'un ménage ait accès à la fibre.

| Métrique (jeu de test, 6 112 ménages) | Valeur |
|---|---|
| AUC | 0,874 |
| F1-score | 0,830 |
| Accuracy | 0,828 |
| Précision | 0,801 |
| Rappel | 0,862 |

**Ce que le modèle sait faire** : ordonner les ménages d'une zone déjà enquêtée selon leur probabilité
d'accès ; parmi les ménages classés « Très élevé », 85,0 % ont effectivement la fibre, contre
10,6 % parmi ceux classés « Faible ».

**Ce qu'il ne sait pas (encore) faire** : prédire l'accès dans une **zone jamais observée**. Les données
satellites ne couvrent que 423 tuiles et presque tous les ménages évalués sont dans une tuile connue du
modèle. Sur des tuiles inconnues, l'AUC tombe à **0,632** : les décisions d'extension du réseau
vers de nouvelles zones ne doivent pas reposer sur ce modèle seul.

---

## 1. Pour les décideurs et responsables politiques

### Enjeux
L'accès au haut débit est un levier d'inclusion (éducation, santé, services publics), de développement
économique et d'équité territoriale. Mieux connaître les ménages connectés et ceux qui ne le sont pas
permet de cibler les politiques publiques.

### Ce que montrent les données
- Les classes sont équilibrées : **48,8 %** des ménages de l'échantillon ont accès à la fibre.
- L'accès dépend d'abord de **la zone** : le taux d'accès varie de 0 % à 100 % selon la tuile satellite,
  et la simple connaissance du taux de la tuile suffit à atteindre une AUC de 0,849.
- Au sein d'une zone, le **raccordement électrique dans la concession**, la cuisson au gaz,
  l'**électroménager** (fer à repasser électrique, réfrigérateur, gazinière), la voiture et le **type de
  logement** affinent la prédiction ; à l'inverse, l'absence d'électroménager, le logement traditionnel et
  la cuisson au charbon ou au bois sont associés à un accès plus faible. Ces liens restent faibles pris
  isolément (corrélations ≤ 0,15).

### Usages recommandés
1. **Prioriser au sein des zones couvertes** : cibler les ménages à fort potentiel non encore raccordés
   pour accélérer l'adoption là où l'infrastructure existe.
2. **Repérer les ménages à faible potentiel** dans des zones bien équipées : cible naturelle de
   programmes d'accompagnement (équipement, formation numérique).
3. **Ne pas utiliser le modèle seul** pour décider de nouvelles zones de déploiement : compléter par des
   données d'infrastructure (réseau existant, coûts, densité) et une validation terrain.

### Indicateurs de suivi proposés
Taux de raccordement effectif vs. probabilités prédites, délai d'adoption après installation, écart de
performance entre zones (urbain/rural) une fois une variable régionale disponible.

---

## 2. Pour les opérateurs télécoms

### Segmentation par niveau de potentiel (jeu de test)

| Niveau | Probabilité prédite | Ménages | Taux d'accès réel observé |
|---|---|---|---|
| Faible | 0 – 0,25 | 2 213 | 10,6 % |
| Moyen | 0,25 – 0,50 | 689 | 25,7 % |
| Élevé | 0,50 – 0,75 | 1 291 | 72,7 % |
| Très élevé | 0,75 – 1 | 1 919 | 85,0 % |

Le taux réel croît nettement d'un niveau à l'autre : les probabilités sont exploitables pour ordonner des
actions commerciales (prospection, offres, planification des installations).

### Mise en œuvre
- Le modèle est exporté sous forme de **pipeline complet** : `ftth-predict` l'applique directement à un
  fichier de ménages au format du recensement et renvoie, par `ID`, la probabilité et le niveau de potentiel.
- Le seuil de décision par défaut est 0,5 ; un seuil maximisant le F1 (0,446) est fourni à titre
  d'option. Le choix du seuil doit refléter le coût relatif d'un faux positif (démarche commerciale inutile)
  et d'un faux négatif (client potentiel ignoré).

### Hypothèses à valider
Les estimations de rentabilité par segment (coûts de déploiement, ARPU, taux d'adoption) ne peuvent pas être
tirées de ces données : elles nécessitent les données internes de l'opérateur et un test terrain (par
exemple une campagne pilote sur un échantillon de ménages « Très élevé » vs. « Moyen »).

---

## 3. Pour les data scientists

### Données
- 30 558 ménages × 4 043 colonnes : 39 variables socio-démographiques, 4 000 features MOSAIKS, `ID`,
  `Connexion`, `BoxLabel` et la cible `Accès internet`.
- Problèmes traités : accents corrompus dans les libellés (`�`/`?`), colonne MOSAIKS n°0 sans nom,
  `BoxLabel` constante, **`Connexion` = fuite de la cible** (manquante ⇔ pas d'accès), exclue.
- Les 4 000 features MOSAIKS ne prennent que **423 valeurs distinctes** (une par tuile).

### Pipeline
| Étape | Choix |
|---|---|
| Variables nominales | one-hot |
| Taille du ménage | imputation médiane + standardisation |
| Équipements | 0/1 |
| MOSAIKS | standardisation + PCA (6 composantes, 95 % de variance, solveur `covariance_eigh`) |
| Split | 80/20 stratifié, `random_state=45` (contrainte du challenge) |
| Sélection | AUC moyenne en validation croisée 5 plis sur le train, prétraitement réappris par pli |
| Optimisation | recherche aléatoire (15 essais) : 500 arbres, `max_features=0.3`, AUC CV 0,860 → 0,861 |

### Comparaison des modèles (validation croisée 5 plis, train)

| Modèle | AUC (moyenne ± écart-type) | F1 |
|---|---|---|
| **Random Forest** | 0,860 ± 0,005 | 0,818 |
| XGBoost | 0,855 ± 0,005 | 0,807 |
| LightGBM | 0,852 ± 0,003 | 0,812 |
| MLP Neural Network | 0,828 ± 0,002 | 0,785 |
| SVM | 0,778 ± 0,005 | 0,691 |
| Logistic Regression | 0,681 ± 0,015 | 0,618 |

### Robustesse géographique

| | AUC |
|---|---|
| Test (split officiel, tuiles connues) | 0,874 |
| Référence « taux d'accès de la tuile », sans ML | 0,849 |
| Validation croisée spatiale (tuiles jamais vues) | 0,632 ± 0,077 |
| Validation spatiale, régression logistique | 0,662 |
| Validation spatiale, Random Forest sans les features MOSAIKS | 0,608 |

Lecture : l'essentiel du gain apporté par MOSAIKS sur le split officiel vient de la reconnaissance des
tuiles. C'est légitime pour prédire dans des zones déjà enquêtées, mais cela ne se transfère pas à de
nouvelles zones, où un modèle plus simple (régression logistique) généralise mieux que les modèles d'arbres.

### Interprétabilité
Les valeurs SHAP (figures `Output/shap_*.png`) confirment le rôle dominant des composantes MOSAIKS,
suivies de la taille du ménage, du raccordement électrique dans la concession, du type de logement et
des équipements. Ces effets sont des **associations**, pas des causes.

### Pistes d'amélioration
1. Features géographiques plus fines et transférables : coordonnées, distance au réseau fibre existant,
   densité de bâti, données d'infrastructure de l'opérateur.
2. Utiliser la validation spatiale comme critère de sélection si l'objectif est l'extension du réseau.
3. Ajouter une variable régionale pour auditer l'équité urbain/rural.
4. Calibrer les probabilités si elles servent à des estimations de volumes.

---

## Annexes

- Notebook : `notebooks/Fiber_Acces_pred.ipynb`
- Limites de la version précédente et corrections : `docs/AUDIT.md`
- Rapport machine : `Output/rapport_modele.json` ; comparaison : `Output/model_comparison.csv` ;
  prédictions du test : `Output/predictions_test.csv`
- Références : Rolf et al. (2021), *A generalizable and accessible approach to machine learning with
  global satellite imagery*, Nature Communications ; Lundberg & Lee (2017), *A Unified Approach to
  Interpreting Model Predictions*, NeurIPS.
