# Audit du dépôt : limites relevées et corrections

Ce document recense les limites de la version précédente du projet (commit `679a07f`) et la façon dont
chacune a été corrigée. Les chiffres « après » proviennent de l'exécution complète du notebook
`notebooks/Fiber_Acces_pred.ipynb` (voir aussi `Output/rapport_modele.json`).

## 1. Constat principal : une performance surestimée pour l'usage visé

| | Avant | Après |
|---|---|---|
| Choix du modèle | meilleure AUC **sur le test** | meilleure AUC en **validation croisée 5 plis sur le train** |
| AUC test annoncée | 0,885 (rapport) / 0,873 (notebook) | 0,874 |
| Référence naïve « taux d'accès de la tuile » | non mesurée | 0,849 |
| AUC sur des zones jamais vues | non mesurée | 0,632 ± 0,077 (régression logistique : 0,662) |

Les 4 000 features MOSAIKS décrivent une **tuile satellite**, pas un ménage : les 30 558 ménages ne se
répartissent que sur **423 tuiles**, et **99,9 %** des ménages du test sont dans une tuile déjà vue à
l'entraînement. Le modèle apprend donc surtout le taux d'accès de chaque tuile : une simple moyenne par
tuile, sans aucun apprentissage, fait presque aussi bien que le meilleur modèle. Sur des tuiles absentes de
l'entraînement (validation croisée groupée par tuile), l'AUC chute nettement.

**Conséquence pratique** : le modèle est utile pour prioriser des ménages **dans des zones déjà enquêtées**,
mais il ne permet pas, en l'état, d'identifier de façon fiable de **nouvelles zones** de déploiement.
Le README, le notebook et le rapport ont été corrigés en ce sens.

## 2. Bugs corrigés

| # | Limite | Correction |
|---|---|---|
| 1 | Notebook non reproductible : les cellules `SelectKBest` et PCA échouaient en `MemoryError`, mais les cellules suivantes affichaient les résultats d'une exécution antérieure (sorties de plusieurs exécutions mélangées). | Pipeline économe en mémoire (`float32`, aucune copie du DataFrame de 1 Go, PCA `covariance_eigh` : 5 s au lieu de 197 s pour le même résultat). Notebook réexécuté intégralement, de haut en bas. |
| 2 | Le modèle exporté (`best_model_*.joblib`) était **inutilisable** : la standardisation (faite « à la main », colonne par colonne) et la PCA n'étaient pas sauvegardées. | Tout le prétraitement est dans un `ColumnTransformer` sauvegardé avec le modèle (`Output/models/ftth_pipeline.joblib`) ; `ftth-predict` l'applique directement à des données brutes. |
| 3 | Les **identifiants étaient modifiés** : la regex `[^a-zA-Z0-9]` était appliquée à tout le DataFrame (`ID_88M0FMHZAM` → `ID88M0FMHZAM`), rendant les prédictions impossibles à joindre aux données d'origine. | Le nettoyage ne touche que les variables catégorielles ; un test vérifie que les `ID` sont inchangés. |
| 4 | La feature MOSAIKS n°0 (colonne sans nom `" "`) était renommée `.0` et laissée **hors de la PCA** ; la liste `mosaiks_cols` était recalculée à 1 seule colonne (préfixe `.` au lieu de ` .`). | Les 4 000 colonnes sont renommées `mosaiks_0` … `mosaiks_3999` et toutes passent par la PCA. |
| 5 | Composantes PCA non remises à l'échelle (PC1 ~50× plus dispersée que les autres features) : la régression logistique ne convergeait pas (165 s de calcul en validation croisée). L'avertissement était masqué par `warnings.filterwarnings('ignore')`. | Standardisation intégrée au pipeline de la régression logistique (8 s, convergence) ; plus de filtre global des avertissements. Blanchir la PCA pour tous les modèles a été testé et écarté : cela dégrade le MLP (AUC 0,83 → 0,78). |
| 6 | Deux dossiers de sortie (`../Output` et `notebooks/Output`, chemins Windows) et 5 exécutions horodatées versionnées. | Un seul dossier `Output/`, noms de fichiers fixes (l'historique est dans Git). |
| 7 | La cellule SHAP « beeswarm » réécrivait `shap_beeswarm.png` **sans noms de features** (« Feature 0, 1… »). | Une seule figure, générée avec les noms issus du pipeline. |
| 8 | `pd.cut` sans `include_lowest` (une probabilité de 0 n'avait pas de niveau) et deux barèmes de priorité incohérents (0,25/0,5/0,75 puis 0,6/0,8). | Une seule fonction `potential_level`, bornes incluses, testée. |
| 9 | Analyse d'incertitude réalisée sur LightGBM alors que le modèle retenu était le Random Forest. | Réalisée sur le modèle retenu, avec une courbe de calibration. |
| 10 | Paramètres sans effet : LightGBM `subsample` sans `subsample_freq` (bagging désactivé), MLP `learning_rate='adaptive'` (ignoré avec `adam`), XGBoost `use_label_encoder` (obsolète) ; commentaire « gamma = régularisation L1 » erroné. | Paramètres corrigés ou supprimés, commentaires rectifiés. |
| 11 | `!pip install -r requirements.txt` échouait (chemin relatif depuis `notebooks/`). | Installation documentée dans le README ; le notebook trouve le package seul. |

## 3. Méthodologie

| # | Limite | Correction |
|---|---|---|
| 12 | Variables nominales (type de logement, éclairage, cuisson) encodées en entiers 0..n (`LabelEncoder`) : ordre artificiel pour les modèles linéaires. | Encodage one-hot (`OneHotEncoder(handle_unknown="ignore")`). |
| 13 | Modèle choisi sur le test et aucune validation croisée, alors que le notebook annonçait une « validation croisée stratifiée (5 folds) ». | Comparaison en CV 5 plis sur le train, prétraitement réappris dans chaque pli ; test utilisé une seule fois. |
| 14 | Hyperparamètres « optimisés » fixés à la main ; `GridSearchCV`/`RandomizedSearchCV` importés mais jamais utilisés. | Recherche aléatoire en CV, adoptée seulement si elle améliore l'AUC de validation. |
| 15 | Dépendance géographique (423 tuiles) non mesurée. | Référence « taux de la tuile » et validation croisée spatiale (`StratifiedGroupKFold`). |
| 16 | `Connexion` exclue sans justification : c'est une fuite de la cible (manquante ⇔ pas d'accès). | Fuite documentée ; les features sont définies par **liste blanche** (test dédié). |
| 17 | Calculs inutilisés : `SelectKBest` (500 features), filtre de variance, `socio_demo_data`, copies `df1`. | Supprimés. |
| 18 | Libellés illisibles après nettoyage (`ElectricitCEETcompteurdanslaconcession`) et dictionnaire de corrections partiel. | Table de libellés canoniques en français correct, nettoyage idempotent et testé. |
| 19 | Seuil de décision : réglage suggéré mais non fait. | Seuil F1-optimal estimé sur les prédictions hors-pli du train (jamais sur le test), fourni en option. |

## 4. Documentation

| # | Limite | Correction |
|---|---|---|
| 20 | `RAPPORT_PRESENTATION.md` : chiffres **incohérents avec le notebook** (AUC 88,5 %, rappel 91,2 %, F1 84,5 %) ; matrice de confusion avec 1 712 positifs au lieu de 2 982 ; déséquilibre « 25–35 % » alors que les classes sont à 48,8 % / 51,2 % ; « PCA 99 % » au lieu de 95 % ; `RandomizedSearchCV`, `GroupKFold` par région et SHAP sur 1 000 ménages décrits mais jamais réalisés ; ROI chiffrés sans source ; « 88,5 % de précision AUC » (confusion AUC / précision). | Rapport réécrit à partir des résultats réels ; les hypothèses business sont présentées comme telles. |
| 21 | README : phrase tronquée, coquilles, mauvais `cd`, URL de clone différente du dépôt, colonnes `longitude`/`latitude` inexistantes, Git LFS non mentionné, fichiers listés absents. | README réécrit. |
| 22 | `requirements.txt` et `environment.yml` divergents ; `catboost`, `plotly`, `tqdm`, `ipywidgets` installés mais inutilisés. | Dépendances déclarées une seule fois dans `pyproject.toml`. |

## 5. Hygiène du dépôt

| # | Limite | Correction |
|---|---|---|
| 23 | Fichiers parasites : 2 archives `.zip` d'une ancienne version du notebook (1,3 Mo chacune), capture d'écran de code `optimization.png`, image inutilisée `arriere-pla.avif`, script ponctuel `remove_emojis.py`, 14 sorties horodatées. | Supprimés ; l'image du notebook est déplacée dans `docs/images/`. |
| 24 | `.gitignore` ignorait `.github/` (aucune CI possible) ; `Métadonnées.csv` (2,6 Ko) stocké dans Git LFS. | `.gitignore` nettoyé ; dictionnaire des variables en fichier texte normal (`Data/metadonnees.csv`). |
| 25 | Aucun code réutilisable, aucun test, aucune CI. | Package `src/ftth_access`, tests `pytest` sur données synthétiques, lint `ruff`, CI GitHub Actions. |

## 6. Limites qui demeurent

- **Généralisation spatiale faible** : voir la section 1. Pistes : features géographiques plus fines
  (coordonnées, distance au réseau existant, densité de bâti), validation spatiale comme critère de sélection,
  modèles plus régularisés.
- **Résolution MOSAIKS** : 423 tuiles seulement ; l'information satellite est un descripteur de zone.
- **Données statiques** : aucune dimension temporelle.
- **Pas de variable régionale** : l'équité entre régions (urbain/rural) ne peut pas être auditée directement.
- **Valeur business** : les segments et ROI du rapport sont des hypothèses à valider sur le terrain.
