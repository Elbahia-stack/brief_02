# Projet Brief –  Predict Delivery Time 

## 🧾 Description
Ce projet a pour objectif de prédire le **temps total de livraison** (`DeliveryTime`) pour une entreprise de logistique.  
Il inclut :  
- L’exploration et l’analyse des données dans **Jupyter Notebook**  
- L’entraînement de **modèles de régression** (RandomForestRegressor et SVR)  
- Des **Tests Automatisés** pour vérifier la cohérence des données et la performance (MAE)  
- Un dataset d’exemple (`data_brief_2.csv`)

---

## 📂 Structure du projet
Voici l'organisation des fichiers du projet :

```text
│
├── Untitled.ipynb # Notebook pour l'exploration et l'analyse des données
├── data_brief_2.csv # Jeu de données utilisé pour l'entraînement et les tests
├── test_split.py # Tests unitaires pour vérifier la cohérence du split
├── requirements.txt # Dépendances Python nécessaires au projet
└── .github/workflows/  # Workflow GitHub Actions pour CI/CD
    └── python-tests.yml # Exécution automatique des tests à chaque push
```
## Modèles évalués

Les modèles suivants ont été entraînés et comparés sur un **jeu de test indépendant** :

- **RandomForestRegressor**

-**Support Vector Regressor (SVR)**


## 📊 Métriques utilisées

Pour évaluer les modèles, nous avons calculé :

**MAE** (Mean Absolute Error) : erreur absolue moyenne

**R²** (Coefficient de détermination) : proportion de variance expliquée

## 📈 Résultats des modèles
| Modèle              | MAE     | R²       |
|----------------------|---------|----------|
| Random Forest        | 7.43    | 0.76     |
| SVR (Support Vector) | 6.37    | 0.80     |

### Analyse
Le SVR obtient une MAE plus faible (6.37) et un R² légèrement meilleur (0.80), ce qui signifie qu’il prédit en moyenne plus précisément que le Random Forest.

### Conclusion
L’évaluation des modèles montre que :

Le Support Vector Regressor (SVR) présente les meilleures performances globales, avec une MAE = 6.37 et un R² = 0.80, indiquant une meilleure précision dans la prédiction du temps de livraison.

Le Random Forest Regressor reste toutefois une alternative solide, avec une MAE = 7.43 et un R² = 0.76, offrant une bonne robustesse et une interprétabilité plus élevée.

En conclusion, le modèle SVR est retenu comme modèle final pour cette preuve de concept, car il offre une meilleure capacité de généralisation et des prédictions plus précises sur le jeu de test.
Le Random Forest pourrait néanmoins être privilégié dans un futur déploiement si l’objectif est d’obtenir un modèle plus explicable et plus rapide à ajuster.