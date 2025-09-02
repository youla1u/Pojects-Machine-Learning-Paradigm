# Bayesian Inference & Ridge Regression (RBF / Polynomial)

## Introduction
- Deux démos complémentaires :
  1) Régression ridge sur features RBF et polynomiales.
  2) Inférence bayésienne pour moyenne et variance d’une loi normale.

## Partie 1 — Régression Ridge
- Données : X simulé (uniforme), y = densité gaussienne multivariée.
- Features RBF :
  - σ² via “median trick”.
  - Centroïdes échantillonnés depuis X.
  - Design: [1, φ₁(X), …, φ_k(X)].
- Features polynomiales :
  - Puissances d’une colonne de X (ex. [x, x², x³, x⁴]).
- Apprentissage :
  - Ridge: w = (ΦᵀΦ + λI)⁻¹ Φᵀy.
  - Évaluation MSE selon λ (split + CV).
- Résumé attendu :
  - λ=0 : modèle instable.
  - λ ↑ : meilleure stabilité, MSE ↓ jusqu’à un optimum (courbe en U attendue).

## Partie 2 — Inférence Bayésienne (Normale)
- Données : s ~ N(μ, σ²) (ex. μ≈2, σ²≈0.1).
- Posterior de μ (σ² connu) :
  - Prior: μ ~ N(μ₀, σ₀²).
  - Posterior: N(μ_post, σ_post²).
- Posterior de σ² (μ connu) :
  - Prior: σ² ~ Inverse-χ²(v, σ₀²).
  - Posterior: Inverse-χ²(v+n, échelle mise à jour).
- Résumé attendu :
  - Tirages de μ proches de la vraie moyenne.
  - Tirages de σ² proches de la vraie variance.

## Résultats clés
- Compromis biais–variance contrôlé par λ (ridge).
- Posteriors bayésiens se concentrent vers les vrais paramètres.
- Petits échantillons : privilégier CV et régularisation.

## Dépendances
- Python, NumPy, SciPy, Matplotlib, scikit-learn.

## À faire / Extensions
- Augmenter n, choisir k-means pour centroïdes RBF.
- GridSearchCV pour λ (échelle log).
- Standardisation des features.
- Utiliser `scipy.stats.invgamma` pour σ².

## Structure (indicative)
- `ridge_rbf_poly.py` — génération données + RBF/Poly + ridge + CV.
- `bayes_normal.py` — posteriors pour μ et σ² + tirages.
- `figures/` — courbes MSE et histogrammes posteriors.
