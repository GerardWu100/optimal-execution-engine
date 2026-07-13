---
title: "De la variance d'ouverture aux ordres exécutables : réparer une chaîne de recherche causale"
description: "Audit des horodatages et des unités d'une prévision de variance réalisée, de son évaluation walk-forward et de son lien avec Almgren-Chriss."
date: 2026-07-12
image: images/cover-optimal-execution.png
categories: ["Quantitative Research", "Capital Markets"]
---

Une prévision n'est exploitable que si chaque entrée existe au moment où elle est calculée. Sa sortie doit aussi avoir l'unité attendue par le modèle suivant. La première version de ce projet échouait sur ces deux points. La variable de variance d'ouverture couvrait les 12 barres suivies et se confondait donc avec la cible. La prévision de variance franchissait ensuite une interface documentée comme une volatilité, sans passer par une racine carrée.

La chaîne corrigée prend une décision à 9 h 55, heure de l'Est. Six barres de cinq minutes sont alors connues. Le modèle prévoit la variance des six barres suivantes, puis l'ordre parent simulé commence à 10 h. Cette chronologie sépare l'information de la cible. Une seconde correction convertit la variance prévue en volatilité avant de modifier l'urgence d'exécution.

Le résultat demande pourtant une mise en garde : l'erreur absolue moyenne, ou Mean Absolute Error (MAE), du modèle linéaire reste égale à $2.44\times10^{-14}$. Ce chiffre ne mesure pas une capacité à prévoir le marché. Le jeu de démonstration est si déterministe et lisse que la corrélation entre la variance d'ouverture et celle de la fenêtre suivante atteint $0.999999998$, même sans chevauchement.

## La chronologie de la décision

L'échantillon AAPL suivi contient 55 séances. Chacune comprend 12 barres de cinq minutes, de 9 h 30 à 10 h 25, heure de l'Est. Notons $d$ la date de négociation, $t\in\{0,\ldots,11\}$ l'indice d'une barre et $P_{d,t}$ sa clôture en dollars.

Le rendement logarithmique entre deux clôtures est

$$
r_{d,t}=\log\left(\frac{P_{d,t}}{P_{d,t-1}}\right).
$$

Le ratio dans le logarithme est sans unité, tout comme $r_{d,t}$. La première barre n'a pas de clôture antérieure dans la séance et ne contribue donc à aucun rendement intrajournalier.

Fixons la coupure d'information à $m=6$ barres. La variable d'ouverture utilise les rendements qui se terminent avant cette coupure :

$$
RV^{\text{open}}_d=\sum_{t=1}^{m-1}r_{d,t}^{2}.
$$

Ici, $RV^{\text{open}}_d$ est la variance réalisée d'ouverture. Elle est connue après la clôture de 9 h 55. La cible utilise les rendements dont la clôture finale arrive plus tard :

$$
RV^{\text{rem}}_d=\sum_{t=m}^{11}r_{d,t}^{2}.
$$

Ici, $RV^{\text{rem}}_d$ est la variance réalisée de la fenêtre restante. Son premier terme est le rendement entre la clôture connue de 9 h 55 et celle, encore inconnue, de 10 h. Aucun rendement cible n'existe au moment de la prévision.

<pre>
09:30                 09:55  10:00                 10:25
|------ 6 known bars ------| |------ 6 future bars ------|
       feature window       ^       target/execution
                             forecast and order arrival
</pre>

Cette construction reprend l'idée usuelle de la volatilité réalisée, estimée par la somme des carrés de rendements à haute fréquence, mais sur une courte fenêtre pédagogique plutôt que sur une séance complète. Andersen, Bollerslev, Diebold et Labys fournissent le cadre empirique plus général de la volatilité réalisée des rendements financiers.[^1]

Le code rejette désormais toute séance qui ne contient aucune barre après la coupure :

```python
bar_counts = returns_frame.groupby(["symbol", "trade_date"]).size()
invalid_sessions = bar_counts.loc[bar_counts <= opening_window_bars]
if not invalid_sessions.empty:
    raise ValueError(
        "Each session must contain at least one bar after opening_window_bars."
    )

remaining_frame = returns_frame.loc[
    returns_frame["bar_index"] >= opening_window_bars
].copy()
```

Cette vérification empêche une fenêtre future vide de recréer le défaut initial ou de produire une cible nulle dépourvue de sens.

## Les variables disponibles à 9 h 55

Le modèle linéaire utilise sept variables. Quatre proviennent de la fenêtre d'ouverture du jour :

- la variance réalisée d'ouverture ;
- le rendement logarithmique d'ouverture ;
- l'amplitude haut-bas divisée par le prix d'ouverture ;
- $\log(1+V^{\text{open}}_d)$, où $V^{\text{open}}_d$ est le volume en actions de la fenêtre d'ouverture.

Trois variables résument les cibles passées de la fenêtre restante :

$$
L_{d,1}=RV^{\text{rem}}_{d-1},
$$

$$
M_{d,k}=\frac{1}{k}\sum_{j=1}^{k}RV^{\text{rem}}_{d-j},\qquad k\in\{5,10\}.
$$

$L_{d,1}$ est le retard d'une séance et $M_{d,k}$ une moyenne mobile sur les $k$ séances antérieures. Le décalage précède la moyenne :

```python
merged["lag_1_remaining_realized_variance"] = merged.groupby("symbol")[
    "target_remaining_realized_variance"
].shift(1)

merged["rolling_5d_remaining_realized_variance"] = merged.groupby("symbol")[
    "target_remaining_realized_variance"
].transform(lambda values: values.shift(1).rolling(5, min_periods=5).mean())
```

L'ancienne variable divisait le volume d'ouverture par le volume total de la fenêtre suivie. Son dénominateur contenait des barres postérieures à 9 h 55. Il était donc inconnu au moment de la prévision. Le logarithme du seul volume d'ouverture ferme cette fuite et réduit l'écart d'échelle numérique avec les autres variables.

La variable et la cible corrigées diffèrent pour chacune des journées modélisées. Leur plus petit écart absolu vaut $2.756\times10^{-8}$ unité de variance.

![La variance d'ouverture et la variance de la fenêtre restante sont distinctes mais très colinéaires](images/01_feature_target_partition.png)

Les points ne sont plus sur la droite d'égalité : la variable ne contient donc plus la cible. Leur courbe presque droite vient des trajectoires de prix très lisses et déterministes du jeu de démonstration. La séparation causale corrige le protocole de recherche, mais elle ne transforme pas cet échantillon en preuve empirique.

## Modèles walk-forward et fonctions de perte

La plus longue moyenne mobile demande dix cibles antérieures, ce qui laisse 45 lignes de modélisation. Chaque fenêtre walk-forward s'ajuste sur 20 lignes consécutives et teste les cinq suivantes. Elle avance ensuite de cinq lignes. On obtient cinq blocs de test sans chevauchement, soit 25 prévisions hors échantillon.

Les trois modèles restent volontairement simples :

1. La persistance prévoit $\widehat{RV}^{\text{rem}}_d=L_{d,1}$.
2. La moyenne mobile prévoit $\widehat{RV}^{\text{rem}}_d=M_{d,5}$.
3. Les moindres carrés ordinaires estiment

$$
\widehat{RV}^{\text{rem}}_d=\beta_0+\sum_{j=1}^{7}\beta_jx_{d,j},
$$

où $x_{d,j}$ est la variable $j$, $\beta_0$ l'ordonnée à l'origine et $\beta_j$ son coefficient ajusté. Pour une matrice de conception $X$ et un vecteur cible $y$, les moindres carrés choisissent le vecteur $\widehat{\beta}$ qui minimise la somme des carrés des résidus :

$$
\widehat{\beta}=\arg\min_{b}\lVert Xb-y\rVert_2^2.
$$

Pour $N$ prévisions, une variance observée $y_i$ et une variance prévue $\widehat{y}_i$, la MAE vaut

$$
\operatorname{MAE}=\frac{1}{N}\sum_{i=1}^{N}|y_i-\widehat{y}_i|.
$$

La racine de l'erreur quadratique moyenne, ou Root Mean Squared Error (RMSE), vaut

$$
\operatorname{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\widehat{y}_i)^2}.
$$

La perte QLIKE calculée par le code est

$$
\operatorname{QLIKE}=\frac{1}{N}\sum_{i=1}^{N}\left[\log(\widehat{y}_i)+\frac{y_i}{\widehat{y}_i}\right].
$$

Les prévisions sont bornées par le bas à $10^{-12}$ avant le calcul de QLIKE afin que le logarithme et la division soient définis. Patton explique pourquoi une fonction de perte résistante à une mesure approchée et bruitée de la volatilité compte dans la comparaison des prévisions.[^2]

| Modèle | MAE (unités de variance) | RMSE (unités de variance) | QLIKE moyen |
|---|---:|---:|---:|
| Linéaire | $2.438\times10^{-14}$ | $2.678\times10^{-14}$ | -11.296174 |
| Persistance | $7.034\times10^{-9}$ | $7.034\times10^{-9}$ | -11.296173 |
| Moyenne mobile 5 jours | $2.117\times10^{-8}$ | $2.117\times10^{-8}$ | -11.296163 |

![Erreurs walk-forward des modèles sur une échelle logarithmique](images/02_model_error_comparison.png)

L'échelle logarithmique rend visible le grand écart numérique. L'interprétation doit rester étroite : une combinaison linéaire de variables synthétiques très colinéaires extrapole presque exactement ce jeu de données. Chaque ajustement ne dispose que de 20 observations pour sept variables et une constante. L'échantillon ne contient ni bruit réaliste ni marché indépendant. Le résultat vérifie la chaîne de calcul, pas un modèle de trading.

## La variance n'est pas la volatilité

Le calendrier d'exécution attend une volatilité, notée $\sigma$, en unité de rendement décimal. Le modèle de recherche prévoit une variance, notée $RV$, en unité de rendement au carré. La conversion découle de la définition de la variance :

$$
\operatorname{Var}(r)=\sigma^2.
$$

En prenant la racine carrée non négative, on obtient

$$
\widehat{\sigma}^{\text{rem}}_d=\sqrt{\widehat{RV}^{\text{rem}}_d}.
$$

Si la variance prévue vaut $4\times10^{-6}$, la transmettre comme volatilité donne $0.000004$. La volatilité correcte est $0.002$, soit 20 points de base de rendement. Dans cet exemple, l'erreur d'interface réduit l'entrée d'un facteur 500.

Le lien d'exécution corrigé effectue la conversion explicitement et rejette les prévisions négatives ou non finies :

```python
predicted_variance = float(forecast_variance_by_trade_date[trade_date])
if predicted_variance < 0.0 or not np.isfinite(predicted_variance):
    raise ValueError("Variance forecasts must be finite and non-negative.")

predicted_volatility = float(np.sqrt(predicted_variance))
```

La version simplifiée du calendrier d'Almgren-Chriss définit l'urgence par

$$
u=\max(\lambda\sigma,10^{-6}),
$$

où $\lambda$ est le coefficient d'aversion au risque configuré et $\sigma$ la volatilité de la fenêtre restante. Au temps normalisé $\tau\in[0,1]$, la fraction de l'inventaire restant est

$$
x(\tau)=\frac{\sinh(u(1-\tau))}{\sinh(u)}.
$$

Une valeur plus élevée de $u$ déplace l'exécution vers le début de l'horizon. Le projet reprend le compromis entre risque et coût ainsi que la forme hyperbolique associée à Almgren et Chriss. Il s'agit toutefois d'une approximation pédagogique, sans leur calibration complète de l'impact temporaire et permanent.[^3]

## Résultats d'exécution après la coupure

Pour chacune des 25 dates prévues, l'ordre arrive après la clôture de 9 h 55. Cette dernière clôture connue sert de prix d'arrivée. Les calendriers traitent contre les six barres allant de 10 h à 10 h 25. L'achat de 10 000 actions est réparti selon le Time-Weighted Average Price (TWAP), le chemin simplifié d'Almgren-Chriss et un calendrier de Volume-Weighted Average Price (VWAP).

Pour la tranche $i$, notons $q_i$ le nombre d'actions exécutées et $V_i$ le volume de marché. Le simulateur utilise le taux de participation $q_i/V_i$ et l'impact

$$
I_i=2+25\frac{q_i}{V_i}
$$

en points de base. Si $P_i$ est la clôture de la barre, le prix d'achat simulé est

$$
P_i^{\text{fill}}=P_i\left(1+\frac{I_i}{10{,}000}\right).
$$

Notons $P^{\text{arr}}$ le prix d'arrivée de 9 h 55 et $Q=\sum_iq_i$ la taille de l'ordre parent. L'implementation shortfall total, en points de base, vaut

$$
C_{\text{bps}}=10{,}000\frac{\sum_iq_i(P_i^{\text{fill}}-P^{\text{arr}})}{P^{\text{arr}}Q}.
$$

Le reporting corrigé utilise ce notionnel au prix d'arrivée et pondère le prix d'exécution moyen par le nombre d'actions. L'ancien calcul moyennait les coûts des tranches, ce qui donnait le même poids à une petite tranche qu'à une grande.

| Calendrier | Coût moyen (pb) | Médiane (pb) | Écart-type (pb) | 90e centile (pb) | Jours |
|---|---:|---:|---:|---:|---:|
| Almgren-Chriss | 23.467 | 23.466 | 0.152 | 23.667 | 25 |
| TWAP | 23.483 | 23.482 | 0.152 | 23.682 | 25 |
| VWAP oracle | 24.288 | 24.285 | 0.157 | 24.495 | 25 |

![Coût moyen d'exécution après la coupure avec un écart-type](images/03_execution_cost_comparison.png)

Almgren-Chriss bat TWAP d'environ $0.016$ point de base en moyenne dans ce simulateur. Cet écart est minuscule et dépend de l'aversion au risque et des constantes d'impact choisies. Le calendrier VWAP est un oracle, car il utilise les volumes futurs réalisés pour répartir les actions. Son coût est plus élevé ici parce que la règle d'impact pénalise un fort taux de participation. Ce résultat décrit surtout le simulateur, pas l'exécution VWAP en production.

Bertsimas et Lo formulent l'exécution comme un problème de contrôle dynamique soumis à l'impact de marché et à l'arrivée d'information.[^4] Le projet ne résout pas ce problème complet. Il ne modélise ni carnet d'ordres, ni dynamique du spread, ni position dans la file, ni calibration séparée des impacts temporaire et permanent, ni choix de place de négociation.

## Ce que la réparation établit

Le code impose maintenant une seule chaîne cohérente :

<pre>
bars ending by 09:55
        -> causal features
        -> forecast 10:00-10:25 variance
        -> square root to volatility
        -> parent order arrives
        -> simulate only 10:00-10:25 bars
</pre>

Cette chaîne corrige les deux défauts matériels et deux problèmes connexes révélés par l'audit. Elle ne sauve pas la conclusion empirique. Une expérience crédible demanderait des données de séance complète, une structure de bruit réaliste, bien plus d'historique et des paramètres d'impact estimés plutôt que choisis.

La leçon tient à la méthode. Un découpage chronologique entre entraînement et test ne peut pas réparer une variable dont l'horodatage franchit la frontière de décision. Des tests réussis ne corrigent pas une incohérence d'unité entre deux modules. Il faut tracer le temps et les unités avant d'interpréter une mesure de performance.

## Références

[^1]: Andersen, T. G., Bollerslev, T., Diebold, F. X., and Labys, P. (2003). [Modeling and Forecasting Realized Volatility](https://doi.org/10.1111/1468-0262.00418). *Econometrica*, 71(2), 579-625.
[^2]: Patton, A. J. (2011). [Volatility Forecast Comparison Using Imperfect Volatility Proxies](https://doi.org/10.1016/j.jeconom.2010.03.034). *Journal of Econometrics*, 160(1), 246-256.
[^3]: Almgren, R., and Chriss, N. (2001). [Optimal Execution of Portfolio Transactions](https://doi.org/10.21314/JOR.2001.041). *Journal of Risk*, 3(2), 5-39.
[^4]: Bertsimas, D., and Lo, A. W. (1998). [Optimal Control of Execution Costs](https://doi.org/10.1016/S1386-4181(97)00012-8). *Journal of Financial Markets*, 1(1), 1-50.
