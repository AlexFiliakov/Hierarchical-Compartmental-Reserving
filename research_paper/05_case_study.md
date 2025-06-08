# 5 Compartmental reserving case study

In this section, we demonstrate how to fit hierarchical compartmental models of varying complexity to paid and outstanding claims development data simultaneously. We also introduce models with parameter variation by both accident and development year, in addition to an application of the previously outlined approach for integrating pricing and reserving cycle information into the modeling process.

Our first model is based on the case study presented in (Morris (2016)), which models outstanding and cumulative paid claims using a Gaussian distribution. In Section 3 we proposed modeling incremental payments with a right-skewed process distribution (e.g., lognormal), which a second model demonstrates. Model 2 also introduces parameter variation by development year, showcasing the usability and flexibility of brms for specifying varying effects. Finally, we build on this work further with a third model, which incorporates pricing and reserving cycle trends into the modeling process. The purpose of this procedure is to capture performance drift across time and apply judgment for individual years—particularly for less mature years, where hierarchical growth curve approaches typically shrink parameters back to an "all-years" average.

The workflow in this case study involves six steps:

1. **Data preparation:** Create a training data set, up to the penultimate calendar year, and test data set based on the most recent calendar year.
2. **Model building:** Develop model structures including process distributions, prior parameter distributions, and hierarchical levels. We omit prior predictive distribution reviews in the text for brevity (see Section 3.4 for more detail).
3. **Training:** Fit the models on the training data and review in-sample posterior predictive checks.
4. **Testing / Selection:** Review each model's predictions against the latest calendar year's paid loss ratios, and select the most appropriate model.
5. **Fitting:** Re-fit (train) the selected model on the combined training and test data, which include the latest calendar year.
6. **Reserving:** Review final model, predict future claims payments and set reserve.

We also have the "lower triangle" of development, which allows us to critique our reserve estimates against actual values.

## 5.1 Data preparation

The data comprise Schedule P Workers' Compensation paid and incurred development data for company 337 (Fannin (2018)), as in the (Morris (2016)) case study, depicted in Figure 5.1.

![Paid and outstanding loss ratio development by accident year shown for all 10 development years](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/VisualiseData-1.png)

Figure 5.1: Paid and outstanding loss ratio development by accident year shown for all 10 development years

We split the data into a training set, with data up to 1996; a test data set which contains the 1997 calendar year movement; and finally, to review our reserve, a "validation" set which contains development for all accident years to age 10. Figure 5.2 shows the validation scheme for the incremental paid loss ratios.

![Train, test, and validation data splits](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/VisualiseTrainTestData-1.png)

Figure 5.2: Train, test, and validation data splits

## 5.2 Model building

We build and train three hierarchical compartmental reserving models:

- Model 1: A Gaussian distribution model, based on the original case study in (Morris (2016)), fitted to outstanding and cumulative paid amounts
- Model 2: A lognormal distribution model, fitted to outstanding and incremental paid loss ratios with additional parameter variation by development year
- Model 3: An enhancement of model 2 which incorporates market cycle data and judgement to influence forecasts for less mature accident years

### 5.2.1 Model 1: Gaussian distribution

Model 1 is analogous to the first model outlined the original case study in (Morris (2016)). We assume a Gaussian process distribution and model outstanding and cumulative paid amounts. For the hierarchical structure, we assume $k_{er}, k_p$ are fixed by accident year and that $RLR[i], RRF[i]$ have correlated varying effects by accident year, with a weak LKJ prior on the correlation between them.

$$\begin{aligned}
y_{ij} &\sim \text{Normal}(\mu(t_{ij}; \Theta, \delta), \sigma[\delta]) \\
\mu(t_{ij}; \Theta, \delta) &= (1-\delta)OS_{ij} + \delta PD_{ij} \\
\delta &= \begin{cases}
0 & \text{if } y_{ij} \text{ is outstanding claim} \\
1 & \text{if } y_{ij} \text{ is paid claim}
\end{cases} \\
\Theta &= (k_{er}, RLR[i], k_p, RRF[i]) \\
OS_{ij} &= \Pi_i RLR[i] \frac{k_{er}}{k_{er} - k_p}(e^{-k_p t_j} - e^{-k_{er} t_j}) \\
PD_{ij} &= \Pi_i RLR[i] RRF[i] \frac{1}{k_{er} - k_p}(k_{er}(1 - e^{-k_p t_j}) - k_p(1 - e^{-k_{er} t_j}))
\end{aligned} \tag{5.1}$$

Next, we specify priors for the parameters being estimated, based on judgement and intuition:

$$\begin{aligned}
\log(\sigma[\delta]) &\sim \text{StudentT}(1, 0, 1000) \\
k_{er} &\sim \text{Lognormal}(\log(3), 0.1) \\
k_p &\sim \text{Lognormal}(\log(1), 0.1) \\
RLR[i] &\sim \mu_{RLR} + u_{RLR} \\
RRF[i] &\sim \mu_{RLR} + u_{RRF} \\
\mu_{RLR} &\sim \text{Lognormal}(\log(0.7), 0.2) \\
\mu_{RRF} &\sim \text{Lognormal}(\log(0.8), 0.1) \\
(u_{RLR}, u_{RRF})' &\sim \text{MultivariateNormal}(0, D\Omega D) \\
\Omega &\sim \text{LKJCorr}(1) \\
D &= \begin{pmatrix} \sigma_{RLR} & 0 \\ 0 & \sigma_{RRF} \end{pmatrix} \\
\sigma_{RLR} &\sim \text{StudentT}(10, 0, 0.2)^+ \\
\sigma_{RRF} &\sim \text{StudentT}(10, 0, 0.1)^+
\end{aligned}$$

In summary, we anticipate:

- **A moderately high reported loss ratio**, reflected by a prior median $RLR$ equal to 70%, with prior CoV around 20%
- **A relatively fast rate of reporting**, reflected by a prior median $k_{er}$ equal to 3. This gives a value of claims reported in the first development year equal to $\Pi RLR (1 - e^{-3}) = \Pi RLR \cdot 95\%$, where $\Pi$ denotes ultimate earned premiums. The prior CoV of 10% covers the interval [2.5, 3.5].
- **Some degree of case overreserving**. A 0.8 prior median for $RRF$ translates to 80% of outstanding losses becoming paid losses on average. The prior CoV of 10% covers the possibility of adequate reserving.
- **A rate of payment $k_p$ which is slower than the rate of earning and reporting**. The prior median of 1 and prior CoV of 10% covers a $k_p$ between 0.85 and 1.2 with approximately 95% probability.

Note that each lognormal median is logged in the above specification, since $\exp(\mu)$ is the median of a lognormal distribution with parameters $\mu$ and $\sigma$, while $\sigma$ is approximately the CoV.

### 5.2.2 Model 2: Lognormal distribution and additional structure

For models 2 & 3 we will assume:

- a lognormal data-generating distribution with constant CoV across development and accident years;
- loss ratio dependent variables (rather than loss amounts), with incremental paid loss ratios being our principle target; and
- accident and development year varying effects for each compartmental parameter.

The implication of assuming a lognormal process distribution is that we estimate the median loss ratio development process. The number of parameters in the vector $\Theta$ will vary between models 2 and 3:

$$\begin{aligned}
y_{ij} &\sim \text{Lognormal}(\log(\mu(t_j; \Theta, \delta)), \sigma[\delta]^2) \\
\mu(t_j; \Theta, \delta) &= (1-\delta)OS_{ij} + \delta(PD_{ij} - PD_{i,j-1}) \\
\delta &= \begin{cases}
0 & \text{if } y \text{ is outstanding claim} \\
1 & \text{if } y \text{ is paid claim}
\end{cases}
\end{aligned} \tag{5.2}$$

For model 2 we use similar prior assumptions to model 1, except for $\sigma[\delta]$ (since we are modeling loss ratios rather than loss amounts). We set this prior to a lognormal with a median of 10%:

$$\sigma[\delta] \sim \text{Lognormal}(\log(0.1), 0.2)$$

Models 2 and 3 allow each compartmental model parameter to vary by both accident and development year. The approach is analogous to the "row" and "column" parameters defined in statistical models for the chain-ladder, but with compartmental parameters varying rather than the expected outcome. As before, each parameter shrinks to a population estimate for sparse accident/development years:

$$\begin{aligned}
\Theta &= \mu_\Theta + u_\Theta \\
\mu_\Theta &= (\mu_{RLR}, \mu_{RRF}, \mu_{k_{er}}, \mu_{k_p}) \\
u_{\Theta_1} &= (u_{RLR}[i,j], u_{RRF}[i,j]) \\
u_{\Theta_2} &= (u_{k_{er}}[i,j], u_{k_p}[i,j]) \\
u_{\Theta_1} &\sim \text{MultivariateNormal}(0, \Sigma) \\
\Sigma &= D\Omega D \\
\Omega[i,j] &\sim \text{LKJCorr}(1) \\
D &= \text{Diag}(\sigma_{RLR}[i,j], \sigma_{RRF}[i,j]) \\
u_{k_{er}}[i,j] &\sim \text{Normal}(0, \sigma_{k_{er}}[i,j]) \\
u_{k_p}[i,j] &\sim \text{Normal}(0, \sigma_{k_p}[i,j]) \\
\sigma_{RLR}[i,j] &\sim \text{StudentT}(10, 0, 0.7)^+ \\
\sigma_{RRF}[i,j] &\sim \text{StudentT}(10, 0, 0.5)^+ \\
\sigma_{k_{er}}[i,j], \sigma_{k_p}[i,j] &\sim \text{StudentT}(10, 0, 0.3)^+
\end{aligned}$$

### 5.2.3 Model 3: Pricing and reserving cycle sub-model

The final model in this case study builds pricing and reserving cycle information into the modeling process, as introduced in Section 4.4.

In lieu of market cycle information for the study, we compile an earned premium movement index and raise it to a judgmental power (0.6) to proxy a rate change index. This defines a set of reported loss ratio multipliers, $RLM_i$. We also select reserve robustness multipliers, $RRF_i$, which are set to correlate with $RLM_i$ to reflect learnings from model 2 (shown later in this section). These multipliers (Figure 5.3) are used to define prior values for $RLR$ and $RRF$ by accident year relative to the oldest year.

![Proxy cycle indices](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/ProxyRateIndex-1.png)

Figure 5.3: Proxy cycle indices

We model the extent to which $RLR$ and $RRF$ depend on the $RLM$ and $RRM$ indices with two additional parameters, $\lambda_{RLR}, \lambda_{RRF}$:

$$\begin{aligned}
RLR[i,j] &= \mu_{RLR}[i] + u_{RLR}[i,j] \\
\mu_{RLR}[i] &= \mu_{RLR_1} \cdot RLM[i]^{\lambda_{RLR}} \\
RRF[i,j] &= \mu_{RRF}[i] + u_{RRF}[i,j] \\
\mu_{RRF}[i] &= \mu_{RRF_1} \cdot RRM[i]^{\lambda_{RRF}} \\
\lambda_{RLR} &\sim \text{Normal}(1, 0.25) \\
\lambda_{RRF} &\sim \text{Normal}(1, 0.25)
\end{aligned}$$

The prior means for $\lambda_{RLR}$ and $\lambda_{RRF}$ are set to 1, which assumes that the expected loss ratio movements year-over-year directly correlate with the selected indices. This allows performance drift across those accident years in which market conditions are changing. However, the priors are weakly regularizing to allow inferences to pull away from our initial judgments values less than 1, for example, would indicate a weaker correlation between the indices and loss ratio movements.

The varying effects $u_{RLR}[i,j]$ and $u_{RRF}[i,j]$ will override the index-driven accident year estimates if there is sufficient information in the data relative to the priors. The largest impact of the market cycle submodels should therefore be seen for less mature accident years where we expect $u_i$ to shrink toward zero.

Note that in practice, we parametrise the model slightly differently to be able to estimate compartmental parameters on the standard normal scale before back-transforming them (see appendix for brms implementation).

For simplicity, we maintain the existing $RLR$ and $RRF$ population priors on $\mu_{RLR_1}$ and $\mu_{RRF_1}$. All other assumptions from model 2 are carried forward.

## 5.3 Training

We train the models on loss and loss ratio development up to the 1996 calendar year. The review of model 1 is kept brief, with greater emphasis placed on models 2 and 3.

### 5.3.1 Training model 1

To review model 1 against the training data, we assess 100 outstanding and cumulative paid loss ratio posterior predictive samples by accident year and development year, shown in Figure 5.4 and 5.5.

![Model 1. Outstanding loss ratio versus 100 simulations from the posterior predictive distribution](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/PosteriorChecksSimple2-1.png)

Figure 5.4: Model 1. Outstanding loss ratio versus 100 simulations from the posterior predictive distribution

![Model 1. Cumulative paid loss ratio versus 100 simulations from the posterior predictive distribution](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/PosteriorChecksSimple1-1.png)

Figure 5.5: Model 1. Cumulative paid loss ratio versus 100 simulations from the posterior predictive distribution

At a glance, the model appears to provide reasonable in-sample coverage of the data points themselves. However, the spaghetti plots illustrate an incompatibility of the constant-variance Gaussian process distribution with our intuition of the claims development process: in particular, we do not usually expect negative outstanding amounts or reductions in cumulative payments over time. Modeling cumulative payments with a constant process variance allows the model to generate negative posterior paid increments. Furthermore, the Gaussian assumption does not prevent negative outstanding posterior realizations.

### 5.3.2 Training model 2

For model 2, we target outstanding and incremental paid loss ratios, and replace the Gaussian process distribution assumption with a lognormal. Each compartmental model parameter is able to vary by accident and development year.

![Model 2. Outstanding loss ratio versus 100 simulations from the posterior predictive distribution](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/model2OSplot-1.png)

Figure 5.6: Model 2. Outstanding loss ratio versus 100 simulations from the posterior predictive distribution

The posterior realizations for the constant CoV lognormal distribution now better reflect our understanding of typical development data: outstanding loss ratio variance shrinks by development year, and outstandings do not fall below 0, as shown in Figure 5.6.

![Model 2. Incremental paid loss ratio versus 100 simulations from the posterior predictive distribution](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/model2incrpaidplot-1.png)

Figure 5.7: Model 2. Incremental paid loss ratio versus 100 simulations from the posterior predictive distribution

The incremental paid loss ratio samples also appear reasonable Figure 5.7). As with the outstandings, we observe a reduction in variance over development time, together with strictly positive realizations. Consequently, when we cumulate the paid loss ratios, the process behavior aligns with expectations (Figure 5.8).

![Model 2. Cumulative paid loss ratio versus 100 simulations from the posterior predictive distribution](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/model2cumpaid-1.png)

Figure 5.8: Model 2. Cumulative paid loss ratio versus 100 simulations from the posterior predictive distribution

To assess the impact of the inclusion of additional varying effects compared with model 1, we inspect marginal posterior mean parameter estimates by development year. If these have any systematic trends, we may consider incorporating them into the ODEs (analytical solutions) to carry through into extrapolation and, hence, reserve setting. Alternatively, we could model certain of the compartmental parameters to be functions of development year in our statistical model.

![Model 3. Marginal compartmental posterior parameter estimates by development year, with 3 degrees of freedom B-spline smoothers](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/DevTimeTrends-1.png)

Figure 5.9: Model 3. Marginal compartmental posterior parameter estimates by development year, with 3 degrees of freedom B-spline smoothers

The model estimates significant variation for $RLR$ and $RRF$ by development year, supporting the decision to allow them to vary across this dimension (Figure 5.9). However, the trends appear somewhat cyclical, with uncertain direction beyond development year 9 in most cases. Therefore we opt not to change the compartmental/ODE structure to account for directional trends into unseen development years.

In Figures 5.10 and 5.11 we review posterior parameter distributions for $RLR$ and $RRF$, in addition to the correlation between them by accident year. A traceplot is shown for the latter to diagnose convergence (inspected for all models and parameters).

![Model 3. Prior versus posterior parameter densities](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/CompromisePriorPosterior-1.png)

Figure 5.10: Model 3. Prior versus posterior parameter densities

The moderate positive posterior correlation between the $RLR$ and $RRF$ varying effects by accident year mirrors the original compartmental reserving paper (Morris (2016)), and is suggestive of a reserving cycle effect, i.e. prudent case reserves in a hard market and vice versa.

![Model 3. Prior versus posterior parameter densities](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/CompromiseCorrelation-1.png)

Figure 5.11: Model 3. Prior versus posterior parameter densities

We can see the correlation between $RLR$ and $RRF$ more clearly by visualizing the marginal posterior parameter distributions by accident year. Overlaying the year-over-year percentage changes in ultimate earned premiums reveals performance improvements for increases in premium and deteriorations for reductions in premium (Figure 5.12). This suggests that the movements in premium may be partially rate change driven.

![Model 3. RLR and RRF posterior distributions by accident year](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/CompromisePosteriorTrend-1.png)

Figure 5.12: Model 3. RLR and RRF posterior distributions by accident year

Note that this correlation breaks down between 1995 and 1996, where a premium reduction is not mirrored by a deterioration in expected performance deterioration.

With only two data points available for 1996, this could be a consequence of regularization. More specifically, we expect the model to credibility weight between the 1996 data and an all-years average. However, the prior years have been relatively favorable up until 1995, where a significant deterioration in performance is estimated.

If we intend to carry forward the prior years' correlation between premium movements and performance, then regularization back to an all-years average loss ratio is not desirable. This motivates the third and final candidate model in this case study.

### 5.3.3 Training model 3

We train our third model on the data, which integrates market cycle indices and judgement into the model to capture trends in $RLR$ and $RRF$ by accident year, and proceed to review the posterior predictive checks (Figures 5.13 – 5.15).

![Model 3. Posterior predictive checks for outstanding loss ratio development](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/PosteriorChecksCompromise1-1.png)

Figure 5.13: Model 3. Posterior predictive checks for outstanding loss ratio development

![Model 3. Posterior predictive checks for incremental paid loss ratio development](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/PosteriorChecksCompromise2-1.png)

Figure 5.14: Model 3. Posterior predictive checks for incremental paid loss ratio development

![Model 3. Posterior predictive checks for cumulative paid loss ratio development](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/PosteriorChecksCompromise3-1.png)

Figure 5.15: Model 3. Posterior predictive checks for cumulative paid loss ratio development

The in-sample fits once again appear reasonable, but observe that in contrast to model 2, this model projects a performance deterioration across the more recent accident years, in which premiums have been reducing. This can be attributed to the use of the $RLM$ and $RRM$ indices which trend upward for more recent years. We also see that the $\lambda_{RLR}$ mean posterior has increased slightly to 1.06 from our prior of 1, whereas the $\lambda_{RRF}$ posterior is materially unchanged from the prior (Figure 5.16).

![Model 3. Prior versus posterior Lambda parameter densities](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/Model3Lambda-1.png)

Figure 5.16: Model 3. Prior versus posterior Lambda parameter densities

We visualise $RLR$ and $RRF$ posterior distributions by accident year once more (Figure 5.17) and observe a stronger correlation between their year-over-year and corresponding premium movements up to 1996. The model estimates that the 1996 accident year has a modest probability of inadequate case reserving ($RRF > 1$).

![Model 3. RLR and RRF posterior distributions by accident year](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/Model3PosteriorTrend-1.png)

Figure 5.17: Model 3. RLR and RRF posterior distributions by accident year

Finally, we review marginal $k_{er}$ and $k_p$ estimates by accident year to investigate trends for additional consideration within the model structure (Figure 5.18).

![Model 3. Marginal compartmental posterior parameter estimates by accident year, with 3 degrees of freedom B-spline smoothers](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/AYTrends-1.png)

Figure 5.18: Model 3. Marginal compartmental posterior parameter estimates by accident year, with 3 degrees of freedom B-spline smoothers

Observe that $k_p$ is estimated to trend upward by accident year, suggestive of a faster claims settlement process in more recent years. The model's accident year parameter variation takes care of this pattern, but if we expected a smoother trend we could model $k_p$ to increase by accident year monotonically. This would be analogous to the changing settlement rate model outlined in (Meyers (2015)), and is left as an exercise for the reader.

## 5.4 Testing and Selection

We exclude Model 1 from the selection process due to the incompatibilities of a Gaussian process distribution. Next, we predict incremental paid loss ratios using models 2 and 3 and compare these against actual loss ratios for the 1997 calendar year.

### 5.4.1 Testing model 2

We first inspect the model 2 future paid loss ratio development distributions by accident year, and overlay the actual one-year-ahead cumulative paid loss ratios (Figure 5.19).

![Model 2. Cumulative paid loss ratio one-step ahead hold-out performance](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/PosteriorOneStepComplex-1.png)

Figure 5.19: Model 2. Cumulative paid loss ratio one-step ahead hold-out performance

The one-step-ahead predictions are within the reserve uncertainty bands for most years; however, the model does not appear to perform as well at the mean level for 1996 and 1997. For 1997 in particular, the projections could be considered optimistic against projected 1995 and 1996 performance.

### 5.4.2 Testing model 3

Compared with model 2, model 3 perhaps does a better job on the one-step-ahead 1996 and 1997 accident year predictions (Figure 5.20).

![Model 3. Cumulative paid loss ratio one-step ahead hold-out performance](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/PosteriorOneStepCompromise-1.png)

Figure 5.20: Model 3. Cumulative paid loss ratio one-step ahead hold-out performance

By tracking proxy market cycle information, the model is able to better account for the increasing loss ratio trend between 1994 and 1996, and into the unseen 1997 accident year.

### 5.4.3 Model selection

To compare each of the models in detail, one-step-ahead incremental paid loss ratio actual-versus-expected plots and root-mean-square errors (RMSEs) are reviewed in Figure 5.21 and Table 5.1, respectively.

![Actual versus expected one-step paid loss ratio. Each circle represents a different accident year, with size relative to its earned premium](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/OneStepRMSEComparisonsFig-1.png)

Figure 5.21: Actual versus expected one-step paid loss ratio. Each circle represents a different accident year, with size relative to its earned premium

| Model | RMSE | Change |
|-------|------|--------|
| Model 2 | 2.55% | - |
| Model 3 | 1.11% | -56.4% |

Table 5.1: one-step ahead paid loss ratio RMSE comparisons

Model 3 offers a 56% one-step ahead RMSE improvement on model 2. The out-of-sample actual-versus-expected plots corroborate this and suggest that model 3 may be more predictive than model 2 at the mean level. We therefore select model 3 as the preferred structure.

In practice, a wide range of models could be tested, comprising structural and prior assumptions, with a view to maximizing one-step-ahead predictive performance. This approach is not taken here primarily for brevity, but also because favorable one-step-ahead performance may not translate to favorable 10-step-ahead performance. A more robust approach would perhaps be to predict n-step-ahead performance based on fitting each model to ever-smaller triangles and optimizing the trade-off between n-step-ahead performance estimates and the quantity of data used to derive the model parameters and performance estimates.

## 5.5 Fitting

Having selected model 3 for our reserving exercise, we fit it to the training and test data (i.e., triangles up to and including calendar year 1997) and compare actual reserves against model-estimated reserve uncertainty as a final validation step.

### 5.5.1 Fitting selected model

We retrain our selected model on all information known to date (in the context of this reserving exercise), and proceed to review reserve posterior intervals for cumulative paid loss ratios.

![Final model. Cumulative paid loss ratio full hold-out performance](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/FinalModelHoldouts-1.png)

Figure 5.22: Final model. Cumulative paid loss ratio full hold-out performance

Based on Figure 5.22, the model has done a reasonable job of predicting reserve uncertainty for 1996 and prior. The 1997 year had a deteriorating development profile and longer tail relative to prior years, which the model was able to anticipate to some extent by leveraging the $RLM$ and $RRM$ indices.

## 5.6 Setting reserves

We take a closer look at the actual-versus-predicted reserve by accident year in Figure 5.23. The same scale is adopted for all years to highlight the insignificance of earlier-year reserves relative to the later years.

![Final model. Age 10 reserve versus predicted reserve distribution by accident year](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/FinalModelReservesAY-1.png)

Figure 5.23: Final model. Age 10 reserve versus predicted reserve distribution by accident year

## 5.7 Discussion

The compartmental model predictions are reasonably accurate at the mean level, with some under- and overprediction for individual years. Across the more recent years, there is posterior mean overprediction for 1994, and underprediction for 1996 and 1997. However, the actual reserves fall within the estimated distributions.

The total estimated reserve distribution at age 10 in aggregate is depicted in Figure 5.24.

![Final model. Age 10 reserve versus predicted reserve distribution](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/FinalModelReservesFull-1.png)

Figure 5.24: Final model. Age 10 reserve versus predicted reserve distribution

The actual reserve was $130M against an estimated mean reserve of $123M (5% underprediction). This is at the 75th percentile of the estimated reserve distribution.

The underprediction can be attributed to the 1996 and 1997 accident years – reviewing the upper and lower triangles, we observe that both these years exhibited underreserving in contrast to the overreserving observed in prior years (Figure 5.25).

![Accident years 1996 and 1997 incurred loss ratio development (green and blue respectively) exhibit under-reserving in contrast to prior years](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/Visualise1997Incurred-1.png)

Figure 5.25: Accident years 1996 and 1997 incurred loss ratio development (green and blue respectively) exhibit under-reserving in contrast to prior years

Model 3 was able to forecast a deterioration through the market cycle submodel and estimated positive correlation between $RLR$ and $RRF$. However, with a marked shift in performance and just two data points at the time of fitting, it is perhaps unsurprising that the model's posterior mean reserve falls short.

We conclude that the incurred data are somewhat misleading in this study due to deteriorating performance and case reserve robustness for less mature years. However, the incorporation of market cycle information (and judgment), together with a separation of portfolio performance and reserve robustness assumptions, can facilitate challenge, scenario analysis, and communication of key uncertainties during the reserving process.

### References

Fannin, Brian A. 2018. _Raw: R Actuarial Workshops_. https://CRAN.R-project.org/package=raw.

Meyers, Glenn. 2015. _Stochastic Loss Reserving Using Bayesian MCMC Models_. CAS Monograph Series. http://www.casact.org/pubs/monographs/papers/01-Meyers.PDF; Casualty Actuarial Society.

Morris, Jake. 2016. _Hierarchical Compartmental Models for Loss Reserving_. Casualty Actuarial Society Summer E-Forum; https://www.casact.org/pubs/forum/16sforum/Morris.pdf.