# 4 Modeling hierarchical structures and correlation

In the previous section we discussed generative models for claims development. We will continue this line of thought and add more complexity in the form of hierarchies for describing claims emergence pattern variation by accident year.

## 4.1 Introduction to hierarchical models

Hierarchical and multilevel models are popular tools in the social and life sciences. A typical motivation for their use is to understand which characteristics are shared among individuals within a population as well as which ones vary, and to what extent. In the frequentist setting, these models are usually termed "mixed-effects" or "fixed- and random-effects" models.

In insurance we face similar challenges: we want to learn as much as possible at a credible "population" level and make adjustments for individual cohorts or policyholders. The Bühlmann-Straub credibility pricing model (Bühlmann and Straub (1970)) is a special case of a hierarchical model.

Hierarchical models have been proposed in the reserving literature previously, for example by (Antonio et al. (2006), Guszcza (2008), Zhang, Dukic, and Guszcza (2012), Morris (2016)).

When it comes to claims reserving, we typically consider the aspects of the data-generating process that we believe to be the same across a dimension and those that will vary "randomly" for the purpose of the model.

It is generally assumed that the loss emergence pattern of claims is similar across accident years, while aggregate loss amounts themselves vary given the "random" nature of loss event occurrence and severity.

For example, the standard chain-ladder method derives a single loss emergence pattern from a claims triangle. The loss development factors are applied to the most recent cumulative claims positions to provide ultimate loss forecasts by accident year. However, the latest cumulative claims positions are the result of a random process for which volatility tends to dominate in earlier development periods (i.e., younger accident years), leading to highly sensitive projections from the chain-ladder approach. This issue is often addressed by using the Bornhuetter-Ferguson method (Bornhuetter and Ferguson (1972)), which incorporates prior information on expected loss ratios and uses the loss emergence expectation as a credibility weight for the chain-ladder forecast.

Hierarchical compartmental models provide a flexible framework to simultaneously model the fixed and random components of the claim development process.

## 4.2 Specifying a hierarchy

The previous section presented models that we can extend to be hierarchical. For example, we could assume that the pattern of loss emergence is the same across accident years and that expected loss ratios vary "randomly" by accident year $i$ around a central value $ELR_c$:

$$\begin{aligned}
\ell_{ij} &\sim \text{Lognormal}(\eta(t_j; \theta, ELR[i]), \sigma) \\
\eta(t; \theta, ELR[i]) &= \log(ELR[i] \cdot (G(t_j; \theta) - G(t_{j-1}; \theta)) \\
&= \log(ELR[i]) + \log(G(t_j; \theta) - G(t_{j-1}; \theta)) \\
ELR[i] &\sim \text{Lognormal}(\log(ELR_c), \sigma[i]) \\
ELR_c &\sim \text{Lognormal}(\log(0.6), 0.1) \\
\sigma[i] &\sim \text{StudentT}(10, 0, 0.05)^+ \\
\theta &\sim \text{Normal}(0.2, 0.02) \\
\sigma &\sim \text{StudentT}(10, 0, 0.05)^+
\end{aligned}$$

This parametrisation is known as the "centered" approach, whereby individual $ELR$ estimates are distributed around an average or central value. For subsequent models in this document, we replace lines 4-6 above with the following structure:

$$\begin{aligned}
\log(ELR)[i] &= \mu_{ELR} + u[i] \\
u[i] &= \sigma[i] \cdot z[i] \\
\mu_{ELR} &\sim \text{Normal}(\log(0.6), 0.1) \\
\sigma[i] &\sim \text{StudentT}(10, 0, 0.025)^+ \\
z[i] &\sim \text{Normal}(0, 1)
\end{aligned}$$

In this specification, individual effects are estimated around the population as additive perturbations, relating naturally to the "fixed" and "random" effects terminology. However, this is a potential source of confusion in the Bayesian setting, where all parameters are random variables. We therefore opt for the terms "population" and "varying" in lieu of "fixed" and "random" to avoid such confusion.

The second "noncentered" parametrisation is the default approach in brms for hierarchical models because it often improves convergence, so we adopt it for all hierarchical models fitted in this paper.

## 4.3 Regularization

Hierarchical models provide an adaptive regularization framework in which the model learns how much weight subgroups of data should get, which can help to reduce overfitting. This is effectively a credibility weighting technique.

Setting a small value for $\sigma[i]$ above ensures that sparse data (e.g., for the most recent accident year) has limited influence on $ELR[i]$. In this scenario, our estimate of $\log(ELR[i])$ will "shrink" more heavily towards $\mu_{ELR}$.

Regularization allows us to estimate the parameters of more complex and thus more flexible models with greater stability and confidence. For example, as we noted earlier, the multistage model (2.4) collapses into the simpler model (2.2) with $k_e = 1$ and $k_{p2} = 0$. We can therefore use the more complex model with priors centered on $1$ and $0$ respectively, to allow flexibility, but only where the data provide a credible signal away from our prior beliefs. In this sense, we can estimate parameters for models which would be considered "overparametrised" in a traditional maximum likelihood setting.

## 4.4 Market Cycles

For the compartmental models introduced in Section 2, hierarchical techniques allow us to estimate "random" variation in reported loss ratios and reserve robustness factors across accident years.

However, changes in the macroeconomic environment, as well as internal changes to pricing strategy, claims settlement processes, and teams, can also impact underwriting and reserving performance in a more systematic manner

Where information relating to such changes exists, we can use it in our modeling to provide more informative priors for the reported loss ratios and reserve robustness factors by accident year.

One approach is to on-level the parameters across years. Suppose we have data on historical cycles in the form of indices, with $RLM_i$ describing reported loss ratio multipliers and $RRM_i$ describing reserve robustness change multipliers on a base accident year.

Sources for the reported loss ratio multipliers could be risk-adjusted rate changes or planning or pricing loss ratios, while the reserve robustness multipliers could be aligned with internal claims metrics.

This data (or judgement, or both) can be used to derive prior parameters $RLR[i]$ and $RRF[i]$ by accident year as follows:

$$\begin{aligned}
RLR[i] &= RLR_{base} \cdot RLM_i^{\lambda_{RLR}} \\
RRF[i] &= RRF_{base} \cdot RRM_i^{\lambda_{RRF}}
\end{aligned}$$

For each accident year, we specify parameter priors as the product of a base parameter (e.g., the expected loss ratio for the oldest year) and an index value for that year. We also introduce additional parameters $\lambda_{RLR}, \lambda_{RRF}$ to describe the extent to which expected loss ratios correlate with the indices.

On a log scale this implies a simple additive linear relationship:

$$\begin{aligned}
\mu_{RLR}[i] &= \mu_{RLR} + \lambda_{RLR}\log(RLM_i) \\
\mu_{RRF}[i] &= \mu_{RRF} + \lambda_{RRF}\log(RRM_i)
\end{aligned}$$

For interpretability, the reported loss ratio and reserve robustness multiplier should be set to 1 for the base accident year. Under this assumption it may be preferable to set prior assumptions for $\lambda$ close to 1 also, provided the indices are considered reliable. Furthermore, the credibility parameters $\lambda$ could be allowed to vary by accident year.

A weakness of this approach is that any index uncertainty observed or estimated in earlier years does not propagate into more recent years. Additionally, the $\lambda$ parameters have minimal influence for years where the indices are close to 1. Although this allows us to set loss ratio priors for each year individually, we could instead adopt time series submodels for the $RLR$ and $RRF$ parameters to address these limitations.

The next section illustrates how to build a hierarchical compartmental model for a single claims triangle. To keep the example compact, we will touch on market cycles but not model them directly. However, the case study in Section 5 will take market cycles into account explicitly. The corresponding R code is presented in the appendix.

## 4.5 Single-triangle hierarchical compartmental model

This example uses the classic "GenIns" paid triangle (Taylor and Ashe (1983)) from the ChainLadder package (Gesmann et al. (2020)). The triangle has been used in many reserving papers, including (Mack (1993), Clark (2003), Guszcza (2008)). We also use the premium information given in (Clark (2003)) for our analysis (see Table 4.1).

| AY | Premium | Dev 1 | Dev 2 | Dev 3 | Dev 4 | Dev 5 | Dev 6 | Dev 7 | Dev 8 | Dev 9 | Dev 10 |
|------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| 1991 | 10,000 | 358 | 1,125 | 1,735 | 2,218 | 2,746 | 3,320 | 3,466 | 3,606 | 3,834 | 3,901 |
| 1992 | 10,400 | 352 | 1,236 | 2,170 | 3,353 | 3,799 | 4,120 | 4,648 | 4,914 | 5,339 |  |
| 1993 | 10,800 | 291 | 1,292 | 2,219 | 3,235 | 3,986 | 4,133 | 4,629 | 4,909 |  |  |
| 1994 | 11,200 | 311 | 1,419 | 2,195 | 3,757 | 4,030 | 4,382 | 4,588 |  |  |  |
| 1995 | 11,600 | 443 | 1,136 | 2,128 | 2,898 | 3,403 | 3,873 |  |  |  |  |
| 1996 | 12,000 | 396 | 1,333 | 2,181 | 2,986 | 3,692 |  |  |  |  |  |
| 1997 | 12,400 | 441 | 1,288 | 2,420 | 3,483 |  |  |  |  |  |  |
| 1998 | 12,800 | 359 | 1,421 | 2,864 |  |  |  |  |  |  |  |
| 1999 | 13,200 | 377 | 1,363 |  |  |  |  |  |  |  |  |
| 2000 | 13,600 | 344 |  |  |  |  |  |  |  |  |  |

Table 4.1: Premiums and cumulative paid claims triangle, with values shown in thousands

The incremental data in Figure 4.1 exhibits substantial volatility, in both quantum and development behavior. Some of the variance seen in the cumulative loss ratio development could be attributed to risk-adjusted rate changes across accident years.

![Example triangle of incremental and cumulative paid loss ratio development by accident year](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/plotClarkGC-1.png)

Figure 4.1: Example triangle of incremental and cumulative paid loss ratio development by accident year

With a single payment triangle we are still able to use a hierarchical compartmental model, such as (2.4), to model paid loss ratio loss emergence. This is similar to how we would with a hierarchical growth curve model; however, we will not be able to make inferences about case reserve robustness.

We allow all compartmental model parameters to vary by accident year and again use the nlf function to transform parameters from $\text{Normal}(0,1)$ into lognormal priors:

```r
frml <- bf(incr_lr ~ eta,
           nlf(eta ~ log(ELR * lossemergence(dev, 1.0, ke, dr, kp1, kp2))),
           nlf(ke ~ exp(oke * 0.5)),
           nlf(dr ~ 1 + 0.1 * exp(odr * 0.5)),
           nlf(kp1 ~ 0.5 * exp(okp1 * 0.5)),
           nlf(kp2 ~ 0.1 * exp(okp2 * 0.5)),
           ELR ~ 1 + (1 | AY),
           oke  ~ 1 + (1 | AY), odr ~ 1 + (1 | AY),
           okp1 ~ 1 + (1 | AY), okp2 ~ 1 + (1 | AY),
           nl = TRUE)
```

We set prior parameter distributions similar to those in the previous section and add priors for the Gaussian perturbation terms of the varying effects. The standard deviations for these are set to narrow Student's t-distributions as regularization to prevent overfitting.

```r
mypriors <- c(prior(inv_gamma(4, 2), nlpar = "ELR", lb=0),
              prior(normal(0, 1), nlpar = "oke"),
              prior(normal(0, 1), nlpar = "odr"),
              prior(normal(0, 1), nlpar = "okp1"),
              prior(normal(0, 1), nlpar = "okp2"),
              prior(student_t(10, 0, 0.1), class = "sd", nlpar = "ELR"),
              prior(student_t(10, 0, 0.1), class = "sd", nlpar = "oke"),
              prior(student_t(10, 0, 0.1), class = "sd", nlpar = "odr"),
              prior(student_t(10, 0, 0.1), class = "sd", nlpar = "okp1"),
              prior(student_t(10, 0, 0.1), class = "sd", nlpar = "okp2"),
              prior(student_t(10, 0, 1), class = "sigma"))
```

Now we can estimate the posterior distributions for all of the parameters in our model:

```r
fit_loss <- brm(frml, prior = mypriors,
                data = lossDat, family = lognormal(), seed = 12345,
                stanvars = stanvar(scode = myFuns, block = "functions"),
                backend = "cmdstan",
                file="models/section_4/GenInsIncModelLog")
```

The model run does not report any obvious warnings. Diagnostics such as $\hat{R}$ and effective sample size look good, so we move on to reviewing the outputs. The case study in the next section will cover model review and validation in more detail; hence we keep it brief here.

We note that the population $k_e$ and $k_{p2}$ from the extended compartmental model are identified with 95% posterior credible intervals that scarcely contain 1 and 0, respectively, indicating possible support for this model structure (see Table 4.2).

| Parameter | Estimate | Est.Error | Q2.5 | Q97.5 |
|-----------|----------|-----------|-------|--------|
| ELR | 0.490 | 0.035 | 0.426 | 0.563 |
| ke | 0.663 | 0.175 | 0.381 | 1.052 |
| dr | 1.147 | 0.079 | 1.046 | 1.343 |
| kp1 | 0.423 | 0.131 | 0.249 | 0.749 |
| kp2 | 0.113 | 0.060 | 0.037 | 0.262 |

Table 4.2: Population-level estimates

Notwithstanding data volatility, the model appears reasonably well behaved against the historical data (see Figure 4.2).

![Posterior predictive distribution for each accident and development year, showing the predicted means and 95 percent predictive intervals](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/plotMarginalEffects-1.png)

Figure 4.2: Posterior predictive distribution for each accident and development year, showing the predicted means and 95 percent predictive intervals

Figure 4.3 plots 50% and 90% posterior credible intervals for each accident year's estimated deviation from the population $ELR$ on the log scale. This allows us to inspect how variable the model estimates performance to be across accident years and the levels of uncertainty for each year.

![Posterior Credible Intervals from HMC draws of ELR by accident year, i.e. the expected performance variance across all years](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/MCMCinterval-1.png)

Figure 4.3: Posterior Credible Intervals from HMC draws of ELR by accident year, i.e. the expected performance variance across all years

Observe that all credible intervals contain 0, i.e. we cannot be sure that any one year's $ELR$ is different from the population average. However, there is some evidence of deviation across years, which, as observed in the cumulative paid developments, could be attributed to historical rate changes.

In addition, we compare the posterior mean loss emergence pattern by accident year against the Cape Cod method outlined in (Clark (2003)) with maximum age set to 20, as implemented in (Gesmann et al. (2020)). Figure 4.4 a) shows that the selected compartmental model's loss emergence patterns do not vary much across accident years due to our relatively tight priors, mirroring initially the Weibull curve and, for later development years, the loglogistic growth curve.

![Comparing hierarchical growth curves with different prior parameter distributions](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/LossEmergence2-1.png)

Figure 4.4: Comparing hierarchical growth curves with different prior parameter distributions

If we increase the uncertainty of the hyperprior parameter distributions from $\text{StudentT}(10, 0, 0.1)$ to $\text{StudentT}(10, 0, 1)$, then the individual accident year development data gets more weight, and estimated loss emergence patterns start to exhibit some variance across accident years, shown Figure 4.4, panel (b).

## 4.6 Expected versus ultimate loss

We parameterized the above model in terms of $ELRs$ (expected loss ratios) rather than $ULRs$ (ultimate loss ratios). This was deliberate, since our model aims to estimate the latent parameter of the underlying development process.

For a given accident year, the $ELR$ parameter describes the underlying expected loss ratio in the statistical process that generated the year's loss emergence. Put another way, if an accident year were to play out repeatedly and infinitely from scratch, then the ELR is an estimate of the average ultimate loss ratio over all possible scenarios.

In reality, of course, we can only observe a single realization of the claims development for each accident year. This realization will generate the $ULR$ (the actual ultimate loss ratio), which derives from the sum of payments to date plus estimated future incremental payments.

Hence, the ultimate loss (or ultimate loss ratio) is anchored to the latest cumulative payment, while the expected loss ratio treats the payments to date as one random series of payment realizations.

| AY | ELR (%) | ELR Std | ULR (%) | ULR Std |
|------|---------|---------|---------|---------|
| 1991 | 46.6 | 4.5 | 43.2 | 1.4 |
| 1992 | 52.7 | 5.6 | 58.6 | 2.2 |
| 1993 | 49.8 | 4.8 | 53.6 | 2.5 |
| 1994 | 47.3 | 4.7 | 50.3 | 2.9 |
| 1995 | 49.0 | 4.9 | 47.1 | 3.7 |
| 1996 | 49.5 | 5.3 | 48.9 | 4.5 |
| 1997 | 50.4 | 5.8 | 52.1 | 5.6 |
| 1998 | 50.3 | 6.1 | 53.9 | 6.7 |
| 1999 | 48.9 | 6.1 | 49.5 | 8.0 |
| 2000 | 48.3 | 6.4 | 48.7 | 8.5 |

Table 4.3: Table of expected and ultimate loss (to age 20) with respective estimated standard errors of second model with wider hyperprior parameters

The estimated $ULR$ standard errors are driven only by the estimated errors for future payments; hence they increase significantly across newer accident years as ultimate uncertainty increases. The estimated errors around the $ELR$ are more stable, as are the estimated mean $ELR$ values. The differences between expected and ultimate loss ratios demonstrate the point made above: the $ULR$ is anchored to the latest payment value and is therefore influenced by that particular random series of payments to date, as shown in Table 4.3.

Finally, we can review the distribution of the posterior predictive reserve, derived as the sum of extrapolated future payments to development year 20, or calendar year 2010 (see Figure 4.5).

The reserve distribution is not an add-on, but part of the model output.

![Histogram of posterior predictive payments up to development year 20 of second second model with wider hyperprior parameters. Mean model reserve highlighted in blue.](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/GenInsReserve-1.png)

Figure 4.5: Histogram of posterior predictive payments up to development year 20 of second second model with wider hyperprior parameters. Mean model reserve highlighted in blue.

Note that reserve uncertainty can be analyzed in more detail, since the model returns the full posterior distribution for all parameters and hence, predictions, by accident year and development year.

## 4.7 Correlations across effects

As a further step we could test for correlations between parameters across accident years. For example, we might expect that lower loss ratios correlate with a faster payment speed.

Assuming a centered multivariate Gaussian distribution for the varying effects with an LKJ prior (Lewandowski, Kurowicka, and Joe (2009)) for the correlations becomes somewhat cumbersome to write down in mathematical notation. However, if we define the growth curve to include $ELR[i]$ as a parameter:

$$\tilde{G}(t; ELR[i], k_e[i], d_r[i], k_{p1}[i], k_{p2}[i]) := ELR[i] \cdot G(t; k_e[i], d_r[i], k_{p1}[i], k_{p2}[i]),$$

then the notation for the varying-effects correlated model can be written as follows:

$$\begin{aligned}
\ell_{ij} &\sim \text{Lognormal}(\eta_i(t), \sigma) \\
\eta_i(t) &= \log(\tilde{G}(t; \Theta)) \\
\Theta &= \mu_\Theta + u_\Theta \\
\mu_\Theta &= (\mu_{ELR}, \mu_{k_e}, \mu_{d_r}, \mu_{k_{p1}}, \mu_{k_{p2}}) \\
u_\Theta &= (u_{ELR}, u_{k_e}, u_{d_r}, u_{k_{p1}}, u_{k_{p2}})
\end{aligned}$$

Priors:
$$\begin{aligned}
\sigma &\sim \text{StudentT}(3, 0, 0.1)^+ \\
\mu_{ELR} &\sim \text{InvGamma}(4, 2) \\
\mu_{k_e}, \mu_{d_r} &\sim \text{Lognormal}(\log(3), 0.2) \\
\mu_{k_{p1}} &\sim \text{Lognormal}(\log(0.5), 0.1) \\
\mu_{k_{p2}} &\sim \text{Normal}(0.2, 0.5) \\
u_\Theta &\sim \text{MultivariateNormal}(0, \Sigma) \\
\Sigma &= D\Omega D \\
D &= \text{Diag}(\sigma_{ELR}, \sigma_{k_e}, \sigma_{d_r}, \sigma_{k_{p1}}, \sigma_{k_{p2}}) \\
(\sigma_{ELR}, \sigma_{k_e}, \sigma_{d_r}, \sigma_{k_{p1}}, \sigma_{k_{p2}}) &\sim \text{StudentT}(3, 0, 0.1)^+ \\
\Omega &\sim \text{LkjCorr}(1)
\end{aligned} \tag{4.1}$$

Implementing this correlation structure in brms is straightforward; we simply add a unique character string to each varying effect term:

```r
frml <- bf(incr_lr ~ log(ELR * lossemergence(dev, 1.0, ke, dr, kp1, kp2)),
          ELR ~ 1 + (1 | ID | AY),
          ke  ~ 1 + (1 | ID | AY), dr ~ 1 + (1 | ID | AY),
          kp1 ~ 1 + (1 | ID | AY), kp2 ~ 1 + (1 | ID | AY),
          nl = TRUE)
```

This notation naturally extends further. Suppose we have development data by company and accident year, as in (Zhang, Dukic, and Guszcza (2012)), and would like to model a structure which allows $ELR$ to vary by accident year and company. With $k_{er}$ and $k_p$ constant by accident year but varying by company, and correlating $ELR$, $k_{er}$ and $k_p$ by company we can write the following:

```r
(ELR ~ 1 + (1 | ID | company) + (1 | AY:company),
 ker ~ 1 + (1 | ID | company),
 kp  ~ 1 + (1 | ID | company))
```

An implementation of a similar multicompany model on a cumulative loss ratio basis in brms is given in (Gesmann (2018)).

These examples illustrate how brms provides a powerful and rich framework to build complex models using intuitive model notation. For more detail, please refer to the various brms and RStan vignettes. Note the underlying Stan code can be extracted from any brms object using the stancode function.

### References

Antonio, Katrien, Jan Beirlant, Tom Hoedemakers, and Robert Verlaak. 2006. "Lognormal Mixed Models for Reported Claims Reserves." _North American Actuarial Journal_ 10 (1): 30–48.

Bornhuetter, R.L., and R. E. Ferguson. 1972. "The Actuary and IBNR." _Proceedings of the Casualty Actuarial Society_ LIX: 181–95.

Bühlmann, H., and E. Straub. 1970. "Glaubgwürdigkeit Für Schadensätze." _Bulletin of the Swiss Association of Actuaries_ 70: 111–33.

Clark, David R. 2003. _LDF Curve-Fitting and Stochastic Reserving: A Maximum Likelihood Approach_. Casualty Actuarial Society; http://www.casact.org/pubs/forum/03fforum/03ff041.pdf.

Gesmann, Markus. 2018. "Hierarchical Loss Reserving with Growth Curves Using Brms." https://magesblog.com/post/2018-07-15-hierarchical-loss-reserving-with-growth-cruves-using-brms/.

Gesmann, Markus, Dan Murphy, Wayne Zhang, Alessandro Carrato, Mario Wüthrich, and Fabio Concina. 2020. _ChainLadder: Statistical Methods and Models for Claims Reserving in General Insurance_. https://github.com/mages/ChainLadder.

Guszcza, James. 2008. "Hierarchical Growth Curve Models for Loss Reserving." In _Casualty Actuarial Society e-Forum, Fall 2008_, 146–73. https://www.casact.org/pubs/forum/08fforum/7Guszcza.pdf.

Lewandowski, Daniel, Dorota Kurowicka, and Harry Joe. 2009. "Generating Random Correlation Matrices Based on Vines and Extended Onion Method." _Journal of Multivariate Analysis_ 100 (9): 1989–2001. https://doi.org/10.1016/j.jmva.2009.04.008.

Mack, Thomas. 1993. "Distribution-Free Calculation of the Standard Error of Chain Ladder Reserve Estimates." _ASTIN Bulletin_ 23: 213–25.

Morris, Jake. 2016. _Hierarchical Compartmental Models for Loss Reserving_. Casualty Actuarial Society Summer E-Forum; https://www.casact.org/pubs/forum/16sforum/Morris.pdf.

Taylor, Greg C, and FR Ashe. 1983. "Second Moments of Estimates of Outstanding Claims." _Journal of Econometrics_ 23 (1): 37–61.

Zhang, Yanwei, Vanja Dukic, and James Guszcza. 2012. "A Bayesian Nonlinear Model for Forecasting Insurance Loss Payments." _Journal of the Royal Statistical Society, Series A_ 175: 637–56.