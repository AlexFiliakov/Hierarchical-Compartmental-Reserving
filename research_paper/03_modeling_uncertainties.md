# 3 Modeling parameter and process uncertainties

In the previous section we developed an understanding of how to model the average behavior of the claims development process using compartmental models. In this section, we start to build statistical models around a central statistic such as the mean or median.

We will not model any data here; instead, the focus is on selecting distributions for the observation scale (the "process") and priors for the system parameters. The aim is to create a model that can simulate data that shares key characteristics of real data. This will lead to a discussion on modeling cumulative versus incremental paid claims.

We demonstrate how these models can be implemented in Stan (Stan Development Team (2019)) using `brms` (Bürkner (2017)) as an interface from R (R Core Team (2019)) to generate prior predictive output.

## 3.1 Data-generating ('process') distribution

To model the data-generating process for our observations, $y_j$, we have to consider the likely distribution of our data ($D$) and how the process can be expressed in terms of parameters. In simple models, we often distinguish between parameters that are direct functions of variables in our data ($\Theta$) and family specific ($\Phi$), which are either fixed or vary indirectly with respect to $\Theta$ in line with specific distributional assumptions. Examples of the latter include the standard deviation $\sigma$ in Gaussian models or the shape $\alpha$ in gamma models.

The generic form of a univariate data generating process for repeated measures data (such as claims development) can be written as follows:

$$y_j \sim D(f(t_j, \Theta), \Phi)$$

Note that in more complex models we can estimate specific relationships between $\Phi$ and data features by introducing additional parameters.

It can be helpful to think about how the variability in $y_j$ is related to changes in the mean or median. In ordinary linear regression, where the process is assumed to follow a normal distribution (or equivalently, to have a Gaussian error term), a constant variance is typically assumed:

$$y_j \sim \text{Normal}(\mu(t_j), \sigma)$$

In the claims reserving setting it is often assumed that volatility changes with the mean. A multiplicative or overdispersed relationship is usually considered.

Given that claims are typically right skewed and that larger claims tend to exhibit larger variation, the lognormal or gamma distributions are often good starting points. Other popular choices are the negative binomial (claim counts) and Tweedie (pure premium) distributions. However, for some problems, standard distributions will not appropriately characterise the level of zero-inflation or extreme losses observed without additional model structure.

For the lognormal distribution, a constant change assumption on the log scale translates to a constant coefficient of variation (CoV) on the original scale ($\text{CoV} = \sqrt{\exp(\sigma^2) - 1}$):

$$y_j \sim \text{Lognormal}(\mu(t_j), \sigma)$$

It can be helpful to model variables on a similar scale so that they have similar orders of magnitude. This can improve sampling efficiency and, in the case of the target variable, makes it easier apply the same model to different data sets. For this reason, we advise modeling loss ratios instead of loss amounts in the first instance. However, we also note that this approach will have an effect on the implicit volume weighting within the optimization routine for constant CoV distributions, and on occasion it may be preferable to target claim amounts.

The choice of the process distribution should be carefully considered, and the modeler should be able to articulate the selection criteria.

## 3.2 Prior parameter distributions

The concept of analyses beginning with "prior" assumptions, which are updated with data, is fundamental to Bayesian inference. In the claims reserving setting, the ability to set prior distributional assumptions for the claims process parameters also gives the experienced practitioner an opportunity to incorporate his or her knowledge into an analysis independently of the data.

An expert may have a view on the required shape of a parameter distribution and the level of uncertainty. Figure 3.1 (Bååth (2012)) provides an overview of typical distributions. In order to select a sensible distribution, it can be helpful to consider the following questions:

- Is the data/parameter continuous or discrete?
- Is the data/parameter symmetric or asymmetric?
- Is the data/parameter clustered around a central value?
- How prevalent are outliers?
- Are the outliers positive or negative?

![Schematic diagram of popular distributions and their parameters.](https://compartmentalmodels.gitlab.io/researchpaper/img/all_dists.png)

Figure 3.1: Schematic diagram of popular distributions and their parameters.

There is no concept of a "prior" in frequentist procedures, and hence Bayesian approaches can offer greater flexibility when such prior knowledge exists. However, note that priors are starting points only, and the more data that are available, the less sensitive posterior inferences will be to those starting points.

## 3.3 Cumulative versus incremental data

Since we intend to model the full aggregated claims distribution at each development time, we have to carefully consider the impact the process variance assumption has on model behavior. This is particularly true for paid claims. Actual payments are incremental by nature, but we have the option to model cumulative payments. Does our choice matter?

Many traditional reserving methods (including the chain-ladder technique) take cumulative claims triangles as an input. Plotting cumulative claims development allows the actuary to quickly understand key data features by eye and identify the appropriateness of the selected projection technique.

In compartmental reserving models we estimate cumulative paid claims in the final compartment – a scaled ($RRF$) and delayed ($k_p$) version of the integrated outstanding claims –so it is also natural to visualize cumulative paid claims development. However, if we assume a constant (e.g., lognormal) CoV process distribution and model cumulative claims, this would imply more volatile paid claims over development time as payments cumulate. As a result, changes from one development period to the next would become more volatile. This feature is in direct contradiction to our intuition and the mean compartmental model solution, which expects less movement in the aggregate cumulative paid claims as fewer claims are outstanding.

To illustrate this concept and get us started with Bayesian model notation, we consider a simple growth curve model for cumulative paid loss ratio development.

Let's assume the loss ratio data-generating process can be modeled using a lognormal distribution, with the median loss ratio over time following a simple exponential growth curve. The loss ratio ($\ell_j$) at any given development time ($t_j$) is modeled as the product of an expected loss ratio ($ELR$) and loss emergence pattern $G(t; \theta)$:

$$\begin{aligned}
\ell_j &\sim \text{Lognormal}(\eta(t_j; \theta, ELR), \sigma) \\
\eta(t; \theta, ELR) &= \log(ELR \cdot G(t; \theta)) \\
G(t; \theta) &= 1 - e^{-\theta t} \\
ELR &\sim \text{InvGamma}(4, 2) \\
\theta &\sim \text{Normal}(0.2, 0.02) \\
\sigma &\sim \text{StudentT}(10, 0.1, 0.1)^+
\end{aligned}$$

Note that we specify prior distributions for the parameters $ELR$, $\theta$, and $\sigma$ to express our uncertainty in these quantities. We assume that the expected loss ratio ($ELR$) follows an inverse gamma distribution to ensure positivity, but also allow for potential larger losses and hence poorer performance.

The parameter $\theta$ describes loss emergence speed, with $\ln(2)/\theta$ being the expected halfway-time of ultimate development. We set a Gaussian prior for this parameter, with a mean of 0.2 and standard deviation of 0.02, which implies that we expect 50% development of claims after around 3.5 years, but perhaps this occurs a month earlier or later.

The process uncertainty ($\sigma$) has to be positive, so we assume a Student-t distribution left-truncated at 0. Figure 3.2 illustrates the prior parameter distributions.

![Density plots of prior parameter distributions](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/showPriorDistr1-1.png)

Figure 3.2: Density plots of prior parameter distributions

Sampling from this model produces payment patterns with many negative increments in later development periods, as depicted in Figure 3.3.

![Spaghetti plot of 100 simulated cumulative loss ratios](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/SampleCumulative-1.png)

Figure 3.3: Spaghetti plot of 100 simulated cumulative loss ratios

The reason for this behavior is the lognormal constant CoV, $\sigma$. As the mean loss ratio increases with development time, volatility increases as well, and there is no constraint in the model for the lognormal realizations to be increasing by development time.

However, this is not what we typically observe in development data. To account for this, (Meyers (2015)) imposes a monotone decreasing constraint on the $\sigma_j$ parameter with respect to development time, while (Zhang, Dukic, and Guszcza (2012)) and (Morris (2016)) include a first-order autoregressive error process.

Many others, including (Zehnwirth and Barnett (2000), Clark (2003)), model incremental payments, for example as follows:

$$\eta(t_j; \theta, ELR) = \log(ELR \cdot [G(t_j; \theta) - G(t_{j-1}; \theta)])$$

Modeling the incremental median payments with a lognormal distribution and a constant CoV is not only straightforward in the brms package in R, as shown in the code below, but the resultant simulations from the model appear more closely aligned to development data observed in practice, as shown in Figure 3.4.

```r
myFun <- "
real incrGrowth(real t, real tfreq, real theta){
  real incrgrowth;
  incrgrowth = (1 - exp(-t * theta));
  if(t > tfreq){
    incrgrowth = incrgrowth - (1 - exp(-(t - tfreq) * theta));
  }
  return(incrgrowth);
}
"
prior_lognorm <- brm(
  bf(incr_lr ~ log(ELR * incrGrowth(t, 1.0, theta)),
     ELR ~ 1, theta ~ 1, nl=TRUE),
  prior = c(prior(inv_gamma(4, 2), nlpar = "ELR", lb=0),
            prior(normal(0.2, 0.02), nlpar = "theta", lb=0),
            prior(student_t(10, 0.1, 0.1), class = "sigma")),
  data = dat, file = "models/section_3/prior_lognorm",
  backend = "cmdstan",
  stanvars = stanvar(scode = myFun, block = "functions"),
  family = brmsfamily("lognormal"),
  sample_prior = "only")
```

![Simulations of incremental claims payments and cumulative aggregation across development period](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/SampleIncremental-1.png)

Figure 3.4: Simulations of incremental claims payments and cumulative aggregation across development period

Additional factors which lead us to favour the use of incremental data include the following:

- Missing or corrupted data for some development periods can be problematic when we require cumulative data from the underlying incremental cash flows. Manual interpolation techniques can be used ahead of modeling, but a parametric growth curve applied to incremental data will deal with missing data as part of the modeling process.

- Changes in underlying processes (claims handling or inflation) causing effects in the calendar period dimension can be masked in cumulative data and are easier to identify and model using incremental data.

- Predictions of future payments are put on an additive scale, rather than a multiplicative scale, which avoids ad hoc anchoring of future claims projections to the latest cumulative data point.

## 3.4 Prior predictive examples

In this section, we provide three more examples of simulation models with different process distributions. These models are generative insofar as they are intended to emulate the data-generating process. However, their parameters are set manually as priors rather than estimated from data, so we term them "prior predictive" models.

The prior predictive distribution ($p(y)$) is also known as the marginal distribution of the data. It is the integral of the likelihood function with respect to the prior distribution

$$p(y) = \int p(y|\theta)p(\theta)d\theta$$

and is not conditional on observed data.

Clark demonstrates (Clark (2003)) how an overdispersed Poisson model can be fitted using maximum likelihood, and Guszcza (Guszcza (2008)) illustrates the use of a Gaussian model with constant coefficient of variation.

Below, we showcase how these ideas can be implemented in Stan with brms.

At the end of the section we outline a prior predictive model for compartmental model (1) in the previous section.

### 3.4.1 Negative binomial process distribution

An overdispersed Poisson process distribution is assumed in (Clark (2003)), but here we will use a negative binomial distribution to model overdispersion generatively. This is also a standard family distribution in brms.

The negative binomial distribution can be expressed as a Poisson($\mu$) distribution where $\mu$ is itself a random variable, coming from a gamma distribution with shape $\alpha = r$ and rate $\beta = (1-p)/p$:

$$\begin{aligned}
\mu &\sim \text{Gamma}(r, (1-p)/p) \\
y &\sim \text{Poisson}(\mu)
\end{aligned}$$

Alternatively, we can specify a negative binomial distribution with mean parameter $\mu$ and dispersion parameter $\phi$:

$$\begin{aligned}
y &\sim \text{NegativeBinomial}(\mu, \phi) \\
E(y) &= \mu \\
\text{Var}(y) &= \mu + \mu^2/\phi
\end{aligned}$$

The support for the negative binomial distribution is $\mathbb{N}$, and therefore we model dollar-rounded loss amounts instead of loss ratios.

A growth curve model can be written as follows, with a log-link for the mean and shape parameters:

$$\begin{aligned}
L_j &\sim \text{NegativeBinomial}(\mu(t_j; \theta, \Pi, ELR), \phi) \\
\mu(t_j; \theta, ELR) &= \log(\Pi \cdot ELR \cdot (G(t_j; \theta) - G(t_{j-1}; \theta))) \\
\theta &\sim \text{Normal}(0.2, 0.02)^+ \\
ELR &\sim \text{InvGamma}(4, 2) \\
\phi &\sim \text{StudentT}(10, 0, \log(50))^+
\end{aligned}$$

This is straightforward to specify in brms:

```r
prior_negbin <- brm(
  bf(incr  ~ log(premium * ELR * incrGrowth(t, 1.0, theta)),
     ELR ~ 1, theta ~ 1, nl = TRUE),
  prior = c(prior(inv_gamma(4, 2), nlpar = "ELR"),
            prior(normal(0.2, 0.02), nlpar = "theta", lb=0),
            prior(student_t(10, 0, log(50)), class = "shape")),
 data = dat, family = negbinomial(link = "log"),
 backend = "cmdstan",
 stanvars = stanvar(scode = myFun, block = "functions"),
 file="models/section_3/prior_negbin", sample_prior = "only")
```

![Prior predictive simulations of 100 losses with a negative binomial process distribution assumption](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/ClarkModelPriorPlot-1.png)

Figure 3.5: Prior predictive simulations of 100 losses with a negative binomial process distribution assumption

### 3.4.2 Gaussian process distribution with constant CoV

In (Guszcza (2008)) proposes a Gaussian model with constant constant CoV to force the standard deviation to scale with the mean.

We can re-express our loss ratio growth curve model from earlier as

$$\begin{aligned}
\ell_j &\sim \text{Normal}(\eta(t_j; \Theta, ELR), \sigma_j) \\
\sigma_j &= \sigma\sqrt{\eta_j} \\
\eta(t_j; \theta, ELR) &= ELR \cdot (G(t_j; \Theta) - G(t_{j-1}; \Theta)) \\
\theta &\sim \text{Normal}(0.2, 0.02)^+ \\
ELR &\sim \text{InvGamma}(4, 2) \\
\sigma &\sim \text{StudentT}(10, 0, 0.1)^+,
\end{aligned}$$

which we can specify in brms once more (note the use of the `nlf` function here, which maps variables to nonlinear functions):

```r
prior_gaussian <- brm(
  bf(incr_lr ~ eta,
     nlf(eta ~ ELR * incrGrowth(t, 1.0, theta)),
     nlf(sigma ~ tau * sqrt(eta)),
     ELR ~ 1, theta ~ 1, tau ~ 1, nl = TRUE),
  data = dat, family = brmsfamily("gaussian", link_sigma = "identity"),
  prior = c(prior(inv_gamma(4, 2), nlpar = "ELR"),
            prior(normal(0.2, 0.02), nlpar = "theta", lb=0),
            prior(student_t(10, 0, 0.1), nlpar = "tau", lb = 0)),
  backend = "cmdstan",
  stanvars = stanvar(scode = myFun, block = "functions"),
  file = "models/section_3/prior_gaussian",
  sample_prior = "only")
```

![Prior predictive simulations of 100 losses with a Gaussian process distribution assumption](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/GuszczaModelPriorPlot-1.png)

Figure 3.6: Prior predictive simulations of 100 losses with a Gaussian process distribution assumption

### 3.4.3 Compartmental model with lognormal distribution

Finally, we simulate model output for our first compartmental model (2.1).

Compartmental models have a little more complexity than the growth curve models above, and so we have additional considerations for the implementation with brms and Stan:

- How to deal with the multivariate nature of the compartmental model, which is specified to fit paid and outstanding claims data simultaneously
- How to solve the ODEs numerically
- How to ensure that as the number of prior assumptions grows, their initialization values are valid

#### 3.4.3.1 Compartmental model setup

To model paid and outstanding loss ratio data simultaneously, we stack both into a single column and add another column with an indicator variable. This indicator ($\delta$) allows us to switch between the two claim stages and specify different variance levels (with a log link):

$$\begin{aligned}
y_j &\sim \text{Lognormal}(\mu(t_j; \Theta, \delta), \sigma[\delta]) \\
\mu(t_j; \Theta, \delta) &= \log((1-\delta)OS_j + \delta(PD_j - PD_{j-1})) \\
\sigma[\delta] &= \exp((1-\delta)\beta_{OS} + \delta\beta_{PD}) \\
\delta &= \begin{cases}
0 & \text{if } y_j \text{ is outstanding claim} \\
1 & \text{if } y_j \text{ is paid claim}
\end{cases} \\
\Theta &= (k_{er}, k_p, RLR, RRF) \\
dEX_j/dt &= -k_{er} \cdot EX_j \\
dOS_j/dt &= k_{er} \cdot RLR \cdot EX_j - k_p \cdot OS_j \\
dPD_j/dt &= k_p \cdot RRF \cdot OS_j
\end{aligned} \tag{3.1}$$

Some of the more complex compartmental models described in the previous section have no analytical solutions for their ODE systems, forcing us to rely on numerical integration.

Fortunately, the Stan language contains a Runge-Kutta solver. We can write our solver in Stan and pass the code into brms in the same way as we did with the analytical growth curve solution earlier.

The Stan code below shows three functional blocks. The first function defines the ODE system, the second the solver, and the third the application to the data. Note the modeling of incremental paid claims for development periods greater than 1.

```r
myCompFun <- "
// ODE System
vector compartmentmodel(real t, vector y, vector theta) {
  vector[3] dydt;
  // Define ODEs
  dydt[1] = - theta[1] * y[1];
  dydt[2] = theta[1] * theta[3] * y[1] - theta[2] * y[2];
  dydt[3] = theta[2] * theta[4] * y[2];

  return dydt;
  }

//Application to OS and Incremental Paid Data
real claimsprocess(real t, real devfreq, real ker, real kp,
                   real RLR, real RRF, real delta){
    vector[3] y0;
    array[1] vector[3] y;
    vector[4] theta;
    theta[1] = ker;
    theta[2] = kp;
    theta[3] = RLR;
    theta[4] = RRF;
    real out;
    // Set initial values
    y0[1] = 1; // Exposure
    y0[2] = 0; // Outstanding
    y0[3] = 0; // Paid


    y = ode_rk45(compartmentmodel, y0, 0, rep_array(t, 1), theta);
    out = y[1, 2] * (1 - delta) + y[1, 3] * delta;

    if( (delta > 0) && (t > devfreq) ){ // paid greater dev period 1
    // incremental paid
     y = ode_rk45(compartmentmodel, y0, 0, rep_array(t - devfreq, 1), theta);
     out = out - y[1, 3];
    }
    return(out);
}
"
```

At the beginning of the HMC simulation Stan initializes all parameter values randomly between -2 and 2. Although these can be changed by the user, the default initializations can cause issues for parameters that cannot be negative in the model. To avoid setting multiple initial values, it is common practice to define parameters on an unconstrained scale and transform them to the required scale afterwards.

For our example, we will assume all compartmental model parameter priors are lognormally distributed. For the implementation, however, we use standardized Gaussians and transform them to lognormal distributions using the nlf function.

$$\begin{aligned}
RLR &\sim \text{Lognormal}(\log(0.6), 0.1) \\
RRF &\sim \text{Lognormal}(\log(0.95), 0.05) \\
k_{er} &\sim \text{Lognormal}(\log(1.7), 0.02) \\
k_p &\sim \text{Lognormal}(\log(0.5), 0.05)
\end{aligned}$$

We assume Gaussians $\beta_{OS}$ and $\beta_{PD}$, with the volatility for outstanding loss ratios slightly higher than for paid loss ratios:

$$\begin{aligned}
\beta_{OS} &\sim \text{Normal}(0.15, 0.025) \\
\beta_{PD} &\sim \text{Normal}(0.1, 0.02)
\end{aligned}$$

The above implies a lognormal distribution for $\sigma[\delta]$, given the log link.

Now that we have prepared our model, we can simulate from it with brms (below) and review the prior predictive output (3.7).

```r
frml <- bf(
  incr_lr ~ eta,
  nlf(eta ~ log(claimsprocess(t, 1.0, ker, kp, RLR, RRF, delta))),
  nlf(RLR ~ 0.6 * exp(oRLR * 0.1)),
  nlf(RRF ~ 0.95 * exp(oRRF * 0.05)),
  nlf(ker ~ 1.7 * exp(oker * 0.02)),
  nlf(kp ~ 0.5 * exp(okp * 0.05)),
  oRLR ~ 1, oRRF ~ 1, oker ~ 1, okp ~ 1, sigma ~ 0 + deltaf,
  nl = TRUE)
mypriors <- c(prior(normal(0, 1), nlpar = "oRLR"),
              prior(normal(0, 1), nlpar = "oRRF"),
              prior(normal(0, 1), nlpar = "oker"),
              prior(normal(0, 1), nlpar = "okp"),
              prior(normal(0.15, 0.025), class = "b",
                    coef="deltafos", dpar= "sigma"),
              prior(normal(0.1, 0.02), class = "b",
                    coef="deltafpaid", dpar = "sigma"))
prior_compartment_lognorm <- brm(frml, data = dat,
  family = brmsfamily("lognormal", link_sigma = "log"),
  prior = mypriors,
  backend = "cmdstan",
  stanvars = stanvar(scode = myCompFun, block = "functions"),
  file="models/section_3/prior_compartment_lognorm",
  sample_prior = "only")
```

![Prior predictive simulations of 100 outstanding and paid development paths with a lognormal process distribution assumption](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/InitialCompartmentPriorPredictive-1.png)

Figure 3.7: Prior predictive simulations of 100 outstanding and paid development paths with a lognormal process distribution assumption

The prior predictive appear to resemble development data, despite having not used any real data to generate them.

As part of a robust Bayesian work flow, one should next try to fit the model to a sample of the prior predictive distribution to establish whether the model parameters are identifiable (Betancourt (2018)). This is left as an exercise for the reader.

### References

Bååth, Rasmus. 2012. "Kruschke Style Diagrams." https://github.com/rasmusab/distribution_diagrams.

Betancourt, Michael. 2018. "Towards a Principled Bayesian Workflow (RStan)." https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html.

Bürkner, Paul-Christian. 2017. "brms: An R Package for Bayesian Multilevel Models Using Stan." _Journal of Statistical Software_ 80 (1): 1–28. https://doi.org/10.18637/jss.v080.i01.

Clark, David R. 2003. _LDF Curve-Fitting and Stochastic Reserving: A Maximum Likelihood Approach_. Casualty Actuarial Society; http://www.casact.org/pubs/forum/03fforum/03ff041.pdf.

Guszcza, James. 2008. "Hierarchical Growth Curve Models for Loss Reserving." In _Casualty Actuarial Society e-Forum, Fall 2008_, 146–73. https://www.casact.org/pubs/forum/08fforum/7Guszcza.pdf.

Meyers, Glenn. 2015. _Stochastic Loss Reserving Using Bayesian MCMC Models_. CAS Monograph Series. http://www.casact.org/pubs/monographs/papers/01-Meyers.PDF; Casualty Actuarial Society.

Morris, Jake. 2016. _Hierarchical Compartmental Models for Loss Reserving_. Casualty Actuarial Society Summer E-Forum; https://www.casact.org/pubs/forum/16sforum/Morris.pdf.

R Core Team. 2019. _R: A Language and Environment for Statistical Computing_. Vienna, Austria: R Foundation for Statistical Computing; https://www.R-project.org/.

Stan Development Team. 2019. "RStan: The R Interface to Stan." http://mc-stan.org/.

Zehnwirth, Ben, and Glen Barnett. 2000. "Best Estimates for Reserves." _Proceedings of the CAS_ LXXXVII (167).

Zhang, Yanwei, Vanja Dukic, and James Guszcza. 2012. "A Bayesian Nonlinear Model for Forecasting Insurance Loss Payments." _Journal of the Royal Statistical Society, Series A_ 175: 637–56.