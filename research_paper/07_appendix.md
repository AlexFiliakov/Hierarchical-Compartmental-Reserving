# 7 Appendix

The appendix presents the R code used in Sections 4 and 5. The code can be copied and pasted into an R session. At the time of writing R version 4.3.1 (2023-06-16) was used, with brms 2.20.0 and rstan 2.21.8.

```r
library(rstan)
library(brms)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
```

## 7.1 R code from section 4

The "GenIns" triangle is part of the ChainLadder package. The triangle is transformed into a long table format, with premiums, incremental paid and loss ratio columns added:

```r
library(ChainLadder)
library(data.table)
data(GenIns)
lossDat <- data.table(
  AY = rep(1991:2000, 10),
  dev = sort(rep(1:10, 10)),
  premium = rep(10000000+400000*0:9, 10),
  cum_loss = as.vector(GenIns),
  incr_loss = as.vector(cum2incr(GenIns))
)[order(AY, dev),
  `:=`(cum_lr = cum_loss/premium,
       incr_lr = incr_loss/premium)]
```

The next code chunk shows how the loss emergence pattern is modelled using differential equations in Stan. The Stan code is stored as a character string in R, and later passed on into `brm`.

```r
myFuns <- "
vector compartmentmodel(real t, vector y, vector theta) {
  vector[3] dydt;
  real ke = theta[1];
  real dr = theta[2];
  real kp1 = theta[3];
  real kp2 = theta[4];

  dydt[1] = pow(ke, dr) * pow(t, dr - 1) * exp(-t * ke)/tgamma(dr)
          - (kp1 + kp2) * y[1]; // Exposure
  dydt[2] = kp2 * (y[1] - y[2]); // Outstanding
  dydt[3] = (kp1 *  y[1] + kp2 * y[2]); // Paid

  return dydt;
}
real lossemergence(real t, real devfreq, real ke, real dr,
                   real kp1, real kp2){
    vector[3] y0;
    array[1] vector[3] y;
    vector[4] theta;
    theta[1] = ke;
    theta[2] = dr;
    theta[3] = kp1;
    theta[4] = kp2;
    real out;
    // Set initial values
    y0[1] = 0; // Exposure
    y0[2] = 0; // Outstanding
    y0[3] = 0; // Paid

    y = ode_rk45(compartmentmodel, y0, 0, rep_array(t, 1), theta);
    out = y[1, 3];

    if(t > devfreq){ // paid greater dev period 1
    y = ode_rk45(compartmentmodel, y0, 0, rep_array(t - devfreq, 1), theta);
    // incremental paid
     out = out - y[1, 3];
    }

    return(out);
}
"
```

The following code defines the hierarchical structure using the formula interface in brms:

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

### 7.1.1 Multilevel effects with narrow priors

Model run with narrow priors for the multilevel effects:

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

```r
fit_loss <- brm(frml, prior = mypriors,
                data = lossDat, family = lognormal(), seed = 12345,
                stanvars = stanvar(scode = myFuns, block = "functions"),
                backend = "cmdstan",
                file="models/section_4/GenInsIncModelLog")
```

```r
fit_loss
#>  Family: lognormal
#>   Links: mu = identity; sigma = identity
#> Formula: incr_lr ~ eta
#>          eta ~ log(ELR * lossemergence(dev, 1, ke, dr, kp1, kp2))
#>          ke ~ exp(oke * 0.5)
#>          dr ~ 1 + 0.1 * exp(odr * 0.5)
#>          kp1 ~ 0.5 * exp(okp1 * 0.5)
#>          kp2 ~ 0.1 * exp(okp2 * 0.5)
#>          ELR ~ 1 + (1 | AY)
#>          oke ~ 1 + (1 | AY)
#>          odr ~ 1 + (1 | AY)
#>          okp1 ~ 1 + (1 | AY)
#>          okp2 ~ 1 + (1 | AY)
#>    Data: lossDat (Number of observations: 55)
#>   Draws: 4 chains, each with iter = 2000; warmup = 1000; thin = 1;
#>          total post-warmup draws = 4000
#>
#> Group-Level Effects:
#> ~AY (Number of levels: 10)
#>                    Estimate Est.Error l-95% CI
#> sd(ELR_Intercept)      0.04      0.03     0.00
#> sd(oke_Intercept)      0.08      0.06     0.00
#> sd(odr_Intercept)      0.09      0.07     0.00
#> sd(okp1_Intercept)     0.08      0.06     0.00
#> sd(okp2_Intercept)     0.09      0.07     0.00
#>                    u-95% CI Rhat Bulk_ESS Tail_ESS
#> sd(ELR_Intercept)      0.11 1.00     1888     2353
#> sd(oke_Intercept)      0.24 1.00     3973     2326
#> sd(odr_Intercept)      0.26 1.00     4078     2331
#> sd(okp1_Intercept)     0.24 1.00     3750     2260
#> sd(okp2_Intercept)     0.25 1.00     4492     2362
#>
#> Population-Level Effects:
#>                Estimate Est.Error l-95% CI u-95% CI
#> ELR_Intercept      0.49      0.03     0.43     0.56
#> oke_Intercept     -0.89      0.52    -1.93     0.10
#> odr_Intercept      0.51      1.04    -1.57     2.46
#> okp1_Intercept    -0.42      0.56    -1.39     0.81
#> okp2_Intercept    -0.00      1.00    -1.99     1.93
#>                Rhat Bulk_ESS Tail_ESS
#> ELR_Intercept  1.00     3962     2802
#> oke_Intercept  1.00     3757     3270
#> odr_Intercept  1.00     6878     2741
#> okp1_Intercept 1.00     3831     2896
#> okp2_Intercept 1.00     5455     3057
#>
#> Family Specific Parameters:
#>       Estimate Est.Error l-95% CI u-95% CI Rhat
#> sigma     0.37      0.04     0.30     0.45 1.00
#>       Bulk_ESS Tail_ESS
#> sigma     7423     3046
#>
#> Draws were sampled using sample(hmc). For each parameter, Bulk_ESS
#> and Tail_ESS are effective sample size measures, and Rhat is the potential
#> scale reduction factor on split chains (at convergence, Rhat = 1).
```

Population-level posterior parameters on original scale:

```r
x <- posterior_samples(fit_loss, "^b")
#> Warning: Method 'posterior_samples' is deprecated.
#> Please see ?as_draws for recommended alternatives.
mySummary <- function(x){
  c(Estimate = mean(x), Est.Error = sd(x),
    `l-95% CI` = as.numeric(quantile(x, probs = 0.025)),
    `u-95% CI` = as.numeric(quantile(x, probs = 0.975)))
}
rbind(
  ELR = mySummary(x[, 'b_ELR_Intercept']),
  ke = mySummary(exp(x[, 'b_oke_Intercept'] * 0.5)),
  dr = mySummary(1 + 0.1 * exp(x[, 'b_odr_Intercept'] * 0.5)),
  kp1 = mySummary(0.5 * exp(x[, 'b_okp1_Intercept'] * 0.5)),
  kp2 = mySummary(0.1 * exp(x[, 'b_okp2_Intercept'] * 0.5))
  )
#>     Estimate Est.Error l-95% CI u-95% CI
#> ELR   0.4904   0.03497  0.42569   0.5628
#> ke    0.6630   0.17537  0.38080   1.0524
#> dr    1.1470   0.07863  1.04562   1.3429
#> kp1   0.4228   0.13066  0.24900   0.7487
#> kp2   0.1132   0.05953  0.03694   0.2624
```

### 7.1.2 Multilevel effects with wider priors

Model run with wider priors for the multilevel effects:

```r
mypriors2 <- c(prior(inv_gamma(4, 2), nlpar = "ELR", lb=0),
              prior(normal(0, 1), nlpar = "oke"),
              prior(normal(0, 1), nlpar = "odr"),
              prior(normal(0, 1), nlpar = "okp1"),
              prior(normal(0, 1), nlpar = "okp2"),
              prior(student_t(10, 0, 1), class = "sd", nlpar = "ELR"),
              prior(student_t(10, 0, 1), class = "sd", nlpar = "oke"),
              prior(student_t(10, 0, 1), class = "sd", nlpar = "odr"),
              prior(student_t(10, 0, 1), class = "sd", nlpar = "okp1"),
              prior(student_t(10, 0, 1), class = "sd", nlpar = "okp2"),
              prior(student_t(10, 0, 1), class = "sigma"))
```

```r
fit_loss2 <- brm(frml, prior = mypriors2,
                data = lossDat, family = lognormal(), seed = 12345,
                control = list(adapt_delta = 0.9, max_treedepth=15),
                stanvars = stanvar(scode = myFuns, block = "functions"),
                backend = "cmdstan",
                file="models/section_4/GenInsIncModelLog2")
```

```r
fit_loss2
#> Warning: There were 3 divergent transitions after
#> warmup. Increasing adapt_delta above 0.9 may help. See
#> http://mc-stan.org/misc/warnings.html#divergent-transitions-after-warmup
#>  Family: lognormal
#>   Links: mu = identity; sigma = identity
#> Formula: incr_lr ~ eta
#>          eta ~ log(ELR * lossemergence(dev, 1, ke, dr, kp1, kp2))
#>          ke ~ exp(oke * 0.5)
#>          dr ~ 1 + 0.1 * exp(odr * 0.5)
#>          kp1 ~ 0.5 * exp(okp1 * 0.5)
#>          kp2 ~ 0.1 * exp(okp2 * 0.5)
#>          ELR ~ 1 + (1 | AY)
#>          oke ~ 1 + (1 | AY)
#>          odr ~ 1 + (1 | AY)
#>          okp1 ~ 1 + (1 | AY)
#>          okp2 ~ 1 + (1 | AY)
#>    Data: lossDat (Number of observations: 55)
#>   Draws: 4 chains, each with iter = 2000; warmup = 1000; thin = 1;
#>          total post-warmup draws = 4000
#>
#> Group-Level Effects:
#> ~AY (Number of levels: 10)
#>                    Estimate Est.Error l-95% CI
#> sd(ELR_Intercept)      0.05      0.04     0.00
#> sd(oke_Intercept)      0.24      0.20     0.01
#> sd(odr_Intercept)      0.67      0.53     0.03
#> sd(okp1_Intercept)     0.25      0.20     0.01
#> sd(okp2_Intercept)     0.79      0.62     0.03
#>                    u-95% CI Rhat Bulk_ESS Tail_ESS
#> sd(ELR_Intercept)      0.13 1.00     1581     1748
#> sd(oke_Intercept)      0.74 1.00     2185     1569
#> sd(odr_Intercept)      1.96 1.00     3455     2032
#> sd(okp1_Intercept)     0.76 1.00     2220     1522
#> sd(okp2_Intercept)     2.31 1.00     2655     1832
#>
#> Population-Level Effects:
#>                Estimate Est.Error l-95% CI u-95% CI
#> ELR_Intercept      0.49      0.04     0.42     0.57
#> oke_Intercept     -0.87      0.54    -1.91     0.16
#> odr_Intercept      0.28      0.99    -1.70     2.17
#> okp1_Intercept    -0.45      0.59    -1.45     0.79
#> okp2_Intercept    -0.02      0.98    -1.97     1.91
#>                Rhat Bulk_ESS Tail_ESS
#> ELR_Intercept  1.00     2589     2615
#> oke_Intercept  1.00     2265     2717
#> odr_Intercept  1.00     4698     3050
#> okp1_Intercept 1.00     2172     2900
#> okp2_Intercept 1.00     4480     2937
#>
#> Family Specific Parameters:
#>       Estimate Est.Error l-95% CI u-95% CI Rhat
#> sigma     0.37      0.04     0.30     0.45 1.00
#>       Bulk_ESS Tail_ESS
#> sigma     4751     2710
#>
#> Draws were sampled using sample(hmc). For each parameter, Bulk_ESS
#> and Tail_ESS are effective sample size measures, and Rhat is the potential
#> scale reduction factor on split chains (at convergence, Rhat = 1).
```

Population-level posterior parameters on original scale:

```r
x <- posterior_samples(fit_loss2, "^b")
#> Warning: Method 'posterior_samples' is deprecated.
#> Please see ?as_draws for recommended alternatives.
rbind(
  ELR = mySummary(x[, 'b_ELR_Intercept']),
  ke = mySummary(exp(x[, 'b_oke_Intercept'] * 0.5)),
  dr = mySummary(1 + 0.1 * exp(x[, 'b_odr_Intercept'] * 0.5)),
  kp1 = mySummary(0.5 * exp(x[, 'b_okp1_Intercept'] * 0.5)),
  kp2 = mySummary(0.1 * exp(x[, 'b_okp2_Intercept'] * 0.5))
  )
#>     Estimate Est.Error l-95% CI u-95% CI
#> ELR   0.4918   0.03773  0.42064   0.5726
#> ke    0.6703   0.18353  0.38439   1.0829
#> dr    1.1299   0.06635  1.04264   1.2963
#> kp1   0.4187   0.13412  0.24213   0.7410
#> kp2   0.1118   0.05736  0.03726   0.2600
```

Note that in order to predict the models, the user defined Stan functions have to be exported to R via:

```r
expose_functions(fit_loss, vectorize = TRUE)
```

## 7.2 R code from case study in section 5

### 7.2.1 Data

The data used for the case study is a subset of the wkcomp data set from the raw R package (Fannin (2018)).

```r
library(raw)
data(wkcomp)
library(data.table)
library(tidyverse)
# Convert to tibble, rename cols, add calendar year and loss ratio columns
wkcomp <- wkcomp %>%
  as_tibble() %>%
  rename(accident_year = AccidentYear, dev_year = Lag,
         entity_id = GroupCode) %>%
  mutate(cal_year = accident_year + dev_year - 1,
         paid_loss_ratio = CumulativePaid/DirectEP,
         os_loss_ratio = (CumulativeIncurred - CumulativePaid)/DirectEP)

# Add incremental paid loss ratio column
wkcomp <- wkcomp %>%
  group_by(entity_id, accident_year) %>%
  arrange(dev_year) %>%
  mutate(incr_paid_loss_ratio = paid_loss_ratio -
           shift(paid_loss_ratio, n=1, fill=0,
                 type="lag")) %>%
  ungroup() %>%
  arrange(entity_id, accident_year, dev_year)

# Stack paid and os into one column + define train and test
wkcomp2 <- wkcomp %>%
  transmute(
    entity_id, accident_year, dev_year, cal_year,
    premium = DirectEP, delta = 1, deltaf = "paid",
    loss_ratio_train = ifelse(cal_year < max(accident_year),
                              incr_paid_loss_ratio,
                              NA),
    loss_ratio_test = ifelse(cal_year >= max(accident_year),
                             incr_paid_loss_ratio,
                             NA),
    loss_amount_train = ifelse(cal_year < max(accident_year),
                               CumulativePaid,
                               NA),
    loss_amount_test = ifelse(cal_year >= max(accident_year),
                              CumulativePaid,
                              NA)
  ) %>%
  bind_rows(
    wkcomp %>%
      transmute(
        entity_id, accident_year, dev_year, cal_year,
        premium = DirectEP, delta = 0, deltaf = "os",
        loss_ratio_train = ifelse(cal_year < max(accident_year),
                                  os_loss_ratio,
                                  NA),
        loss_ratio_test = ifelse(cal_year >= max(accident_year),
                                 os_loss_ratio,
                                 NA),
        loss_amount_train = ifelse(cal_year < max(accident_year),
                                   CumulativeIncurred - CumulativePaid,
                                   NA),
        loss_amount_test = ifelse(cal_year >= max(accident_year),
                                  CumulativeIncurred - CumulativePaid,
                                  NA)
      )
  )
```

Filter for company "337":

```r
dat337 <- wkcomp2 %>% filter(entity_id ==337)
```

### 7.2.2 Model 1

```r
myFunsCumPaid <- "
real paid(real t, real ker, real kp, real RLR, real RRF){
 return(
  RLR*RRF/(ker - kp) * (ker *(1 - exp(-kp*t)) -
  kp*(1 - exp(-ker*t)))
 );
}
real outstanding(real t, real ker, real kp, real RLR){
 return(
  (RLR*ker/(ker - kp) * (exp(-kp*t) - exp(-ker*t)))
 );
}
real claimsprocess(real t, real ker, real kp,
                   real RLR, real RRF, real delta){
    real out;
    out = outstanding(t, ker, kp, RLR) * (1 - delta) +
          paid(t, ker, kp, RLR, RRF) * delta;

    return(out);
}
"
```

```r
frml1 <- bf(loss_amount_train ~ premium * claimsprocess(dev_year, ker, kp,
                                                        RLR, RRF, delta),
            nlf(ker ~ 3 * exp(oker * 0.1)),
            nlf(kp ~ 1 * exp(okp * 0.1)),
            nlf(RLR ~ 0.7 * exp(oRLR * 0.2)),
            nlf(RRF ~ 0.8 * exp(oRRF * 0.1)),
            oRLR ~ 1 + (1 | ID | accident_year),
            oRRF ~ 1 + (1 | ID | accident_year),
            oker ~ 1,  okp ~ 1,
            sigma ~ 0 + deltaf,
            nl = TRUE)
```

```r
mypriors1 <- c(prior(normal(0, 1), nlpar = "oRLR"),
               prior(normal(0, 1), nlpar = "oRRF"),
               prior(normal(0, 1), nlpar = "oker"),
               prior(normal(0, 1), nlpar = "okp"),
               prior(student_t(1, 0, 1000), class = "b",
                     coef="deltafpaid", dpar= "sigma"),
               prior(student_t(1, 0, 1000), class = "b",
                     coef="deltafos", dpar= "sigma"),
               prior(student_t(10, 0, 0.2), class = "sd", nlpar = "oRLR"),
               prior(student_t(10, 0, 0.1), class = "sd", nlpar = "oRRF"),
               prior(lkj(1), class="cor"))
```

```r
m1fit <- brm(frml1, data = dat337[!is.na(loss_ratio_train)],
             family = gaussian(),
             prior = mypriors1,
             stanvars = stanvar(scode = myFunsCumPaid, block = "functions"),
             control = list(adapt_delta = 0.99, max_treedepth=15),
             backend = "cmdstan",
             file="models/section_5/CaseStudy_Model_1",
             seed = 123, iter = 2000, chains = 4)
```

```r
m1fit
#>  Family: gaussian
#>   Links: mu = identity; sigma = log
#> Formula: loss_amount_train ~ premium * claimsprocess(dev_year, ker, kp, RLR, RRF, delta)
#>          ker ~ 3 * exp(oker * 0.1)
#>          kp ~ 1 * exp(okp * 0.1)
#>          RLR ~ 0.7 * exp(oRLR * 0.2)
#>          RRF ~ 0.8 * exp(oRRF * 0.1)
#>          oRLR ~ 1 + (1 | ID | accident_year)
#>          oRRF ~ 1 + (1 | ID | accident_year)
#>          oker ~ 1
#>          okp ~ 1
#>          sigma ~ 0 + deltaf
#>    Data: dat337[!is.na(loss_ratio_train)] (Number of observations: 90)
#>   Draws: 4 chains, each with iter = 2000; warmup = 1000; thin = 1;
#>          total post-warmup draws = 4000
#>
#> Group-Level Effects:
#> ~accident_year (Number of levels: 9)
#>                                    Estimate Est.Error
#> sd(oRLR_Intercept)                     0.64      0.15
#> sd(oRRF_Intercept)                     0.74      0.22
#> cor(oRLR_Intercept,oRRF_Intercept)     0.54      0.26
#>                                    l-95% CI u-95% CI
#> sd(oRLR_Intercept)                     0.40     0.98
#> sd(oRRF_Intercept)                     0.31     1.22
#> cor(oRLR_Intercept,oRRF_Intercept)    -0.03     0.95
#>                                    Rhat Bulk_ESS
#> sd(oRLR_Intercept)                 1.00     1761
#> sd(oRRF_Intercept)                 1.00     1303
#> cor(oRLR_Intercept,oRRF_Intercept) 1.00     1570
#>                                    Tail_ESS
#> sd(oRLR_Intercept)                     2307
#> sd(oRRF_Intercept)                      822
#> cor(oRLR_Intercept,oRRF_Intercept)     1495
#>
#> Population-Level Effects:
#>                  Estimate Est.Error l-95% CI u-95% CI
#> oRLR_Intercept       1.45      0.27     0.92     1.98
#> oRRF_Intercept      -1.36      0.41    -2.18    -0.58
#> oker_Intercept      -5.43      0.81    -6.89    -3.76
#> okp_Intercept       -8.48      0.36    -9.17    -7.79
#> sigma_deltafos       8.26      0.16     7.96     8.59
#> sigma_deltafpaid     6.66      0.20     6.31     7.07
#>                  Rhat Bulk_ESS Tail_ESS
#> oRLR_Intercept   1.00     1056     1614
#> oRRF_Intercept   1.00     2499     2600
#> oker_Intercept   1.00     1549     2224
#> okp_Intercept    1.00     2148     3089
#> sigma_deltafos   1.00     1237     1411
#> sigma_deltafpaid 1.00     1347     2074
#>
#> Draws were sampled using sample(hmc). For each parameter, Bulk_ESS
#> and Tail_ESS are effective sample size measures, and Rhat is the potential
#> scale reduction factor on split chains (at convergence, Rhat = 1).
```

### 7.2.3 Model 2

```r
myFunsIncrPaid <- "
real paid(real t, real ker, real kp, real RLR, real RRF){
 return(
  RLR*RRF/(ker - kp) * (ker *(1 - exp(-kp*t)) -
  kp*(1 - exp(-ker*t)))
 );
}
real outstanding(real t, real ker, real kp, real RLR){
 return(
  (RLR*ker/(ker - kp) * (exp(-kp*t) - exp(-ker*t)))
 );
}
real incrclaimsprocess(real t, real devfreq, real ker, real kp,
                   real RLR, real RRF, real delta){
    real out;
    out = outstanding(t, ker, kp, RLR) * (1 - delta) +
          paid(t, ker, kp, RLR, RRF) * delta;

    if( (delta > 0) && (t > devfreq) ){ // paid greater dev period 1
    // incremental paid
     out = out - paid(t - devfreq, ker, kp, RLR, RRF)*delta;
    }
    return(out);
}
"
```

```r
frml2 <- bf(loss_ratio_train ~ eta,
            nlf(eta ~ log(incrclaimsprocess(dev_year, 1.0, ker, kp,
                                        RLR, RRF, delta))),
            nlf(ker ~ 3 * exp(oker * 0.1)),
            nlf(kp ~ 1 * exp(okp * 0.1)),
            nlf(RLR ~ 0.7 * exp(oRLR * 0.2)),
            nlf(RRF ~ 0.8 * exp(oRRF * 0.1)),
            oRLR ~ 1 + (1 | ID | accident_year) + (1 | dev_year),
            oRRF ~ 1 + (1 | ID | accident_year) + (1 | dev_year),
            oker ~ 1  + (1 | accident_year) + (1 | dev_year),
            okp ~ 1 + (1 | accident_year) + (1 | dev_year),
            sigma ~ 0 + deltaf, nl = TRUE)
```

```r
mypriors2 <- c(prior(normal(0, 1), nlpar = "oRLR"),
               prior(normal(0, 1), nlpar = "oRRF"),
               prior(normal(0, 1), nlpar = "oker"),
               prior(normal(0, 1), nlpar = "okp"),
               prior(normal(log(0.2), 0.2), class = "b",
                     coef="deltafpaid", dpar= "sigma"),
               prior(normal(log(0.2), 0.2), class = "b",
                     coef="deltafos", dpar= "sigma"),
               prior(student_t(10, 0, 0.3), class = "sd", nlpar = "oker"),
               prior(student_t(10, 0, 0.3), class = "sd", nlpar = "okp"),
               prior(student_t(10, 0, 0.7), class = "sd", nlpar = "oRLR"),
               prior(student_t(10, 0, 0.5), class = "sd", nlpar = "oRRF"),
               prior(lkj(1), class="cor"))
```

```r
m2fit <- brm(frml2, data = dat337[!is.na(loss_ratio_train)],
             family = brmsfamily("lognormal", link_sigma = "log"),
             prior = mypriors2,
             stanvars = stanvar(scode = myFunsIncrPaid, block = "functions"),
             control = list(adapt_delta = 0.99, max_treedepth=15),
             backend = "cmdstan",
             file="models/section_5/CaseStudy_Model_2",
             seed = 123, iter = 2000, chains = 4)
```

```r
m2fit
#>  Family: lognormal
#>   Links: mu = identity; sigma = log
#> Formula: loss_ratio_train ~ eta
#>          eta ~ log(incrclaimsprocess(dev_year, 1, ker, kp, RLR, RRF, delta))
#>          ker ~ 3 * exp(oker * 0.1)
#>          kp ~ 1 * exp(okp * 0.1)
#>          RLR ~ 0.7 * exp(oRLR * 0.2)
#>          RRF ~ 0.8 * exp(oRRF * 0.1)
#>          oRLR ~ 1 + (1 | ID | accident_year) + (1 | dev_year)
#>          oRRF ~ 1 + (1 | ID | accident_year) + (1 | dev_year)
#>          oker ~ 1 + (1 | accident_year) + (1 | dev_year)
#>          okp ~ 1 + (1 | accident_year) + (1 | dev_year)
#>          sigma ~ 0 + deltaf
#>    Data: dat337[!is.na(loss_ratio_train)] (Number of observations: 90)
#>   Draws: 4 chains, each with iter = 2000; warmup = 1000; thin = 1;
#>          total post-warmup draws = 4000
#>
#> Group-Level Effects:
#> ~accident_year (Number of levels: 9)
#>                                    Estimate Est.Error
#> sd(oRLR_Intercept)                     0.67      0.30
#> sd(oRRF_Intercept)                     0.75      0.48
#> sd(oker_Intercept)                     0.27      0.21
#> sd(okp_Intercept)                      2.07      1.23
#> cor(oRLR_Intercept,oRRF_Intercept)     0.36      0.47
#>                                    l-95% CI u-95% CI
#> sd(oRLR_Intercept)                     0.15     1.33
#> sd(oRRF_Intercept)                     0.04     1.81
#> sd(oker_Intercept)                     0.01     0.79
#> sd(okp_Intercept)                      0.53     4.96
#> cor(oRLR_Intercept,oRRF_Intercept)    -0.73     0.98
#>                                    Rhat Bulk_ESS
#> sd(oRLR_Intercept)                 1.00      985
#> sd(oRRF_Intercept)                 1.00      906
#> sd(oker_Intercept)                 1.00     3517
#> sd(okp_Intercept)                  1.01      414
#> cor(oRLR_Intercept,oRRF_Intercept) 1.00     2448
#>                                    Tail_ESS
#> sd(oRLR_Intercept)                      992
#> sd(oRRF_Intercept)                     1410
#> sd(oker_Intercept)                     1826
#> sd(okp_Intercept)                      1430
#> cor(oRLR_Intercept,oRRF_Intercept)     2245
#>
#> ~dev_year (Number of levels: 9)
#>                    Estimate Est.Error l-95% CI
#> sd(oRLR_Intercept)     0.36      0.27     0.01
#> sd(oRRF_Intercept)     1.09      0.51     0.14
#> sd(oker_Intercept)     0.27      0.23     0.01
#> sd(okp_Intercept)      0.55      0.70     0.02
#>                    u-95% CI Rhat Bulk_ESS Tail_ESS
#> sd(oRLR_Intercept)     1.00 1.00     1105     1757
#> sd(oRRF_Intercept)     2.19 1.01      818      833
#> sd(oker_Intercept)     0.84 1.00     2732     1372
#> sd(okp_Intercept)      2.49 1.01      456      259
#>
#> Population-Level Effects:
#>                  Estimate Est.Error l-95% CI u-95% CI
#> oRLR_Intercept       1.54      0.44     0.67     2.43
#> oRRF_Intercept      -1.45      0.69    -2.77    -0.10
#> oker_Intercept      -1.30      1.06    -3.38     0.79
#> okp_Intercept       -5.16      1.67    -7.69    -1.71
#> sigma_deltafos      -1.79      0.17    -2.13    -1.49
#> sigma_deltafpaid    -1.88      0.16    -2.20    -1.55
#>                  Rhat Bulk_ESS Tail_ESS
#> oRLR_Intercept   1.00     1224     2349
#> oRRF_Intercept   1.00     1704     2550
#> oker_Intercept   1.00     3574     3139
#> okp_Intercept    1.01      507     1728
#> sigma_deltafos   1.00     1200     1821
#> sigma_deltafpaid 1.00     1105     2179
#>
#> Draws were sampled using sample(hmc). For each parameter, Bulk_ESS
#> and Tail_ESS are effective sample size measures, and Rhat is the potential
#> scale reduction factor on split chains (at convergence, Rhat = 1).
```

### 7.2.4 Model 3

```r
CycleIndex <- data.table(
  accident_year = 1988:1997,
  RLM = c(1, 1.18, 1.22, 1.05, 1, 0.87, 0.94, 1.34, 1.64, 2.14)^0.6,
  RRM = c(1, 1.05, 1.05, 1.01, 1.0, 0.95, 0.99, 1.1, 1.25, 1.35)^0.6
)
setkey(dat337, accident_year)
setkey(CycleIndex, accident_year)
dat337 <- CycleIndex[dat337]
```

```r
frml3 <- bf(loss_ratio_train ~ eta,
            nlf(eta ~ log(incrclaimsprocess(dev_year, 1.0, ker, kp,
                                        RLR, RRF, delta))),
            nlf(ker ~ 3 * exp(oker * 0.1)),
            nlf(kp ~ 1 * exp(okp * 0.1)),
            nlf(RLR ~ 0.7 * exp(oRLR * 0.2) * (RLM^lambda1)),
            nlf(RRF ~ 0.8 * exp(oRRF * 0.1) * (RRM^lambda2)),
            oRLR ~ 1 + (1 | ID | accident_year) + (1 | dev_year),
            oRRF ~ 1 + (1 | ID | accident_year) + (1 | dev_year),
            lambda1 ~  1,
            lambda2 ~ 1,
            oker ~ 1  + (1 | accident_year) + (1 | dev_year),
            okp ~ 1 + (1 | accident_year) + (1 | dev_year),
            sigma ~ 0 + deltaf, nl = TRUE)
```

```r
mypriors3 <- c(prior(normal(0, 1), nlpar = "oRLR"),
               prior(normal(0, 1), nlpar = "oRRF"),
               prior(normal(0, 1), nlpar = "oker"),
               prior(normal(0, 1), nlpar = "okp"),
               prior(normal(log(0.2), 0.2), class = "b",
                     coef="deltafpaid", dpar= "sigma"),
               prior(normal(log(0.2), 0.2), class = "b",
                     coef="deltafos", dpar= "sigma"),
               prior(student_t(10, 0, 0.3), class = "sd", nlpar = "oker"),
               prior(student_t(10, 0, 0.3), class = "sd", nlpar = "okp"),
               prior(student_t(10, 0, 0.7), class = "sd", nlpar = "oRLR"),
               prior(student_t(10, 0, 0.5), class = "sd", nlpar = "oRRF"),
               prior(normal(1, 0.25), nlpar = "lambda1"),
               prior(normal(1, 0.25), nlpar = "lambda2"),
               prior(lkj(1), class="cor"))
```

```r
m3fit <- brm(frml3, data = dat337[!is.na(loss_ratio_train)],
             family = brmsfamily("lognormal", link_sigma = "log"),
             prior = mypriors3,
             stanvars = stanvar(scode = myFunsIncrPaid, block = "functions"),
             control = list(adapt_delta = 0.99, max_treedepth=15),
             backend = "cmdstan",
             file="models/section_5/CaseStudy_Model_3",
             seed = 123, iter = 2000, chains = 4)
```

```r
m3fit
#>  Family: lognormal
#>   Links: mu = identity; sigma = log
#> Formula: loss_ratio_train ~ eta
#>          eta ~ log(incrclaimsprocess(dev_year, 1, ker, kp, RLR, RRF, delta))
#>          ker ~ 3 * exp(oker * 0.1)
#>          kp ~ 1 * exp(okp * 0.1)
#>          RLR ~ 0.7 * exp(oRLR * 0.2) * (RLM^lambda1)
#>          RRF ~ 0.8 * exp(oRRF * 0.1) * (RRM^lambda2)
#>          oRLR ~ 1 + (1 | ID | accident_year) + (1 | dev_year)
#>          oRRF ~ 1 + (1 | ID | accident_year) + (1 | dev_year)
#>          lambda1 ~ 1
#>          lambda2 ~ 1
#>          oker ~ 1 + (1 | accident_year) + (1 | dev_year)
#>          okp ~ 1 + (1 | accident_year) + (1 | dev_year)
#>          sigma ~ 0 + deltaf
#>    Data: dat337[!is.na(loss_ratio_train)] (Number of observations: 90)
#>   Draws: 4 chains, each with iter = 2000; warmup = 1000; thin = 1;
#>          total post-warmup draws = 4000
#>
#> Group-Level Effects:
#> ~accident_year (Number of levels: 9)
#>                                    Estimate Est.Error
#> sd(oRLR_Intercept)                     0.31      0.22
#> sd(oRRF_Intercept)                     0.78      0.41
#> sd(oker_Intercept)                     0.26      0.20
#> sd(okp_Intercept)                      1.73      1.28
#> cor(oRLR_Intercept,oRRF_Intercept)     0.16      0.52
#>                                    l-95% CI u-95% CI
#> sd(oRLR_Intercept)                     0.01     0.81
#> sd(oRRF_Intercept)                     0.06     1.63
#> sd(oker_Intercept)                     0.01     0.76
#> sd(okp_Intercept)                      0.41     4.96
#> cor(oRLR_Intercept,oRRF_Intercept)    -0.87     0.96
#>                                    Rhat Bulk_ESS
#> sd(oRLR_Intercept)                 1.00     1259
#> sd(oRRF_Intercept)                 1.00     1200
#> sd(oker_Intercept)                 1.00     3046
#> sd(okp_Intercept)                  1.01      371
#> cor(oRLR_Intercept,oRRF_Intercept) 1.00     1444
#>                                    Tail_ESS
#> sd(oRLR_Intercept)                     1703
#> sd(oRRF_Intercept)                     1403
#> sd(oker_Intercept)                     1884
#> sd(okp_Intercept)                      1220
#> cor(oRLR_Intercept,oRRF_Intercept)     1659
#>
#> ~dev_year (Number of levels: 9)
#>                    Estimate Est.Error l-95% CI
#> sd(oRLR_Intercept)     0.38      0.27     0.02
#> sd(oRRF_Intercept)     1.16      0.51     0.14
#> sd(oker_Intercept)     0.26      0.23     0.01
#> sd(okp_Intercept)      0.57      0.76     0.02
#>                    u-95% CI Rhat Bulk_ESS Tail_ESS
#> sd(oRLR_Intercept)     1.03 1.01      893     1731
#> sd(oRRF_Intercept)     2.20 1.00      843      775
#> sd(oker_Intercept)     0.83 1.00     2960     1726
#> sd(okp_Intercept)      2.85 1.00      593      286
#>
#> Population-Level Effects:
#>                   Estimate Est.Error l-95% CI u-95% CI
#> oRLR_Intercept        1.32      0.44     0.52     2.31
#> oRRF_Intercept       -1.57      0.73    -2.98    -0.12
#> lambda1_Intercept     1.07      0.20     0.68     1.46
#> lambda2_Intercept     1.02      0.25     0.52     1.50
#> oker_Intercept       -1.29      1.07    -3.42     0.85
#> okp_Intercept        -5.65      1.70    -7.92    -1.81
#> sigma_deltafos       -1.81      0.16    -2.13    -1.52
#> sigma_deltafpaid     -1.91      0.16    -2.21    -1.59
#>                   Rhat Bulk_ESS Tail_ESS
#> oRLR_Intercept    1.00      906     1901
#> oRRF_Intercept    1.00     1249     1599
#> lambda1_Intercept 1.00     4564     3036
#> lambda2_Intercept 1.00     5639     3012
#> oker_Intercept    1.00     3529     2966
#> okp_Intercept     1.01      429     1497
#> sigma_deltafos    1.00     1631     2742
#> sigma_deltafpaid  1.00     1591     2323
#>
#> Draws were sampled using sample(hmc). For each parameter, Bulk_ESS
#> and Tail_ESS are effective sample size measures, and Rhat is the potential
#> scale reduction factor on split chains (at convergence, Rhat = 1).
```

Note, in order to predict from the models, the user defined Stan functions have to be exported to R via:

```r
expose_functions(m1, vectorize = TRUE)
expose_functions(m2, vectorize = TRUE)
expose_functions(m3, vectorize = TRUE)
```

## 7.3 Session info

```r
utils:::print.sessionInfo(session_info, local=FALSE)
#> R version 4.3.1 (2023-06-16)
#> Platform: x86_64-pc-linux-gnu (64-bit)
#> Running under: Debian GNU/Linux bookworm/sid
#>
#> Matrix products: default
#> BLAS:   /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3
#> LAPACK: /usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblasp-r0.3.21.so;  LAPACK version 3.11.0
#>
#> attached base packages:
#> NULL
#>
#> other attached packages:
#>  [1] rstan_2.21.8        cowplot_1.1.1
#>  [3] ggridges_0.5.4      raw_0.1.8
#>  [5] MASS_7.3-60         knitr_1.43
#>  [7] modelr_0.1.11       lubridate_1.9.2
#>  [9] forcats_1.0.0       stringr_1.5.0
#> [11] dplyr_1.1.2         purrr_1.0.1
#> [13] readr_2.1.4         tidyr_1.3.0
#> [15] tibble_3.2.1        ggplot2_3.4.2
#> [17] tidyverse_2.0.0     tidybayes_3.0.4
#> [19] latticeExtra_0.6-30 lattice_0.20-45
#> [21] ChainLadder_0.2.18  data.table_1.14.8
#> [23] bayesplot_1.10.0    brms_2.20.0
#> [25] Rcpp_1.0.11         cmdstanr_0.6.0
#> [27] deSolve_1.35
#>
#> loaded via a namespace (and not attached):
#>   [1] RColorBrewer_1.1-3   tensorA_0.36.2
#>   [3] rstudioapi_0.15.0    jsonlite_1.8.7
#>   [5] magrittr_2.0.3       farver_2.1.1
#>   [7] rmarkdown_2.23       vctrs_0.6.3
#>   [9] minqa_1.2.5          base64enc_0.1-3
#>  [11] htmltools_0.5.5      distributional_0.3.2
#>  [13] broom_1.0.5          sass_0.4.7
#>  [15] StanHeaders_2.26.27  bslib_0.5.0
#>  [17] htmlwidgets_1.6.2    plyr_1.8.8
#>  [19] sandwich_3.0-2       zoo_1.8-12
#>  [21] cachem_1.0.8         igraph_1.5.0.1
#>  [23] mime_0.12            lifecycle_1.0.3
#>  [25] pkgconfig_2.0.3      colourpicker_1.2.0
#>  [27] Matrix_1.6-0         R6_2.5.1
#>  [29] fastmap_1.1.1        shiny_1.7.4.1
#>  [31] digest_0.6.33        colorspace_2.1-0
#>  [33] ps_1.7.5             crosstalk_1.2.0
#>  [35] labeling_0.4.2       timechange_0.2.0
#>  [37] fansi_1.0.4          mgcv_1.8-42
#>  [39] abind_1.4-5          compiler_4.3.1
#>  [41] withr_2.5.0          backports_1.4.1
#>  [43] inline_0.3.19        shinystan_2.6.0
#>  [45] carData_3.0-5        DBI_1.1.3
#>  [47] biglm_0.9-2.1        pkgbuild_1.4.2
#>  [49] highr_0.10           gtools_3.9.4
#>  [51] loo_2.6.0            tools_4.3.1
#>  [53] lmtest_0.9-40        httpuv_1.6.11
#>  [55] threejs_0.3.3        systemfit_1.1-30
#>  [57] glue_1.6.2           callr_3.7.3
#>  [59] nlme_3.1-162         promises_1.2.0.1
#>  [61] grid_4.3.1           checkmate_2.2.0
#>  [63] reshape2_1.4.4       generics_0.1.3
#>  [65] gtable_0.3.3         tzdb_0.4.0
#>  [67] hms_1.1.3            car_3.1-2
#>  [69] utf8_1.2.3           pillar_1.9.0
#>  [71] ggdist_3.3.0         markdown_1.7
#>  [73] posterior_1.4.1      later_1.3.1
#>  [75] splines_4.3.1        deldir_1.0-9
#>  [77] tidyselect_1.2.0     miniUI_0.1.1.1
#>  [79] expint_0.1-8         arrayhelpers_1.1-0
#>  [81] gridExtra_2.3        bookdown_0.32
#>  [83] stats4_4.3.1         xfun_0.39
#>  [85] bridgesampling_1.1-2 statmod_1.5.0
#>  [87] matrixStats_1.0.0    DT_0.28
#>  [89] stringi_1.7.12       yaml_2.3.7
#>  [91] evaluate_0.21        codetools_0.2-19
#>  [93] RcppEigen_0.3.3.9.3  interp_1.1-4
#>  [95] cli_3.6.1            RcppParallel_5.1.7
#>  [97] shinythemes_1.2.0    tweedie_2.3.5
#>  [99] xtable_1.8-4         cplm_0.7-11
#> [101] munsell_0.5.0        processx_3.8.2
#> [103] jquerylib_0.1.4      coda_0.19-4
#> [105] png_0.1-8            svUnit_1.0.6
#> [107] parallel_4.3.1       rstantools_2.3.1.1
#> [109] ellipsis_0.3.2       prettyunits_1.1.1
#> [111] dygraphs_1.1.1.6     jpeg_0.1-10
#> [113] Brobdingnag_1.2-9    viridisLite_0.4.2
#> [115] mvtnorm_1.2-2        actuar_3.3-2
#> [117] scales_1.2.1         xts_0.13.1
#> [119] crayon_1.5.2         rlang_1.1.1
#> [121] shinyjs_2.1.0
```

### References

Fannin, Brian A. 2018. _Raw: R Actuarial Workshops_. https://CRAN.R-project.org/package=raw.