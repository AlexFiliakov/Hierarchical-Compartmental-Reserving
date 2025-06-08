# 1 Introduction

Claims reserving, pricing, and capital modeling are core to actuarial functions.
The assumptions used in the underlying actuarial models play a key role in the
management of any insurance company.

Knowing when those underlying assumptions are no longer valid is critical for
the business to initiate change. Transparent models that clearly state the
underlying assumptions are easier to test and challenge, and hence can speed
up the process for change.

Unfortunately, many underlying risk factors in insurance are not directly
measurable and are latent in nature. Although prices are set for all policies,
only a fraction of policies will incur losses. Reserving is often based on
relatively sparse data to make predictions about future payments, potentially
over long time horizons.

Combining judgment about future developments with historical data is therefore
common practice for many reserving teams, particularly when entering a new
product, line of business, or geography, or when changes to products and
business processes would make past data a less credible predictor. Modern
Bayesian modeling provides a rich tool kit for bringing together the expertise
and business insight of the actuary and augmenting and updating it with data.

In situations where the actuary has access to large volumes of data,
nonparametric machine learning techniques might provide a better approach.
Some of these are based on enhancement of traditional approaches such as the
chain-ladder method (Wüthrich (2018), Carrato and Visintin (2019)), with
others using neural networks (Kuo (2018), Gabrielli, Richman, and Wüthrich (2018))
and Gaussian processes (Lally and Hartman (2018)).

With small and sparse data, parametric models such as growth curves
(Sherman (1984), Clark (2003), Guszcza (2008)) can help the actuary capture key
claims development features without overfitting, however, the actuary may
require expertise and judgement in the selection of the growth curve and its
parametrization.

Hierarchical compartmental reserving models provide an alternative parametric
framework for describing the high-level business processes driving claims
development in insurance (Morris (2016)). Rather than selecting a growth
curve, the experienced modeler can build loss emergence patterns from first
principles using differential equations. Additionally, these loss emergence
patterns can be constructed in a way that allows outstanding and paid data to
be described simultaneously.

The starting point mirrors that of a scientist trying to describe a
particular process in the real world using a mathematical model. By its very
nature, the model will only be able to approximate the real world. We derive a
"small-world" view that makes simplified assumptions about the real world, but
which may allow us to improve our understanding of key processes. In turn, we
can attempt to address our real-world questions by testing various ideas about
how the real world functions.

Compared with many machine-learning methods, which are sometimes described as
"black boxes", hierarchical compartmental reserving models can be viewed as
"transparent boxes." All modeling assumptions must be articulated by the
practitioner, with the benefit that expert knowledge can be incorporated, and
each modeling assumption can be challenged more easily by other experts.

Finally, working within a parametric framework allows us to simulate artificial
data in advance of fitting any models. An a priori understanding of model
suitability should steer practitioners to align modeling assumptions
with their expectations of reality, and therefore may improve predictive
performance.

## 1.1 Outline of the document

This document builds on the original paper by Morris (Morris (2016)).
It provides a practical introduction to hierarchical compartmental reserving in
a Bayesian framework and is outlined as follows:

- In Section 2 we develop the original ordinary differential equation (ODE)
model and demonstrate how the model can be modified to allow for different
claims processes, including different settlement speeds for standard versus
disputed claims and different exposure to reporting processes.
- In Section 3 we build the stochastic part of the model and provide guidance
on how to parameterize prior parameter distributions to optimize model
convergence. Furthermore, we discuss why one should model incremental paid
data in the context of underlying statistical assumptions and previously
published methodologies.
- In Section 4 we add hierarchical structure to the model, which links
compartmental models back to credibility theory and regularization. The
'GenIns' data set is used to illustrate these concepts as we fit the model
to actual claims data, and we highlight the conceptual differences between
expected and ultimate loss ratios when interpreting model outputs.
- Section 5 concludes with a case study demonstrating how such models can be
implemented in "Stan" using the "brms" package. Models of varying complexity
are tested against each other, with add-ons such as parameter variation by
both origin and development period, and market cycle submodels. Model
selection and validation is demonstrated using posterior predictive checks
and holdout sample methods.
- Section 6 summarizes the document and provides an outlook for future research.
- The appendix presents the R code to replicate the models in Sections 4 and 5.

We assume the reader is somewhat familiar with Bayesian modelling concepts.
Good introductory textbooks to Bayesian data analysis are
(McElreath (2015), Kruschke (2014), A. Gelman et al. (2014)). For hierarchical models
we recommend (Andrew Gelman and Hill (2007)), and for best practices on a Bayesian workflow, see
(Betancourt (2018)).

In this document we will demonstrate practical examples using the brms (Bürkner (2017))
interface to the probabilistic programming language Stan (Carpenter et al. (2017)) from
R (R Core Team (2019)).

The brm function – short for "Bayesian regression model" – in brms
allows us to write our models in a similar way to a GLM or multilevel model
with the popular glm or lme4::lmer (Bates et al. (2015)) R functions. The Stan code is
generated and executed by brm. Experienced users can access all underlying
Stan code from brms as required.

Stan is a C++ library for Bayesian inference using the No-U-Turn sampler
(a variant of Hamiltonian Monte Carlo – "HMC") or frequentist inference via L-BFGS
optimization (Carpenter et al. (2017)). For an introduction to HMC see (Betancourt (2017)).

The Stan language is similar to Bayesian Inference Using Gibbs Sampling, or BUGS
(Lunn et al. (2000)) and JAGS (Plummer (2003)), which use Gibbs sampling instead of
Hamiltonian Monte Carlo. BUGS was used in (Morris (2016)), and has been
used for Bayesian reserving models by others
(Scollnik (2001), Verrall (2004), Zhang, Dukic, and Guszcza (2012)),
while (Schmid (2010), Meyers (2015)) use JAGS. Examples of reserving
models built in Stan can be found in (Cooney (2017), Gao (2018)).