# 6 Summary and future developments

In this paper we have presented a fully Bayesian modeling framework for the aggregated claims process to capture trends observed in paid and outstanding claims development data.

In Section 2 we outlined how to map the claims process to a system of differential equations from first principles to describe key dynamics. Using the basic building blocks of compartmental models, readers can extend and adjust the presented models to their own individual requirements.

In Section 3 we developed stochastic models for the claims process, describing the random nature of claims and latent underlying process parameters.

We showed how practitioners can utilize their expertise to describe the structure of underlying risk exposure profiles and corresponding parameter uncertainties. In addition, we highlighted the subtle but important difference between modeling incremental and cumulative claims payments.

This discussion culminated in a stochastic compartmental model, developed without reference to any particular data set, which was used to generate artificial prior predictive samples. These were used to test whether underlying model assumptions could produce data that bear a resemblance to actual observations. This is a critical aspect of the modeling process to understand model behavior. Note that the CAS Loss Simulator (CAS Loss Simulation Model Working Party (2018)) based on (Parodi (2014)), uses similar ideas for individual claims simulation.

In Section 4, the model was further extended to allow for fixed and varying parameters across grouping dimensions. Thanks to regularization we can incorporate many modeling parameters, while at the same time mitigating the risk of overfitting. Having fitted a model, we discussed the difference between the expected loss for a given accident year (i.e., the underlying latent mean loss) and the ultimate loss (i.e., actual cumulative claim payments to date, plus the sum of future claim payments). While the expected loss provides a means for us to challenge our model and has applications in pricing, the actual reserve is the key metric for financial reporting and capital setting.

The case study in Section 5 provided a practical guide to hierarchical compartmental model building. A work flow based on training and test data sets was outlined, which included model checking and improvement, and selection criteria. We introduced the concept of parameter variation by both accident year and development year, together with a method for incorporating market cycle information and explicit judgments into the modeling process.

Code snippets were shown throughout the document to illustrate how this modeling framework can be applied in practice using the brms interface to Stan from R. The succinct model notation language used by brms allows the user to test different models and structures quickly, including across several companies and/or lines of business, with or without explicit correlations.

Those familiar with probabilistic programming languages can write hierarchical compartmental reserving models directly in Stan (Carpenter et al. (2017)), PyMC (Salvatier, Wiecki, and Fonnesbeck (2016)), TensorFlow Probability (Abadi et al. (2015)), or other software.

Well-specified models with appropriate priors run within minutes on modern computers, and therefore hierarchical compartmental reserving models can be a part of the modern actuary's reserving toolbox. The transparency of model assumptions and ability to simulate claims process behavior provides a means of testing and challenging models in a visually intuitive manner.

Finally, as new data are collected, the Bayesian framework allows us to update our model and challenge its existing assumptions. If the posterior output changes significantly, then this should raise a call for action, either to investigate the model further or to challenge business assumptions.

## 6.1 Extensions

The framework and tools provided in this paper accommodate a wide range of modeling extensions, which may target ODE structures, statistical modeling assumptions, or both.

Examples of extensions that may warrant further investigation include the following:

- Double compartmental modelling of claim counts (IBNR – "incurred but not reported") and claims severity (IBNER – "incurred but not enough reported"). An approach to developing severity using a growth curve approach is given in (McNulty (2017))

- Using Gaussian processes in conjunction with compartmental models to model the stochastic loss ratio development process

- Mixture models which combine the compartmental approach with other parametric models, e.g. growth curves. Mixing proportions that vary by development age would provide greater flexibility to describe "nonstandard" average claims development patterns

### About the authors

**Markus Gesmann** is an analyst and data scientist with over 15 years' experience in the London Market. He is the maintainer of the ChainLadder (Gesmann et al. (2020)) and googleVis (Gesmann and de Castillo (2011)) R packages. Markus is the co-founder of the [Insurance Data Science conference](https://insurancedatascience.org/) series and the [Bayesian Mixer Meetups](https://www.meetup.com/Bayesian-Mixer-London/) in London. On his [blog](https://magesblog.com/) he published various implementations of different hierarchical loss reserving models in Stan and brms, including Jake Morris' hierarchical compartmental reserving models (Morris (2016)).

**Jake Morris** is an actuarial data scientist with 10 years' experience in predictive modeling and commercial insurance analytics. He has presented on Bayesian and hierarchical techniques at multiple international actuarial and data science conferences, and is the author of (Morris (2016)). Jake is a Fellow of the Institute and Faculty of Actuaries (FIA) and Certified Specialist in Predictive Analytics (CSPA).

### Acknowledgements

The authors would like to thank the following individuals for taking the time to review this work and provide valuable feedback: Michael Betancourt, Paul-Christian Bürkner, Dave Clark, Mick Cooney, Adam M. Gerdes, Michael Henk, Robert Hooley, Dan Murphy, Roland Ramsahai, and Mario Wüthrich.

They would also like to thank the Casualty Actuarial Society for sponsoring the research and providing periodic feedback throughout the process.

### References

Abadi, Martı́n, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, et al. 2015. "TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems." http://tensorflow.org/.

Carpenter, Bob, Andrew Gelman, Matthew Hoffman, Daniel Lee, Ben Goodrich, Michael Betancourt, Marcus Brubaker, Jiqiang Guo, Peter Li, and Allen Riddell. 2017. "Stan: A Probabilistic Programming Language." _Journal of Statistical Software, Articles_ 76 (1): 1–32. https://doi.org/10.18637/jss.v076.i01.

CAS Loss Simulation Model Working Party. 2018. "CASLoss Simulator 2.0." https://www.casact.org/research/lsmwp/index.cfm?fa=software.

Gesmann, Markus, and Diego de Castillo. 2011. "googleVis: Interface Between R and the Google Visualisation API." _The R Journal_ 3 (2): 40–44. https://journal.r-project.org/archive/2011-2/RJournal_2011-2_Gesmann+de~Castillo.pdf.

Gesmann, Markus, Dan Murphy, Wayne Zhang, Alessandro Carrato, Mario Wüthrich, and Fabio Concina. 2020. _ChainLadder: Statistical Methods and Models for Claims Reserving in General Insurance_. https://github.com/mages/ChainLadder.

McNulty, Gregory. 2017. "Severity Curve Fitting for Long Tailed Lines: An Application of Stochastic Processes and Bayesian Models." _Variance_ 11 (1): 118–32.

Morris, Jake. 2016. _Hierarchical Compartmental Models for Loss Reserving_. Casualty Actuarial Society Summer E-Forum; https://www.casact.org/pubs/forum/16sforum/Morris.pdf.

Parodi, Pietro. 2014. "Triangle-Free Reserving. A Non-Traditional Framework for Estimating Reserves and Reserve Uncertainty." _British Actuarial Journal_ 19 (1): 219–33. https://doi.org/10.1017/S1357321713000354.

Salvatier, John, Thomas V. Wiecki, and Christopher Fonnesbeck. 2016. "Probabilistic Programming in Python Using PyMC3." _PeerJ Computer Science_ 2 (April): e55. https://doi.org/10.7717/peerj-cs.55.