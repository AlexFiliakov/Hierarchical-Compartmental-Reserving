# 2 Modeling the average claims development process

Many different approaches have been put forward to model the average claims development process. The most well-known is perhaps the nonparametric chain-ladder method, which uses average loss development factors to model loss emergence (Schmidt (2006)). Parametric approaches, such as growth curve models, have also been widely documented (Sherman (1984), Clark (2003), Guszcza (2008)).

## 2.1 Introduction to compartmental models

Compartmental models are a popular tool in many disciplines to describe the behavior and dynamics of interacting processes using differential equations.

Disciplines in which compartmental models are used include:

- Pharmaceutical sciences, to model how drugs interact with the body
- Electrical engineering, to describe the flow of electricity
- Biophysics, to explain the interactions of neurons
- Epidemiology, to understand the spread of diseases
- Biology, to describe the interaction of different populations

Each compartment typically relates to a different stage or population of the modeled process, usually described with its own differential equation.

## 2.2 Multicompartmental claims modeling

Similar to salt-mixing problem models, which describe the flow of fluids from one tank into another (Winkel (1994)), we can model the flow of information or monetary amounts between exposure, claims outstanding, and claims payment states for a group of policies.

![Schematic diagram of claims flow.](https://compartmentalmodels.gitlab.io/researchpaper/img/OriginalCompartmentFlow.png)

Figure 2.1: Schematic diagram of claims flow.

The diagram in Figure 2.1 gives a schematic view of three compartments ("tanks") and the flow of monetary amounts between them. We start with a "bucket" of exposure or premium, which outflows into a second bucket, labeled OS, for reported outstanding claims.

The parameter $k_{er}$ describes how quickly the exposure expires as claims are reported. For a group of risks, it is unlikely that 100% of exposure will convert to claims. Therefore a proportion, or multiple of exposure ($RLR$ = reported loss ratio), is assumed to convert to outstanding claim amounts.

Once claims have been processed, the insurer proceeds to pay their policyholders. The parameter $k_p$ describes the speed of claims settlement, and the parameter $RRF$ (reserve robustness factor) denotes the proportion of outstanding claims that are paid. An $RRF$ greater than 1 would indicate case underreserving, whereas an $RRF$ less than 1 would indicate case overreserving.

The set of compartments (the 'state-space') and claims process through them can be expressed with a set of ordinary differential equations (ODEs). Denoting the 'state-variables' $EX=$ exposure, $OS=$ outstanding claims, $PD=$ paid claims (i.e. the individual compartments) we have the following:

$$\begin{aligned}
dEX/dt &= -k_{er} \cdot EX \\
dOS/dt &= k_{er} \cdot RLR \cdot EX - k_p \cdot OS \\
dPD/dt &= k_p \cdot RRF \cdot OS
\end{aligned} \tag{2.1}$$

The initial conditions at time 0 are typically set as $EX(0) = \Pi$ (ultimate earned premiums), $OS(0) = 0, PD(0) = 0$ for accident period cohorts. Alternative approaches can be taken for policy year cohorts, which are discussed later in this section.

For exposure defined in terms of ultimate earned premium amounts, the parameters describe:

- **Rate of earning and reporting** ($k_{er}$): rate at which claim events occur and are subsequently reported to the insurer
- **Reported loss ratio** ($RLR$): proportion of exposure that becomes reported claims
- **Reserve robustness factor** ($RRF$): proportion of outstanding claims that are eventually paid
- **Rate of payment** ($k_p$): rate at which outstanding claims are paid

Here we assume that parameters are time independent, but in later sections we will allow for increased structural flexibility.

The expected loss ratio, $ELR$ (expected ultimate losses $\div$ ultimate premiums), can be derived as the product of $RLR$ and $RRF$ (the reported loss ratio scaled by the reserve robustness factor).

Setting parameters $k_{er} = 1.7$, $RLR = 0.8$, $k_p = 0.5$, $RRF = 0.95$ produces the output in Figure 2.2.

![Illustration of the different compartment amounts for a group of policies over time](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/odeSolve-1.png)

Figure 2.2: Illustration of the different compartment amounts for a group of policies over time

The autonomous system of ODEs above can be solved analytically by iterative integration:

$$\begin{aligned}
EX(t) &= \Pi \cdot \exp(-k_{er}t) \\
OS(t) &= \Pi \cdot RLR \cdot \frac{k_{er}}{k_{er} - k_p} \cdot (\exp(-k_p t) - \exp(-k_{er} t)) \\
PD(t) &= \frac{\Pi \cdot RLR \cdot RRF}{k_{er} - k_p} (k_{er} \cdot (1 - \exp(-k_p t) - k_p \cdot (1 - \exp(-k_{er} t))
\end{aligned} \tag{2.2}$$

The first equation describes an exponential decay of exposure over time.

Outstanding claims are modeled as the difference between two exponential decay curves with different time scales, which determines how the reported losses ($\Pi \cdot RLR$) are spread out over time and how outstanding losses decay as payments are made.

The paid curve is an integration of the outstanding. It represents a classic loss emergence pattern with two parameters, $k_{er}$ and $k_p$, multiplied by an expected ultimate claims cost, represented by the product $\Pi \cdot RLR \cdot RRF$.

The peak of the outstanding claims cost is at $t = \log(k_p/k_{er})/(k_p - k_{er})$, representing the inflection point in paid loss emergence. Note that the summation of $OS(t)$ and $PD(t)$ gives us the implied incurred losses at time t.

## 2.3 Two-stage outstanding compartmental model

We can increase the flexibility of the model in many ways, for example by introducing time-dependent parameters or adding one or more compartments to the model, as outlined in (Morris (2016)).

Adding compartments keeps our ODEs autonomous and easier to solve analytically, and easier to visualise in a single diagram.

The diagram in Figure 2.3 depicts a compartmental model that assumes reported claims fall into two categories: they are either dealt with by the insurance company quickly, with claims paid to policyholders in a timely fashion, or they go through another, more time-consuming process (such as investigation, dispute, and/or litigation).

![Two compartments for outstanding claims to allow some claims to be settled faster than others](https://compartmentalmodels.gitlab.io/researchpaper/img/TwoStageCompartmentFlow.png)

Figure 2.3: Two compartments for outstanding claims to allow some claims to be settled faster than others

We translate this diagram into a new set of ODEs:

$$\begin{aligned}
dEX/dt &= -k_{er} \cdot EX \\
dOS_1/dt &= k_{er} \cdot RLR \cdot EX - (k_{p1} + k_{p2}) \cdot OS_1 \\
dOS_2/dt &= k_{p2} \cdot (OS_1 - OS_2) \\
dPD/dt &= RRF \cdot (k_{p1} \cdot OS_1 + k_{p2} \cdot OS_2)
\end{aligned} \tag{2.3}$$

Solving the system of autonomous ODEs can be done iteratively, resulting in the solutions below. However, numerical solvers are typically preferred to reduce algebraic computation and minimise risk of error.

$$\begin{aligned}
EX(t) &= \Pi\exp(-k_{er}t) \\
OS_1(t) &= \Pi RLR \frac{k_{er}}{k_{er} - k_{p1} - k_{p2}} [\exp(-(k_{p1} + k_{p2})t) - \exp(-k_{er}t)] \\
OS_2(t) &= \Pi RLR \frac{k_{er}k_{p2}}{k_{p1}(k_{p2} - k_{er})(k_{er} - k_{p1} - k_{p2})} [\exp(-(k_{p1} + k_{p2})t)(k_{er} - k_{p2}) \\
&\quad - \exp(-k_{p2}t)(k_{er} - k_{p1} - k_{p2}) - \exp(-k_{er}t)k_{p1}] \\
PD(t) &= \frac{\Pi RLR \cdot RRF}{k_{p1}(k_{p2} - k_{er})(k_{er} - k_{p1} - k_{p2})} [(k_{p1}(k_{er}(k_{p1} - k_{er}) - k_{p2}(k_{p1} + k_{p2})) + 2k_{er}k_{p2}) \\
&\quad + \exp(-(k_{p1} + k_{p2})t)(k_{er}(k_{er}k_{p1} - k_{er}k_{p2} + k_{p2}^2 - k_{p1}k_{p2})) \\
&\quad + \exp(-k_{p2}t)(k_{er}k_{p2}(k_{er} - k_{p1} - k_{p2}) + \exp(-k_{er}t)(k_{p1}(k_{p1}k_{p2} + k_{p2}^2 - k_{er}k_{p1}))]
\end{aligned}$$

Plotting the solutions illustrates faster and slower processes for two distinct groups of outstanding claims, producing a paid curve that exhibits a steep start followed by a longer tail, shown in Figure 2.4.

Many datasets will not separate outstanding claims into different categories, in which case, the sum of $OS_1$ and $OS_2$ will be used for fitting purposes.

![Example of a two-stage outstanding model, with a portion of the claims settled at a faster rate (0.7) than others (0.5)](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/twostageexample-1.png)

Figure 2.4: Example of a two–stage outstanding model, with a portion of the claims settled at a faster rate (0.7) than others (0.5)

It is trivial to expand this approach by adding further compartments to allow for more than two distinct settlement processes. The next section introduces a multistage exposure compartment model in which the number of compartments becomes a variable itself.

## 2.4 Multistage exposure model

The models thus far have assumed an exponential decay of exposure over time. Furthermore, we have assumed the exposure at $t=0$ can be represented by ultimate earned premiums.

In reality, at $t=0$ we may expect some exposures to still be earning out (on an accident year basis) or not yet be written (on a policy year basis). If we have a view on how exposures have earned in the past and may earn into the future (e.g., from our business plan), then we can feed blocks of exposure into the compartmental system over development time (Morris (2016)), with the result that $k_{er}$ simplifies to $k_r$.

Alternatively, we can use a cascading compartmental model to allow for different earning and reporting processes as part of the modeling process, as in Figure 2.5.

![Schematic of a multistage transition model](https://compartmentalmodels.gitlab.io/researchpaper/img/GammaCompartmentFlow2.png)

Figure 2.5: Schematic of a multistage transition model

We assume that risks are written and earned at a constant rate, analogous to water flowing from one tank to the next at a constant rate. The claims reporting delay is then modeled by the number of different compartments.

We can express this compartmental model as a system of $\alpha$ ODEs:

$$\begin{aligned}
\dot{g}_1 &= -\beta g_1 \\
\dot{g}_2 &= \beta g_1 - \beta g_2 \\
\vdots &= \beta g_1 - \beta \vdots \\
\dot{g}_\alpha &= \beta g_{\alpha-1} - \beta g_\alpha
\end{aligned}$$

More succinctly, we express the system as an ODE of $\alpha$ order:

$$\frac{d^\alpha}{dt^\alpha} g(t; \alpha, \beta) = -\sum_{i=1}^\alpha \binom{\alpha}{i} \beta^i \frac{d^{(\alpha-i)}}{dt^{(\alpha-i)}} g(t; \alpha, \beta),$$

For $\alpha = 1$ we get back to an exponential decay model.

This ODE can be solved analytically (Gesmann (2002)):

$$g(t; \alpha, \beta) = \begin{cases}
\frac{\beta^\alpha t^{\alpha-1}}{(\alpha-1)!} e^{-\beta t} & t \geq 0 \\
0 & t < 0
\end{cases}$$

Relaxing the assumption of $\alpha$ is a positive integer gives

$$g(t; \alpha, \beta) = \frac{\beta^\alpha t^{\alpha-1}}{\Gamma(\alpha)} e^{-\beta t} \text{ for } t \geq 0,$$

with the gamma function $\Gamma(\alpha) = \int_0^\infty x^{\alpha-1} e^{-x} dx$.

The trained eye may recognize that $g(t; \alpha, \beta)$ is the probability density function (PDF) of the gamma distribution, which is commonly used to model waiting times.

Finally, let's imagine we collect the outflowing water in another tank, with amounts in this compartment calculated by integrating $g(t; \alpha, \beta)$ over time. This integration results in a gamma cumulative distribution function (CDF),

$$G(t; \alpha, \beta) = \int_0^t g(x; \alpha, \beta) dx = \frac{\gamma(\alpha, t\beta)}{\Gamma(\alpha)} \text{ for } t \geq 0,$$

using the incomplete gamma function $\gamma(s,t) = \int_0^x t^{(s-1)} e^{-t} dt$.

Visualizing the functions $g(t; \alpha, \beta)$ and $G(t; \alpha, \beta)$ shows that for a fixed $\beta$ the parameter $\alpha$ can be used to determine how quickly the curves converge, see Figure 2.6.

![Visualization of the Gamma function for different values of $\alpha$](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/alphafunction-1.png)

Figure 2.6: Visualization of the Gamma function for different values of $\alpha$

The gamma function has previously been proposed to model loss emergence patterns, labeled as the 'Hoerl curve' (England and Verrall (2001)).

For our purpose of modeling exposure decay, we introduce parameters $k_e$, describing the earning speed, and $d_r$, describing the reporting delay.

We define $k_e = \beta$ and $d_r = \alpha$; which implies that the speed of earning $k_e$ matches the flow of water from one tank into the next, while $d_r$ can be linked to the number of transient tanks.

The analytical form for the exposure can be then expressed as

$$EX(t) = \Pi \frac{k_e^{d_r} t^{d_r-1}}{\Gamma(d_r)} e^{-k_e t}$$

In other words, we model exposure as ultimate earned premium ($\Pi$) weighted over time with a gamma PDF.

Inserting the solution into the ODEs produces the following:

$$\begin{aligned}
dOS_1/dt &= \Pi \cdot RLR \cdot \frac{k_e^{d_r} t^{d_r-1}}{\Gamma(d_r)} e^{-k_e t} - (k_{p1} + k_{p2}) \cdot OS_1 \\
dOS_2/dt &= k_{p2} \cdot (OS_1 - OS_2) \\
dPD/dt &= RRF \cdot (k_{p1} \cdot OS_1 + k_{p2} \cdot OS_2)
\end{aligned} \tag{2.4}$$

Figure 2.7 illustrates the impact the multistage exposure submodel has on the two-stage outstanding curves and paid loss emergence pattern. Claim reports and payments develop more slowly, as typically observed for longer-tailed business lines.

![Example of multi-stage exposure model ($k_e=3$, $d_r=1.7$) with a two-stage outstanding process](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/twostageexample2-1.png)

Figure 2.7: Example of multi-stage exposure model ($k_e = 3$, $d_r = 1.7$) with a two–stage outstanding process

Note that with the proposed extensions, setting $d_r = 1$ and $k_{p2} = 0$ gets us back to the original model (2.1), with one parameter, $k_{er}$, for the exposure and reporting process, and one parameter, $k_p$, for the payment process.

We can express our model as an autonomous ODE system for various extensions, but integrating the system is not always straightforward. Fortunately, as we will see in a later section, Stan (Stan Development Team (2019)) has a Runge-Kutta solver to integrate the system numerically.

It is worth emphasizing that this framework allows us to build parametric curves that share parameters across paid and outstanding data. This enables us to learn from both data sources at the same time and have consistent paid and incurred ultimate projections (see the phase plots in Figure 2.8). This is in contrast to fitting separate curves for paid and incurred data, resulting in two different answers.

![The 3-D plot (left) illustrates that models 1 and 2 assume exposure to peak at t = 0, while model 3 assumes exposure to be 0 at t = 0, with gradual increase and decrease over time. Note that for models 2 and 3 OS displays the sum of OS1 + OS2. The 2-D plot (right) shows the relationship between outstanding and paid claims, which can be compared against actual data.](https://compartmentalmodels.gitlab.io/researchpaper/Compartmental_Reserving_Models_files/figure-html/3dplotmodels-1.png)

Figure 2.8: The 3-D plot (left) illustrates that models 1 and 2 assume exposure to peak at t = 0, while model 3 assumes exposure to be 0 at t = 0, with gradual increase and decrease over time. Note that for models 2 and 3 OS displays the sum of OS1 + OS2. The 2-D plot (right) shows the relationship between outstanding and paid claims, which can be compared against actual data.

In practice, for some business lines, claim characteristics can be heterogeneous or case handling processes inconsistent, resulting in volatile outstanding claims development. The value of incorporating outstandings may be diminished if the data do not broadly conform to the model's assumption on how outstandings and payments link.

Similarly, for the model assumptions to hold, the process from earning exposure through to paying claims should be approximable as continuous for a volume of policies.

In summary, compartmental models provide a flexible framework to build loss emergence patterns for outstanding and paid claims from first principles. We outline two extensions here, yet many more are feasible depending on the underlying features the practitioner is hoping to build within a structural model for the average development process.

Getting a "feel" for the parameters, their interpretations, and how they determine loss emergence characteristics in each case will become important when we have to set prior distributions for them.

### References

Clark, David R. 2003. _LDF Curve-Fitting and Stochastic Reserving: A Maximum Likelihood Approach_. Casualty Actuarial Society; http://www.casact.org/pubs/forum/03fforum/03ff041.pdf.

England, Peter D, and Richard J Verrall. 2001. "A Flexible Framework for Stochastic Claims Reserving." _Proceedings of the Casualty Actuarial Society_ 88 (1): 1–38. https://www.casact.org/pubs/proceed/proceed01/01001.pdf.

Gesmann, Markus. 2002. "Modellierung Und Analyse Neuronaler Dynamiken (Modelling and Analysing Neuronal Dynamics)." Master's thesis, University of Cologne.

Guszcza, James. 2008. "Hierarchical Growth Curve Models for Loss Reserving." In _Casualty Actuarial Society e-Forum, Fall 2008_, 146–73. https://www.casact.org/pubs/forum/08fforum/7Guszcza.pdf.

Morris, Jake. 2016. _Hierarchical Compartmental Models for Loss Reserving_. Casualty Actuarial Society Summer E-Forum; https://www.casact.org/pubs/forum/16sforum/Morris.pdf.

Schmidt, Klaus D. 2006. "Methods and Models of Loss Reserving Based on Run-Off Triangles : A Unifying Survey." _CAS Fall Forum_, 269–317. https://www.math.tu-dresden.de/sto/schmidt/publications_online/2006-cas-method.pdf.

Sherman, Richard E. 1984. "Extrapolating, Smoothing, and Interpolating Development Factors." _Proceedings of the Casualty Actuarial Society_ LXXI (135,136): 122–55.

Stan Development Team. 2019. "RStan: The R Interface to Stan." http://mc-stan.org/.

Winkel, Brian J. 1994. "Modelling Mixing Problems with Differential Equations Gives Rise to Interesting Questions." _International Journal of Mathematical Education in Science and Technology_ 25 (1): 55–60. https://doi.org/10.1080/0020739940250107.