# R to Python Conversion Plan: Hierarchical Compartmental Reserving Model

## Overview
This document outlines the conversion plan for migrating the R-based hierarchical compartmental reserving model to Python using Stan. The goal is to maintain the same statistical methodology while leveraging Python's ecosystem for data manipulation and visualization.

## 1. Analysis of Current R Implementation

### 1.1 Core Components
The R notebook implements a Bayesian hierarchical model for insurance loss reserving with the following key elements:

1. **Compartmental ODE Model**: Three compartments (Exposure, Outstanding, Paid) with transition rates
2. **Hierarchical Structure**: Random effects for accident years
3. **Bayesian Framework**: Using brms (R interface to Stan) with custom Stan functions
4. **Prior Predictive Checks**: Simulations to validate prior distributions
5. **Visualization**: Fan charts showing uncertainty quantiles

### 1.2 Key Dependencies
- `data.table`: High-performance data manipulation
- `brms`: Bayesian regression models using Stan
- `cmdstanr`: Interface to CmdStan
- `ggplot2` + `ggdist`: Advanced statistical visualization
- `repr`: Plot rendering in notebooks

### 1.3 Model Specifications
- **Development periods**: 48 months
- **Accident years**: 2024-2027
- **Rate indices**: [1, 1.1, 1.05, 0.95]
- **Two claim types**: Outstanding and Paid
- **Log-normal likelihood** with hierarchical priors

## 2. Python Package Equivalents

| R Package | Python Equivalent | Notes |
|-----------|------------------|-------|
| data.table | pandas | Standard data manipulation library |
| brms | CmdStanPy/PyStan | Direct Stan interface (no brms equivalent) |
| cmdstanr | CmdStanPy | Python interface to CmdStan |
| ggplot2 | plotnine/matplotlib | plotnine for ggplot-style, matplotlib for custom |
| ggdist | matplotlib/seaborn | Custom implementation needed for fan charts |
| repr | IPython.display | Built into Jupyter |

## 3. Multi-Step Implementation Plan

### Phase 1: Environment Setup
1. **Install dependencies**
   - CmdStanPy for Stan interface
   - pandas for data manipulation
   - numpy/scipy for numerical operations
   - matplotlib/seaborn for visualization
   - arviz for Bayesian analysis utilities

### Phase 2: Data Preparation
1. **Recreate data structures**
   - Convert data.table operations to pandas
   - Maintain the same data schema
   - Implement the cycle index logic

### Phase 3: Stan Model Translation
1. **Extract Stan code from brms**
   - The ODE system is already in Stan
   - Need to manually write the full Stan model (brms generates this automatically)
   - Implement the hierarchical structure explicitly

### Phase 4: Model Fitting
1. **Replace brms workflow**
   - Write Stan model file
   - Use CmdStanPy for compilation and sampling
   - Implement prior predictive sampling

### Phase 5: Visualization
1. **Recreate fan charts**
   - Implement quantile-based ribbon plots
   - Match the faceted layout
   - Maintain color schemes and styling

## 4. Implementation Subtasks

### 4.1 Data Preparation Tasks
- [ ] Create pandas DataFrames for cycle index and main data
- [ ] Implement data expansion for Outstanding/Paid
- [ ] Add rate index merging logic

### 4.2 Stan Model Tasks
- [ ] Write complete Stan model file
- [ ] Define data block matching R structure
- [ ] Implement parameters and transformed parameters
- [ ] Code the model block with hierarchical priors
- [ ] Add generated quantities for predictions

### 4.3 Python Interface Tasks
- [ ] Create model compilation wrapper
- [ ] Implement prior predictive sampling
- [ ] Extract and process posterior samples
- [ ] Calculate derived quantities (cumulative values)

### 4.4 Visualization Tasks
- [ ] Implement fan chart function
- [ ] Create faceted plot layout
- [ ] Add individual simulation traces
- [ ] Match color schemes and labels

## 5. Approach: Direct Translation
**Pros:**
- Maintains exact statistical methodology
- Easier to validate against R results
- Preserves hierarchical structure

**Cons:**
- Requires manual Stan model writing (brms automates this)
- More verbose than brms formula syntax
- No automatic prior predictive checks

## 6. Key Technical Challenges

### 6.1 ODE Integration in Stan
- The `ode_rk45` function syntax differs between Stan versions
- Need to ensure compatibility with Python's Stan interface
- May need to adjust tolerances for numerical stability

### 6.2 Formula Interface
- brms provides a high-level formula interface
- Python requires explicit Stan model specification
- Need to carefully translate the non-linear formula structure

### 6.3 Fan Chart Implementation
- ggdist provides sophisticated uncertainty visualization
- Will need custom matplotlib implementation
- Consider using arviz for some plotting utilities

## 7. Validation Strategy

1. **Numerical Validation**
   - Compare prior predictive distributions
   - Check parameter estimates match within MCMC error
   - Validate ODE solutions independently

2. **Visual Validation**
   - Ensure plots show same patterns
   - Match uncertainty intervals
   - Verify facet structure preserved

## 8. File Structure

```
Hierarchical_Compartmental_Reserving/
├── planning/
│   └── convert_to_python.md (this file)
├── src/
│   ├── reserving.ipynb (original R)
│   ├── reserving_python.ipynb (new Python version)
│   └── models/
│       └── hierarchical_compartmental.stan
└── outputs/
    ├── figures/
    └── results/
```

## 9. Next Steps

1. Begin with Phase 1: Environment setup
2. Create the Stan model file first (most complex part)
3. Continue with Phase 2: implement data preparation
4. Continue with Phase 3: Stan Model translation
5. Continue with Phase 4: implement model fitting
6. Finish with Phase 5: Visualization

## 10. Detailed Stan Model Specification

### 10.1 Stan Model Structure
The complete Stan model will need the following blocks:

```stan
functions {
  // ODE system for compartmental model
  vector compartmentmodel(real t, vector y, vector theta) {
    vector[3] dydt;
    dydt[1] = -theta[1] * y[1];                           // exposure 
    dydt[2] = theta[1] * theta[3] * y[1] - theta[2] * y[2]; // outstanding
    dydt[3] = theta[2] * theta[4] * y[2];                   // paid
    return dydt;
  }
  
  // Incremental claims process function
  real incrclaimsprocess(real t, real devfreq, real ker, real kp, 
                         real RLR, real RRF, real delta) {
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
    
    if ((delta > 0) && (t > devfreq)) { // paid greater dev period 1
      // incremental paid
      y = ode_rk45(compartmentmodel, y0, 0, rep_array(t - devfreq, 1), theta);
      out = out - y[1, 3];
    }
    return(out);
  }
}

data {
  int<lower=1> N;                    // Number of observations
  int<lower=1> n_accident_years;     // Number of accident years
  array[N] int accident_year_id;    // Accident year index for each obs
  vector[N] dev;                     // Development period
  vector[N] delta;                   // 0 for outstanding, 1 for paid
  vector[n_accident_years] RateIndex; // Rate indices by accident year
  vector[N] loss_ratio;              // Response variable
}

parameters {
  // Population-level parameters
  real oker;                         // Log-scale parameter for ker
  real okp;                          // Log-scale parameter for kp
  
  // Hierarchical parameters
  real oRLR_mu;                      // Population mean for RLR
  real oRRF_mu;                      // Population mean for RRF
  real<lower=0> oRLR_sd;            // Population SD for RLR
  real<lower=0> oRRF_sd;            // Population SD for RRF
  
  // Random effects
  vector[n_accident_years] oRLR_raw; // Non-centered parameterization
  vector[n_accident_years] oRRF_raw; // Non-centered parameterization
  
  // Correlation
  cholesky_factor_corr[2] L_Omega;   // Cholesky factor of correlation matrix
  
  // Observation-level parameters
  real<lower=0> sigma_outstanding;   // SD for outstanding claims
  real<lower=0> sigma_paid;          // SD for paid claims
}

transformed parameters {
  vector[n_accident_years] oRLR;
  vector[n_accident_years] oRRF;
  matrix[2, n_accident_years] random_effects;
  
  // Non-centered parameterization for random effects
  random_effects[1] = oRLR_raw;
  random_effects[2] = oRRF_raw;
  random_effects = diag_pre_multiply([oRLR_sd, oRRF_sd], L_Omega) * random_effects;
  
  oRLR = oRLR_mu + random_effects[1]';
  oRRF = oRRF_mu + random_effects[2]';
}

model {
  // Priors
  oker ~ normal(0, 1);
  okp ~ normal(0, 1);
  oRLR_mu ~ normal(0, 1);
  oRRF_mu ~ normal(0, 1);
  oRLR_sd ~ student_t(10, 0, 0.05);
  oRRF_sd ~ student_t(10, 0, 0.05);
  oRLR_raw ~ std_normal();
  oRRF_raw ~ std_normal();
  L_Omega ~ lkj_corr_cholesky(1);
  sigma_outstanding ~ lognormal(log(0.05), 0.02);
  sigma_paid ~ lognormal(log(0.1), 0.05);
  
  // Likelihood
  for (n in 1:N) {
    real ker = 0.1 * exp(oker * 0.05);
    real kp = 0.5 * exp(okp * 0.025);
    real RLR = 0.55 * exp(oRLR[accident_year_id[n]] * 0.025) / RateIndex[accident_year_id[n]];
    real RRF = exp(oRRF[accident_year_id[n]] * 0.05);
    
    real eta = incrclaimsprocess(dev[n], 1.0, ker, kp, RLR, RRF, delta[n]);
    real sigma = delta[n] == 0 ? sigma_outstanding : sigma_paid;
    
    loss_ratio[n] ~ lognormal(log(eta), sigma);
  }
}

generated quantities {
  // For posterior predictive checks
  vector[N] loss_ratio_rep;
  
  for (n in 1:N) {
    real ker = 0.1 * exp(oker * 0.05);
    real kp = 0.5 * exp(okp * 0.025);
    real RLR = 0.55 * exp(oRLR[accident_year_id[n]] * 0.025) / RateIndex[accident_year_id[n]];
    real RRF = exp(oRRF[accident_year_id[n]] * 0.05);
    
    real eta = incrclaimsprocess(dev[n], 1.0, ker, kp, RLR, RRF, delta[n]);
    real sigma = delta[n] == 0 ? sigma_outstanding : sigma_paid;
    
    loss_ratio_rep[n] = lognormal_rng(log(eta), sigma);
  }
}
```

### 10.2 Key Conversion Examples

#### Data Preparation (R to Python)
```r
# R version
DT1 <- data.table(
    accident_year = rep(ay_start:(ay_start + ay_n - 1), each = dev_n),
    dev = rep(1:dev_n, ay_n),
    loss_ratio = 1
)
```

```python
# Python version
import pandas as pd
import numpy as np

DT1 = pd.DataFrame({
    'accident_year': np.repeat(np.arange(ay_start, ay_start + ay_n), dev_n),
    'dev': np.tile(np.arange(1, dev_n + 1), ay_n),
    'loss_ratio': 1
})
```

#### Model Fitting (brms to CmdStanPy)
```r
# R version
prior_model <- brm(
    frml, prior = mypriors, data = myDT, 
    family = brmsfamily("lognormal", link_sigma = "log"),
    stanvars = stanvar(scode = claims_dynamics, block = "functions"),
    backend = "cmdstan", sample_prior = "only", 
    seed = 123, iter = nSims, chains = nChains
)
```

```python
# Python version
import cmdstanpy as cmdstan

# Compile model
model = cmdstan.CmdStanModel(
    stan_file='models/hierarchical_compartmental.stan'
)

# Prepare data
stan_data = {
    'N': len(myDT),
    'n_accident_years': n_accident_years,
    'accident_year_id': myDT['accident_year_id'].values,
    'dev': myDT['dev'].values,
    'delta': myDT['delta'].values,
    'RateIndex': rate_indices,
    'loss_ratio': myDT['loss_ratio'].values
}

# Sample from prior only
prior_fit = model.sample(
    data=stan_data,
    seed=123,
    chains=nChains,
    iter_warmup=nSims//2,
    iter_sampling=nSims//2,
    fixed_param=True  # For prior predictive
)
```

#### Fan Chart Visualization
```python
# Python implementation of fan charts
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plot_fan_chart(df, x_col, y_col, group_cols, quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    
    groups = df.groupby(group_cols)
    
    for idx, (name, group) in enumerate(groups):
        ax = axes[idx]
        
        # Calculate quantiles
        quantile_data = group.groupby(x_col)[y_col].quantile(quantiles).unstack()
        x = quantile_data.index
        
        # Plot ribbons from outer to inner
        for i in range(len(quantiles)//2):
            lower = quantiles[i]
            upper = quantiles[-(i+1)]
            ax.fill_between(x, quantile_data[lower], quantile_data[upper], 
                          alpha=0.15, color='blue', linewidth=0)
        
        # Plot median
        median = group.groupby(x_col)[y_col].median()
        ax.plot(x, median, color='darkblue', linewidth=2)
        
        # Add title
        ax.set_title(f"{name[0]}\n{name[1]}")
        
    plt.tight_layout()
    return fig
```

## 11. Critical Implementation Notes

### 11.1 ODE Solver Considerations
- Stan's `ode_rk45` requires specific array formats
- Python data must be converted to Stan-compatible arrays
- Consider using `ode_bdf` for stiff systems if convergence issues arise

### 11.2 Prior Predictive Sampling
- Unlike brms, we need to manually implement prior predictive sampling
- Can either use `fixed_param=True` or create a separate Stan model
- Ensure random seed consistency for reproducibility

### 11.3 Hierarchical Structure
- The non-centered parameterization is crucial for sampling efficiency
- Correlation structure must be carefully implemented
- Consider monitoring divergences and energy diagnostics

## 12. Additional Implementation Examples

### 12.1 Complete Data Preparation Pipeline
```python
import pandas as pd
import numpy as np

# Constants
dev_n = 48
ay_start = 2024
ay_end = 2027
ay_n = ay_end - ay_start + 1

# Create cycle index
CycleIndex = pd.DataFrame({
    'accident_year': np.arange(ay_start, ay_end + 1),
    'RateIndex': [1, 1.1, 1.05, 0.95]
})

# Create base data
DT1 = pd.DataFrame({
    'accident_year': np.repeat(np.arange(ay_start, ay_end + 1), dev_n),
    'dev': np.tile(np.arange(1, dev_n + 1), ay_n),
    'loss_ratio': 1
})

# Expand for Outstanding/Paid
DT2_outstanding = DT1.copy()
DT2_outstanding['delta'] = 0
DT2_outstanding['deltaf'] = 'Outstanding'

DT2_paid = DT1.copy()
DT2_paid['delta'] = 1
DT2_paid['deltaf'] = 'Paid'

DT2 = pd.concat([DT2_outstanding, DT2_paid], ignore_index=True)

# Merge with rate index
myDT = DT2.merge(CycleIndex, on='accident_year')

# Add accident year ID for Stan
myDT['accident_year_id'] = myDT['accident_year'] - ay_start + 1
```

### 12.2 Prior Predictive Sampling Implementation
```python
def sample_prior_predictive(model, stan_data, n_samples=1000, chains=4):
    """Sample from prior predictive distribution"""
    
    # Create modified Stan model for prior sampling
    prior_model_code = """
    generated quantities {
        // Sample parameters from priors
        real oker = normal_rng(0, 1);
        real okp = normal_rng(0, 1);
        real oRLR_mu = normal_rng(0, 1);
        real oRRF_mu = normal_rng(0, 1);
        real<lower=0> oRLR_sd = student_t_rng(10, 0, 0.05);
        real<lower=0> oRRF_sd = student_t_rng(10, 0, 0.05);
        real<lower=0> sigma_outstanding = lognormal_rng(log(0.05), 0.02);
        real<lower=0> sigma_paid = lognormal_rng(log(0.1), 0.05);
        
        // Generate predictions
        vector[N] loss_ratio_prior;
        // ... (implementation)
    }
    """
    
    # Alternative: Use fixed_param sampling
    prior_fit = model.sample(
        data=stan_data,
        fixed_param=True,
        iter_sampling=n_samples,
        chains=chains,
        seed=123
    )
    
    return prior_fit
```

### 12.3 Cumulative Claims Calculation
```python
def calculate_cumulative_claims(sim_data):
    """Convert incremental to cumulative claims"""
    
    # For paid claims: cumulative sum
    paid_mask = sim_data['deltaf'] == 'Paid'
    sim_data.loc[paid_mask, 'CumSim'] = (
        sim_data[paid_mask]
        .groupby(['accident_year', 'SimID'])['Sim']
        .cumsum()
    )
    
    # For outstanding: keep as is
    outstanding_mask = sim_data['deltaf'] == 'Outstanding'
    sim_data.loc[outstanding_mask, 'CumSim'] = sim_data.loc[outstanding_mask, 'Sim']
    
    # Calculate incurred (paid + outstanding)
    incurred = (
        sim_data.groupby(['accident_year', 'SimID', 'dev'])
        .agg({'CumSim': 'sum'})
        .reset_index()
        .rename(columns={'CumSim': 'Incurred'})
    )
    
    # Update outstanding to show incurred
    sim_data = sim_data.merge(incurred, on=['accident_year', 'SimID', 'dev'], how='left')
    sim_data.loc[outstanding_mask, 'CumSim'] = sim_data.loc[outstanding_mask, 'Incurred']
    
    # Create new grouping variable
    sim_data['deltaf2'] = sim_data['deltaf'].map({'Paid': 'Paid', 'Outstanding': 'Incurred'})
    
    return sim_data
```

### 12.4 Enhanced Fan Chart with Individual Traces
```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

def plot_enhanced_fan_chart(sim_data, n_traces=3):
    """Create fan chart with uncertainty bands and individual traces"""
    
    # Set up the plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # Color palette for traces
    trace_colors = sns.color_palette("YlOrRd", n_traces)
    
    # Quantiles for fan
    quantiles = [0.05, 0.1, 0.25, 0.4, 0.6, 0.75, 0.9, 0.95]
    alphas = [0.15, 0.2, 0.25, 0.3, 0.3, 0.25, 0.2, 0.15]
    
    # Group by deltaf and accident_year
    groups = sim_data.groupby(['deltaf', 'accident_year'])
    
    for idx, ((deltaf, ay), group) in enumerate(groups):
        ax = axes[idx]
        
        # Calculate quantiles
        quantile_df = (
            group.groupby('dev')['Sim']
            .quantile(quantiles)
            .unstack()
            .reset_index()
        )
        
        # Plot ribbons
        for i in range(len(quantiles)//2):
            ax.fill_between(
                quantile_df['dev'],
                quantile_df[quantiles[i]],
                quantile_df[quantiles[-(i+1)]],
                alpha=alphas[i],
                color='blue',
                linewidth=0
            )
        
        # Plot median
        median = group.groupby('dev')['Sim'].median()
        ax.plot(median.index, median.values, 'darkblue', linewidth=2)
        
        # Plot individual traces
        for i in range(n_traces):
            trace_data = group[group['SimID'] == i + 1]
            ax.plot(trace_data['dev'], trace_data['Sim'], 
                   color=trace_colors[i], alpha=0.8, label=f'Sim {i+1}')
        
        # Formatting
        ax.set_title(f'{deltaf}\n{ay}')
        ax.set_xlabel('Month')
        ax.set_ylabel('Incremental loss ratio')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        if idx == 3:  # Add legend to one subplot
            ax.legend(loc='upper right')
    
    plt.suptitle('Prior predictive loss ratio distribution', fontsize=16)
    plt.tight_layout()
    return fig
```

### 12.5 Model Diagnostics Wrapper
```python
import arviz as az

def diagnose_model_fit(fit, param_names=None):
    """Run comprehensive diagnostics on Stan fit"""
    
    # Convert to ArviZ InferenceData
    idata = az.from_cmdstanpy(fit)
    
    # Check convergence
    print("Convergence Diagnostics:")
    print("-" * 40)
    
    # R-hat
    rhat = az.rhat(idata)
    print(f"Max R-hat: {rhat.max().values:.3f}")
    
    # ESS
    ess = az.ess(idata)
    print(f"Min ESS bulk: {ess['ess_bulk'].min().values:.0f}")
    print(f"Min ESS tail: {ess['ess_tail'].min().values:.0f}")
    
    # Divergences
    divergences = fit.method_variables()['divergent__'].sum()
    print(f"Number of divergences: {divergences}")
    
    # Energy
    energy = fit.method_variables()['energy__']
    print(f"Mean energy: {energy.mean():.1f}")
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Trace plots for key parameters
    if param_names:
        az.plot_trace(idata, var_names=param_names, axes=axes)
    
    return idata
```

## 13. Testing and Validation Framework

### 13.1 Unit Tests for Key Components
```python
import pytest
import numpy.testing as npt

def test_ode_solution():
    """Test ODE solver matches R implementation"""
    # Test parameters
    t = 10
    ker = 0.1
    kp = 0.5
    RLR = 0.55
    RRF = 1.0
    
    # Expected values from R
    expected_outstanding = 0.123  # From R implementation
    expected_paid = 0.456
    
    # Run Python implementation
    result = incrclaimsprocess(t, 1.0, ker, kp, RLR, RRF, delta=0)
    
    npt.assert_allclose(result, expected_outstanding, rtol=1e-4)

def test_data_preparation():
    """Test data preparation matches R output"""
    # Create test data
    myDT = prepare_data()
    
    # Check dimensions
    assert len(myDT) == 48 * 4 * 2  # dev_n * ay_n * 2 (outstanding/paid)
    
    # Check rate indices
    assert all(myDT[myDT['accident_year'] == 2024]['RateIndex'] == 1.0)
    assert all(myDT[myDT['accident_year'] == 2025]['RateIndex'] == 1.1)
```

### 13.2 Visual Comparison Script
```python
def compare_with_r_output(python_results, r_results_path):
    """Compare Python results with saved R outputs"""
    
    # Load R results
    r_results = pd.read_csv(r_results_path)
    
    # Compare distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Python results
    ax1.hist(python_results['loss_ratio_sim'], bins=50, alpha=0.5, label='Python')
    ax1.set_title('Python Prior Predictive')
    
    # R results
    ax2.hist(r_results['loss_ratio_sim'], bins=50, alpha=0.5, label='R', color='orange')
    ax2.set_title('R Prior Predictive')
    
    # Statistical comparison
    from scipy import stats
    ks_stat, p_value = stats.ks_2samp(python_results['loss_ratio_sim'], 
                                      r_results['loss_ratio_sim'])
    
    print(f"KS test: statistic={ks_stat:.4f}, p-value={p_value:.4f}")
    
    return fig
```

## 14. Performance Optimization Tips

1. **Vectorize Stan Operations**: Use vectorized statements in Stan where possible
2. **Efficient Data Structures**: Use NumPy arrays for large data operations
3. **Parallel Chains**: Leverage CmdStanPy's parallel chain capability
4. **Caching**: Cache compiled Stan models to avoid recompilation
5. **Memory Management**: Process results in chunks for large simulations

## 15. Estimated Timeline (Revised)

- Environment setup: 1 hour
- Stan model development: 4-5 hours (including debugging)
- Data preparation: 2 hours
- Model fitting interface: 2-3 hours
- Visualization: 4-5 hours (fan charts are complex)
- Testing and validation: 3-4 hours
- Documentation and cleanup: 1-2 hours

**Total estimate: 17-22 hours**