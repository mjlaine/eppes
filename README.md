# EPPES - ensemble prediction and parameter estimation system

This repository contains python code for EPPES system.

EPPES is a concept and an algorithmic method to include model parameter estimation in an ensemble prediction systems (EPS) used for probabilistic uncertainty assessment in operational numerical weather prediction (NWP). Such an [EPS system](http://www.ecmwf.int/products/forecasts/d/charts/medium/eps/) is used at the ECMWF, for example. The target is to estimate so-called closure parameters that represent sub-grid scale physical processes, such as boundary layer turbulence or cloud microphysics. The physical parametrization schemes are used to express unresolved variables by predefined parameters rather than by explicit modelling. EPPES is described in the following two research articles:

  * Järvinen H, Laine M, Solonen A, Haario H. 2011. *Ensemble prediction and parameter estimation system: the concept*, Quarterly Journal of the Royal Meteorological Society. [doi: 10.1002/qj.923](http://dx.doi.org/10.1002/qj.923)
  * Laine M, Solonen A, Haario H, Järvinen H. 2011. *Ensemble prediction and parameter estimation system: the method*, Quarterly Journal of the Royal Meteorological Society. [doi: 10.1002/qj.922](http://dx.doi.org/10.1002/qj.922)

The method for ensemble prediction and parameter estimation system (EPPES) is based on adding parameter perturbations to the ensemble members in addition to initial value and stochastic physics perturbations. Importance sampling algorithm is used to weight the proposed parameters in accordance to a suitable forecast skill score metric and an hierarchical statistical model is used to sequentially aggregate the information about the parameter uncertainty in the ensemble.

