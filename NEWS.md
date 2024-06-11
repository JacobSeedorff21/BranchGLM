BranchGLM News
================

# BranchGLM 2.1.6

- Adding plot method for BranchGLM objects
- Fixing another very small bug in branch and bound algorithms
- Fixing bug in log-likelihood calculation for gaussian models when the
  number of observations is odd

# BranchGLM 2.1.5

- Fixing a small bug with parallel computation for forward and switch
  branch and bound algorithms.
- Modifying stepwise variable selection algorithms to return the whole
  solution path which can now be accessed with `predict()` or `coef()`
  and can be plotted with `plot()`

# BranchGLM 2.1.4

- Updating all documentation and adding more tests
- Fixing a couple of bugs

# BranchGLM 2.1.3

- Updating predict functions to include na.action and to better handle
  offset variables.
- The `VariableSelection()` function now returns the coefficients as
  part of the selection process, so `fit()` doesn’t need to be used for
  every call to `coef()` or `predict()`.
- Adding new cols arguments to `plot.summary.BranchGLMVS()` and
  `plot.BranchGLMVS()` to allow for control over the colors

# BranchGLM 2.1.2

- Adding new cex arguments to `plot.summary.BranchGLMVS()` and
  `plot.BranchGLMVS()` to allow for more control over text size
- Fixing bug in the “variables” plot from `plot.summary.BranchGLMVS()`
  that resulted in one of the variables being removed
- Fixing bug in root-finding algorithm for finding confidence intervals
  with `confint.BranchGLM()`

# BranchGLM 2.1.1

- Adding multiple new arguments to `plot.summary.BranchGLMVS()` and
  `plot.BranchGLMVS()`
- Changing estimate of dispersion for gaussian GLMs to be the MLE
  instead of the unbiased estimator
- Adding keepintercept argument to `VariableSelection()` to specify
  whether the intercept should be kept or not
- Removing standard errors and p-values from output of `fit()` since
  these are biased due to the variable selection process

# BranchGLM 2.1.0

- Fixing bug in `VariableSelection()` when using formulas of the form
  `y ~ . - variable`
- Changing default type for `VariableSelection()` to be “branch and
  bound”
- Adding vcov and confint methods for `BranchGLM` objects
- Adding plot method for objects resulting from `confint.BranchGLM()`
- Adding coef and predict methods for `BranchGLMVS` objects
- Fixing bug that caused the switch branch and bound type in
  `VariableSelection()` to be slower

# BranchGLM 2.0.1

- Fixing bug in `VariableSelection()` when using the switch branch and
  bound method where duplicate models are returned
- Fixing bug in `VariableSelection()` when using the switch branch and
  bound method or branch and bound method where factor variables were
  handled incorrectly
- No longer allowing models that failed to converge to be one of the top
  models in the `VariableSelection()` function

# BranchGLM 2.0.0

- Added the following features to the `VariableSelection()` function
  - Finding the top k models according to the metric via the bestmodels
    argument
  - Finding all models with a metric value within k of the best metric
    value via the cutoff argument
  - Added HQIC as a possible metric, HQIC is the Hannan-Quinn
    information criterion
- Added summary method for `BranchGLMVS` objects along with the
  following functions
  - `plot.summary.BranchGLMVS()` for plotting results from variable
    selection
  - `fit.summary.BranchGLMVS()` which can be used to get a `BranchGLM`
    object for one of the best models found

# BranchGLM 1.3.2

- Improving efficiency for the “branch and bound” and “switch branch and
  bound” methods in the `VariableSelection()` function.
- Fixed bug related to initial values in the `VariableSelection()`
  function.

# BranchGLM 1.3.1

- The `VariableSelection()` function should now properly handle
  interaction terms.

# BranchGLM 1.3.0

- Updated GLM fitting to use backtracking line search with strong Wolfe
  conditions instead of Armijo-Goldstein condition to find step size.
- Adding new variable selection types for `VariableSelection()` which
  are called “backward branch and bound” and “switch branch and bound”.
  These methods are similar to the regular branch and bound method, but
  sometimes they can be much faster.
- Added predict method for `BranchGLMVS` objects.

# BranchGLM 1.2.0

- Introducing new function `BranchGLM.fit()` which is similar to
  `glm.fit()`, it fits GLMs when given the design matrix X and the
  outcome vector Y. Can be faster than calling BranchGLM if X and Y are
  readily available.
- Fixing number of models fit that are reported by stepwise selection
  procedures when using parallel computation via the
  `VariableSelection()` function.
- Fixing SEs and p-values for gaussian and gamma GLMs.

# BranchGLM 1.1.3

- Fixing number of observations returned from `VariableSelection()`
  function in the presence of missing values

# BranchGLM 1.1.2

- Fixing multiple different bugs
- Speeding up linear regression fitting, especially for large models

# BranchGLM 1.1.1

- Fixing multiple different bugs

# BranchGLM 1.1.0

- Adding NEWS.md
- Minimized repeated work for linear regression, so it should now be
  much faster
- Added gamma regression along with some additional link functions
- Additional arguments added to `BranchGLM()` to reduce memory usage if
  desired
- Fixed print statement for `BranchGLMVS` objects when parallel
  computation was employed
