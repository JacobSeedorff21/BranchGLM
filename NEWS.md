BranchGLM News
================

# BranchGLM 1.3.1

-   The `VariableSelection()` function should now properly handle
    interaction terms.

# BranchGLM 1.3.0

-   Updated GLM fitting to use backtracking line search with strong
    Wolfe conditions instead of Armijo-Goldstein condition to find step
    size.
-   Adding new variable selection types for `VariableSelection()` which
    are called “backward branch and bound” and “switch branch and
    bound”. These methods are similar to the regular branch and bound
    method, but sometimes they can be much faster.
-   Added predict method for `BranchGLMVS` objects.

# BranchGLM 1.2.0

-   Introducing new function `BranchGLM.fit()` which is similar to
    `glm.fit()`, it fits GLMs when given the design matrix X and the
    outcome vector Y. Can be faster than calling BranchGLM if X and Y
    are readily available.
-   Fixing number of models fit that are reported by stepwise selection
    procedures when using parallel computation via the
    `VariableSelection()` function.
-   Fixing SEs and p-values for gaussian and gamma GLMs.

# BranchGLM 1.1.3

-   Fixing number of observations returned from `VariableSelection()`
    function in the presence of missing values

# BranchGLM 1.1.2

-   Fixing multiple different bugs
-   Speeding up linear regression fitting, especially for large models

# BranchGLM 1.1.1

-   Fixing multiple different bugs

# BranchGLM 1.1.0

-   Adding NEWS.md
-   Minimized repeated work for linear regression, so it should now be
    much faster
-   Added gamma regression along with some additional link functions
-   Additional arguments added to `BranchGLM()` to reduce memory usage
    if desired
-   Fixed print statement for `BranchGLMVS` objects when parallel
    computation was employed
