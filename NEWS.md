BranchGLM News
================

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
