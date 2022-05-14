BranchGLM
================

# Overview

**BranchGLM** is a package for fitting glms and performing branch and
bound variable selection for glms.

# How to install

**BranchGLM** can be installed using the `install_github()` function
from the **devtools** package.

``` r
devtools::install_github("JacobSeedorff21/BranchGLM")
```

# Usage

## Fitting glms

**BranchGLM** can fit large regression models very quickly, next is a
comparison of runtimes with the built-in `glm()` function.

``` r
library(BranchGLM)
library(microbenchmark)
library(ggplot2)
set.seed(78771)

LogisticSimul <- function(n, d, Bprob = .5, sd = 1){
  
  x <- MASS::mvrnorm(n, mu = rep(1, d), Sigma = diag(.5, nrow = d, ncol = d) + 
                 matrix(.5, ncol = d, nrow = d))
  
  beta <- rnorm(d + 1, mean = 0, sd = sd) 
  
  beta[sample(2:length(beta), floor((length(beta) - 1) * Bprob))] = 0
  
  p <- 1/(1 + exp(-x %*% beta[-1] - beta[1]))
  
  y <- rbinom(n, 1, p)
  
  df <- cbind(y, x) |> 
    as.data.frame()
  df
}

### Big simulation

df <- LogisticSimul(10000, 50)

Times <- microbenchmark(BFGS = {BranchGLM(y ~ ., data = df, family = "binomial",
                                                   link = "logit", method = "BFGS")}, 
                        LBFGS = {BranchGLM(y ~ ., data = df, family = "binomial",
                                                   link = "logit", method = "LBFGS")},
                        Fisher = {BranchGLM(y ~ ., data = df, family = "binomial",
                                                   link = "logit", method = "Fisher")},
                        ParallelBFGS = {BranchGLM(y ~ ., data = df, family = "binomial",
                                                   link = "logit", method = "BFGS",
                                                   parallel = TRUE)}, 
                        ParallelLBFGS = {BranchGLM(y ~ ., data = df, 
                                                       family = "binomial",
                                                   link = "logit", method = "LBFGS",
                                                   parallel = TRUE)},
                        ParallelFisher = {BranchGLM(y ~ ., data = df, 
                                                        family = "binomial",
                                                   link = "logit", method = "Fisher",
                                                   parallel = TRUE)},
                        glm = {glm(y ~ ., data = df, family = "binomial")},
                        times = 100)

autoplot(Times, log = FALSE)
```

![](README_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

## Variable selection

**BranchGLM** can also perform best subsets selection very quickly, here
is a comparison of runtimes with the `bestglm()` function from the
**bestglm** package.

``` r
library(bestglm)
set.seed(33391)

df <- LogisticSimul(1000, 15, .5, sd = .5)

## Times
### Timing branch and bound
system.time(BranchVS <- VariableSelection(y ~ ., data = df, 
                                      family = "binomial", link = "logit",
                  type = "branch and bound", showprogress = FALSE))
```

    ##    user  system elapsed 
    ##    0.34    0.00    0.34

``` r
Xy <- cbind(df[,-1], df[,1])

### Timing exhaustive search
system.time(BestVS <- bestglm(Xy, family = binomial(), IC = "AIC", TopModels = 1))
```

    ##    user  system elapsed 
    ##  196.54    0.09 7672.36

``` r
## Results
### Branch and bound
coef(BranchVS$finalmodel) |>
  names()
```

    ##  [1] "(Intercept)" "V3"          "V4"          "V5"          "V7"         
    ##  [6] "V8"          "V9"          "V10"         "V13"         "V14"        
    ## [11] "V16"

``` r
### Exhaustive search
coef(BestVS$BestModel) |>
  names()
```

    ##  [1] "(Intercept)" "V3"          "V4"          "V5"          "V7"         
    ##  [6] "V8"          "V9"          "V10"         "V13"         "V14"        
    ## [11] "V16"

The branch and bound method can be many times faster than an exhaustive
search and is still guaranteed to find the optimal model.

## Binomial glm utility functions

-   **BranchGLM** also has some utility functions for binomial glms
    -   `Table()` creates a confusion matrix based on the predicted
        classes and observed classes
    -   `ROC()` creates an ROC curve which can be plotted with `plot()`
    -   `AUC()` and `Cindex()` calculate the area under the ROC curve
    -   `MultipleROCCurves()` allows for the plotting of multiple ROC
        curves on the same plot
