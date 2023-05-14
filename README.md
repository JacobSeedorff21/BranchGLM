BranchGLM
================

# Overview

**BranchGLM** is a package for fitting GLMs and performing branch and
bound variable selection for GLMs.

# How to install

**BranchGLM** can be installed using the `install.packages()` function

``` r
install.packages("BranchGLM")
```

It can also be installed via the `install_github()` function from the
**devtools** package.

``` r
devtools::install_github("JacobSeedorff21/BranchGLM")
```

# Usage

## Fitting GLMs

### Linear regression

**BranchGLM** can fit large linear regression models very quickly, next
is a comparison of runtimes with the built-in `lm()` function.

``` r
library(BranchGLM)
library(microbenchmark)
library(ggplot2)
set.seed(99601)

NormalSimul <- function(n, d, Bprob = .5){
  
  x <- MASS::mvrnorm(n, mu = rep(1, d), Sigma = diag(.5, nrow = d, ncol = d) + 
                 matrix(.5, ncol = d, nrow = d))
  
  beta <- rnorm(d + 1, mean = 1, sd = 1) 
  
  beta[sample(2:length(beta), floor((length(beta) - 1) * Bprob))] = 0
  
  y <- x %*% beta[-1] + beta[1] + rnorm(n, sd = 3)
  
  df <- cbind(y, x) |> 
    as.data.frame()
  
  df$y <- df$V1
  
  df$V1 <- NULL
  
  df
}
### Big simulation

df <- NormalSimul(10000, 250)

Times <- microbenchmark("BranchGLM" = {BranchGLM(y ~ ., data = df, 
                                                        family = "gaussian",
                                                   link = "identity")},
                        "Parallel BranchGLM" = {BranchGLM(y ~ ., data = df, 
                                                        family = "gaussian",
                                                   link = "identity",
                                                   parallel = TRUE)},
                        "lm" = {lm(y ~ ., data = df)},
                        times = 100)

autoplot(Times, log = FALSE)
```

![](README_images/linear-1.png)<!-- -->

### Logistic regression

**BranchGLM** can also fit large logistic regression models very
quickly, next is a comparison of runtimes with the built-in `glm()`
function.

``` r
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

df <- LogisticSimul(10000, 100)

Times <- microbenchmark("BFGS" = {BranchGLM(y ~ ., data = df, family = "binomial",
                                                   link = "logit", method = "BFGS")}, 
                        "L-BFGS" = {BranchGLM(y ~ ., data = df, family = "binomial",
                                                   link = "logit", method = "LBFGS")},
                        "Fisher" = {BranchGLM(y ~ ., data = df, family = "binomial",
                                                   link = "logit", method = "Fisher")},
                        "Parallel BFGS" = {BranchGLM(y ~ ., data = df, family = "binomial",
                                                   link = "logit", method = "BFGS",
                                                   parallel = TRUE)}, 
                        "Parallel L-BFGS" = {BranchGLM(y ~ ., data = df, 
                                                       family = "binomial",
                                                   link = "logit", method = "LBFGS",
                                                   parallel = TRUE)},
                        "Parallel Fisher" = {BranchGLM(y ~ ., data = df, 
                                                        family = "binomial",
                                                   link = "logit", method = "Fisher",
                                                   parallel = TRUE)},
                        "glm" = {glm(y ~ ., data = df, family = "binomial")},
                        times = 100)

autoplot(Times, log = FALSE)
```

![](README_images/logistic-1.png)<!-- -->

## Variable selection

**BranchGLM** can also perform best subset selection very quickly, here
is a comparison of runtimes with the `bestglm()` function from the
**bestglm** package.

``` r
library(bestglm)
set.seed(33391)

df <- LogisticSimul(1000, 15, .5, sd = 0.5)

## Times
### Timing branch and bound
BranchTime <- system.time(BranchVS <- VariableSelection(y ~ ., data = df, 
                                      family = "binomial", link = "logit",
                  type = "switch branch and bound", showprogress = FALSE,
                  parallel = FALSE, nthreads = 8, method = "Fisher", 
                  bestmodels = 10, metric = "AIC"))

BranchTime
```

    ##    user  system elapsed 
    ##    0.17    0.02    0.22

``` r
### Timing exhaustive search
Xy <- cbind(df[,-1], df[,1])
ExhaustiveTime <- system.time(BestVS <- bestglm(Xy, family = binomial(), IC = "AIC", 
                                                TopModels = 10))
ExhaustiveTime
```

    ##    user  system elapsed 
    ##  135.51    0.44  137.78

Finding the top 10 logistic regression models for this simulated dataset
with 15 variables with the branch and bound method is about 626.27 times
faster than an exhaustive search.

### Checking results

``` r
## Results
### Checking if both methods give same results
BranchModel <- fit(summary(BranchVS))
ExhaustiveModel <- BestVS$BestModel
identical(names(coef(BranchModel)),  names(coef(ExhaustiveModel)))
```

    ## [1] TRUE

Hence the two methods result in the same best model and the branch and
bound method was much faster than an exhaustive search.

### Visualization

There is also a convenient way to visualize the top models with the
**BranchGLM** package.

``` r
# Getting summary of model selection process
summ <- summary(BranchVS)

## Plotting models
plot(summ, type = "b")
```

![](README_images/visualization-1.png)<!-- -->![](README_images/visualization-2.png)<!-- -->

## Parallel computation

Parallel computation can be used to greatly speed up the branch and
bound algorithm, especially when the number of variables is large.

### Non-parallel time

``` r
# Setting seed and simulating data
set.seed(871980)
df <- LogisticSimul(1000, 40, .5, sd = .5)

# Performing branch and bound
SerialTime <- system.time(BranchVS <- VariableSelection(y ~ ., data = df, 
                                      family = "binomial", link = "logit",
                  type = "switch branch and bound", showprogress = FALSE,
                  parallel = FALSE, nthreads = 8, method = "Fisher"))

SerialTime
```

    ##    user  system elapsed 
    ##   55.91    0.91   57.94

### Parallel time

``` r
# Performing parallel branch and bound
ParallelTime <- system.time(ParBranchVS <- VariableSelection(y ~ ., data = df, 
                                      family = "binomial", link = "logit",
                  type = "switch branch and bound", showprogress = FALSE,
                  parallel = TRUE, nthreads = 12, method = "Fisher"))

ParallelTime
```

    ##    user  system elapsed 
    ##    5.58    0.01   15.86

Finding the top logistic regression model for this simulated dataset
with 40 variables with parallel computation is about 3.65 times faster
than without parallel computation.

### Checking results

``` r
# Checking if 
identical(summary(BranchVS)$results, summary(ParBranchVS)$results)
```

    ## [1] TRUE

They both give the same results
