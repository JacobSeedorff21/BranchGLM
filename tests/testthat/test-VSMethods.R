# Testing variable selection methods
## Gamma regression
test_that("Testing VS methods gamma", {
  library(BranchGLM)
  set.seed(8621)
  
  ### Making sure x and beta are positive to use inverse link
  x <- sapply(rep(1, 10), rexp, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rexp(11, rate = 1)
  y <- rgamma(n = 1000, shape = 1, scale = 1 / (x %*% beta))
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ### Fitting upper model
  Fit <- BranchGLM(y ~ ., data = Data, family = "gamma", link = "inverse")
  
  ### Exact Variable Selection
  BB <- VariableSelection(Fit, type = "Branch AND BOUnd", bestmodels = 1, 
                          metric = "AiC")
  BBB <- VariableSelection(Fit, type = "BackWArD Branch AND BOUnd", bestmodels = 1, 
                           metric = "aic")
  SBB <- VariableSelection(Fit, type = "SWITCH Branch AND BOUnd", bestmodels = 1, 
                           metric = "AIC")
  
  ### Checking results
  expect_equal(coef(BB), coef(BBB))
  expect_equal(coef(BB), coef(SBB))
  
  ### checking GLM fitting
  ind <- which(coef(BB) != 0)
  myCoefs <- rep(0, ncol(x))
  names(myCoefs) <- rownames(coef(BB))
  tempFit <- BranchGLM.fit(x[, ind, drop = FALSE], y, offset = Fit$offset,
                           family = Fit$family, link = Fit$link)
  myCoefs[ind] <- tempFit$coefficients[, 1]
  
  #### Testing results
  expect_equal(coef(BB)[, 1], myCoefs, tolerance = 1e-4)
  
  ### Checking that forward and backward are the same as step
  #### VariableSelection
  ##### forward
  forward <- VariableSelection(Fit, type = "FORWARD", bestmodels = 1, 
                               metric = "AIC")
  forwardCoef <- coef(forward)
  forwardCoef <- forwardCoef[forwardCoef != 0, ]
  forwardCoef <- forwardCoef[order(names(forwardCoef))]
  
  ##### backward
  backward <- VariableSelection(Fit, type = "backWARD", bestmodels = 1, 
                                metric = "AIC")
  backwardCoef <- coef(backward)
  backwardCoef <- backwardCoef[backwardCoef != 0, ]
  backwardCoef <- backwardCoef[order(names(backwardCoef))]
  
  ##### fast backward
  fastbackward <- VariableSelection(Fit, type = "fast backWARD", bestmodels = 1, 
                                    metric = "AIC")
  fastbackwardCoef <- coef(fastbackward)
  fastbackwardCoef <- fastbackwardCoef[fastbackwardCoef != 0, ]
  fastbackwardCoef <- fastbackwardCoef[order(names(fastbackwardCoef))]
  
  ##### double backward
  doublebackward <- VariableSelection(Fit, type = "double backWARD", bestmodels = 1, 
                                      metric = "AIC")
  doublebackwardCoef <- coef(doublebackward)
  doublebackwardCoef <- doublebackwardCoef[doublebackwardCoef != 0, ]
  doublebackwardCoef <- doublebackwardCoef[order(names(doublebackwardCoef))]
  
  ##### double backward
  fastdoublebackward <- VariableSelection(Fit, type = "fast double backWARD", bestmodels = 1, 
                                          metric = "AIC")
  fastdoublebackwardCoef <- coef(fastdoublebackward)
  fastdoublebackwardCoef <- fastdoublebackwardCoef[fastdoublebackwardCoef != 0, ]
  fastdoublebackwardCoef <- fastdoublebackwardCoef[order(names(fastdoublebackwardCoef))]
  
  
  #### step
  fullmodel <- glm(y ~ ., data = Data, family = Gamma(link = "inverse"))
  forwardStep <- step(glm(y ~ 1, data = Data, family = Gamma(link = "inverse")), 
                      direction = "forward", trace = 0, scope = formula(fullmodel))
  forwardCoefGLM <- coef(forwardStep)
  forwardCoefGLM <- forwardCoefGLM[order(names(forwardCoefGLM))]
  backwardStep <- step(fullmodel, direction = "backward", trace = 0)
  backwardCoefGLM <- coef(backwardStep)
  backwardCoefGLM <- backwardCoefGLM[order(names(backwardCoefGLM))]
  
  #### Checking results
  expect_equal(forwardCoef, forwardCoefGLM, tolerance = 1e-2)
  expect_equal(backwardCoef, backwardCoefGLM, tolerance = 1e-2)
  expect_equal(fastbackwardCoef, backwardCoef)
  expect_equal(fastdoublebackwardCoef, doublebackwardCoef)
})

## Gaussian
test_that("Testing VS methods gaussian", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 10), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(11)
  y <- rnorm(n = 1000, mean = x %*% beta, sd = 2)
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ## Fitting upper model
  Fit <- BranchGLM(y ~ ., data = Data, family = "gaussian", link = "identity")
  
  ### Exact Variable Selection
  BB <- VariableSelection(Fit, type = "Branch AND BOUnd", bestmodels = 1, 
                          metric = "AiC")
  BBB <- VariableSelection(Fit, type = "BackWArD Branch AND BOUnd", bestmodels = 1, 
                           metric = "aic")
  SBB <- VariableSelection(Fit, type = "SWITCH Branch AND BOUnd", bestmodels = 1, 
                           metric = "AIC")
  
  ### Checking results
  expect_equal(coef(BB), coef(BBB))
  expect_equal(coef(BB), coef(SBB))
  
  ### checking GLM fitting
  ind <- which(coef(BB) != 0)
  myCoefs <- rep(0, ncol(x))
  names(myCoefs) <- rownames(coef(BB))
  tempFit <- BranchGLM.fit(x[, ind, drop = FALSE], y, offset = Fit$offset,
                           family = Fit$family, link = Fit$link)
  myCoefs[ind] <- tempFit$coefficients[, 1]
  
  #### Testing results
  expect_equal(coef(BB)[, 1], myCoefs, tolerance = 1e-4)
  
  ### Checking that forward and backward are the same as step
  #### VariableSelection
  ##### forward
  forward <- VariableSelection(Fit, type = "FORWARD", bestmodels = 1, 
                               metric = "AIC")
  forwardCoef <- coef(forward)
  forwardCoef <- forwardCoef[forwardCoef != 0, ]
  forwardCoef <- forwardCoef[order(names(forwardCoef))]
  
  ##### backward
  backward <- VariableSelection(Fit, type = "backWARD", bestmodels = 1, 
                                metric = "AIC")
  backwardCoef <- coef(backward)
  backwardCoef <- backwardCoef[backwardCoef != 0, ]
  backwardCoef <- backwardCoef[order(names(backwardCoef))]
  
  ##### fast backward
  fastbackward <- VariableSelection(Fit, type = "fast backWARD", bestmodels = 1, 
                                    metric = "AIC")
  fastbackwardCoef <- coef(fastbackward)
  fastbackwardCoef <- fastbackwardCoef[fastbackwardCoef != 0, ]
  fastbackwardCoef <- fastbackwardCoef[order(names(fastbackwardCoef))]
  
  ##### double backward
  doublebackward <- VariableSelection(Fit, type = "double backWARD", bestmodels = 1, 
                                      metric = "AIC")
  doublebackwardCoef <- coef(doublebackward)
  doublebackwardCoef <- doublebackwardCoef[doublebackwardCoef != 0, ]
  doublebackwardCoef <- doublebackwardCoef[order(names(doublebackwardCoef))]
  
  ##### double backward
  fastdoublebackward <- VariableSelection(Fit, type = "fast double backWARD", bestmodels = 1, 
                                          metric = "AIC")
  fastdoublebackwardCoef <- coef(fastdoublebackward)
  fastdoublebackwardCoef <- fastdoublebackwardCoef[fastdoublebackwardCoef != 0, ]
  fastdoublebackwardCoef <- fastdoublebackwardCoef[order(names(fastdoublebackwardCoef))]
  
  
  #### step
  fullmodel <- glm(y ~ ., data = Data, family = gaussian(link = "identity"))
  forwardStep <- step(glm(y ~ 1, data = Data, family = gaussian(link = "identity")), 
                      direction = "forward", trace = 0, scope = formula(fullmodel))
  forwardCoefGLM <- coef(forwardStep)
  forwardCoefGLM <- forwardCoefGLM[order(names(forwardCoefGLM))]
  backwardStep <- step(fullmodel, direction = "backward", trace = 0)
  backwardCoefGLM <- coef(backwardStep)
  backwardCoefGLM <- backwardCoefGLM[order(names(backwardCoefGLM))]
  
  #### Checking results
  expect_equal(forwardCoef, forwardCoefGLM, tolerance = 1e-2)
  expect_equal(backwardCoef, backwardCoefGLM, tolerance = 1e-2)
  expect_equal(fastbackwardCoef, backwardCoef)
  expect_equal(fastdoublebackwardCoef, doublebackwardCoef)
})

## Binomial
### Probit
test_that("Testing VS methods binomial", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 10), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(11)
  y <- rbinom(n = 1000, size = 1, p = pnorm(x %*% beta + rnorm(1000, sd = 3)))
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ## Fitting upper model
  Fit <- BranchGLM(y ~ ., data = Data, family = "binomial", link = "probit")
  
  ### Exact Variable Selection
  BB <- VariableSelection(Fit, type = "Branch AND BOUnd", bestmodels = 1, 
                          metric = "AiC")
  BBB <- VariableSelection(Fit, type = "BackWArD Branch AND BOUnd", bestmodels = 1, 
                           metric = "aic")
  SBB <- VariableSelection(Fit, type = "SWITCH Branch AND BOUnd", bestmodels = 1, 
                           metric = "AIC")
  
  ### Checking results
  expect_equal(coef(BB), coef(BBB))
  expect_equal(coef(BB), coef(SBB))
  
  ### checking GLM fitting
  ind <- which(coef(BB) != 0)
  myCoefs <- rep(0, ncol(x))
  names(myCoefs) <- rownames(coef(BB))
  tempFit <- BranchGLM.fit(x[, ind, drop = FALSE], y, offset = Fit$offset,
                           family = Fit$family, link = Fit$link)
  myCoefs[ind] <- tempFit$coefficients[, 1]
  
  #### Testing results
  expect_equal(coef(BB)[, 1], myCoefs, tolerance = 1e-4)
  
  ### Checking that forward and backward are the same as step
  #### VariableSelection
  ##### forward
  forward <- VariableSelection(Fit, type = "FORWARD", bestmodels = 1, 
                               metric = "AIC")
  forwardCoef <- coef(forward)
  forwardCoef <- forwardCoef[forwardCoef != 0, ]
  forwardCoef <- forwardCoef[order(names(forwardCoef))]
  
  ##### backward
  backward <- VariableSelection(Fit, type = "backWARD", bestmodels = 1, 
                                metric = "AIC")
  backwardCoef <- coef(backward)
  backwardCoef <- backwardCoef[backwardCoef != 0, ]
  backwardCoef <- backwardCoef[order(names(backwardCoef))]
  
  ##### fast backward
  fastbackward <- VariableSelection(Fit, type = "fast backWARD", bestmodels = 1, 
                                    metric = "AIC")
  fastbackwardCoef <- coef(fastbackward)
  fastbackwardCoef <- fastbackwardCoef[fastbackwardCoef != 0, ]
  fastbackwardCoef <- fastbackwardCoef[order(names(fastbackwardCoef))]
  
  ##### double backward
  doublebackward <- VariableSelection(Fit, type = "double backWARD", bestmodels = 1, 
                                      metric = "AIC")
  doublebackwardCoef <- coef(doublebackward)
  doublebackwardCoef <- doublebackwardCoef[doublebackwardCoef != 0, ]
  doublebackwardCoef <- doublebackwardCoef[order(names(doublebackwardCoef))]
  
  ##### double backward
  fastdoublebackward <- VariableSelection(Fit, type = "fast double backWARD", bestmodels = 1, 
                                          metric = "AIC")
  fastdoublebackwardCoef <- coef(fastdoublebackward)
  fastdoublebackwardCoef <- fastdoublebackwardCoef[fastdoublebackwardCoef != 0, ]
  fastdoublebackwardCoef <- fastdoublebackwardCoef[order(names(fastdoublebackwardCoef))]
  
  
  #### step
  fullmodel <- glm(y ~ ., data = Data, family = binomial(link = "probit"))
  forwardStep <- step(glm(y ~ 1, data = Data, family = binomial(link = "probit")), 
                      direction = "forward", trace = 0, scope = formula(fullmodel))
  forwardCoefGLM <- coef(forwardStep)
  forwardCoefGLM <- forwardCoefGLM[order(names(forwardCoefGLM))]
  backwardStep <- step(fullmodel, direction = "backward", trace = 0)
  backwardCoefGLM <- coef(backwardStep)
  backwardCoefGLM <- backwardCoefGLM[order(names(backwardCoefGLM))]
  
  #### Checking results
  expect_equal(forwardCoef, forwardCoefGLM, tolerance = 1e-2)
  expect_equal(backwardCoef, backwardCoefGLM, tolerance = 1e-2)
  expect_equal(fastbackwardCoef, backwardCoef)
  expect_equal(fastdoublebackwardCoef, doublebackwardCoef)
})

### cloglog
test_that("Testing VS methods binomial", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 10), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(11)
  y <- rbinom(n = 1000, size = 1, p = 1 - exp(-exp(x %*% beta + rnorm(1000, sd = 5))))
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ## Fitting upper model
  Fit <- BranchGLM(y ~ ., data = Data, family = "binomial", link = "cloglog")
  
  ### Exact Variable Selection
  BB <- VariableSelection(Fit, type = "Branch AND BOUnd", bestmodels = 1, 
                          metric = "AiC")
  BBB <- VariableSelection(Fit, type = "BackWArD Branch AND BOUnd", bestmodels = 1, 
                           metric = "aic")
  SBB <- VariableSelection(Fit, type = "SWITCH Branch AND BOUnd", bestmodels = 1, 
                           metric = "AIC")
  
  ### Checking results
  expect_equal(coef(BB), coef(BBB))
  expect_equal(coef(BB), coef(SBB))
  
  ### checking GLM fitting
  ind <- which(coef(BB) != 0)
  myCoefs <- rep(0, ncol(x))
  names(myCoefs) <- rownames(coef(BB))
  tempFit <- BranchGLM.fit(x[, ind, drop = FALSE], y, offset = Fit$offset,
                           family = Fit$family, link = Fit$link)
  myCoefs[ind] <- tempFit$coefficients[, 1]
  
  #### Testing results
  expect_equal(coef(BB)[, 1], myCoefs, tolerance = 1e-4)
  
  ### Checking that forward and backward are the same as step
  #### VariableSelection
  ##### forward
  forward <- VariableSelection(Fit, type = "FORWARD", bestmodels = 1, 
                               metric = "AIC")
  forwardCoef <- coef(forward)
  forwardCoef <- forwardCoef[forwardCoef != 0, ]
  forwardCoef <- forwardCoef[order(names(forwardCoef))]
  
  ##### backward
  backward <- VariableSelection(Fit, type = "backWARD", bestmodels = 1, 
                                metric = "AIC")
  backwardCoef <- coef(backward)
  backwardCoef <- backwardCoef[backwardCoef != 0, ]
  backwardCoef <- backwardCoef[order(names(backwardCoef))]
  
  ##### fast backward
  fastbackward <- VariableSelection(Fit, type = "fast backWARD", bestmodels = 1, 
                                    metric = "AIC")
  fastbackwardCoef <- coef(fastbackward)
  fastbackwardCoef <- fastbackwardCoef[fastbackwardCoef != 0, ]
  fastbackwardCoef <- fastbackwardCoef[order(names(fastbackwardCoef))]
  
  ##### double backward
  doublebackward <- VariableSelection(Fit, type = "double backWARD", bestmodels = 1, 
                                      metric = "AIC")
  doublebackwardCoef <- coef(doublebackward)
  doublebackwardCoef <- doublebackwardCoef[doublebackwardCoef != 0, ]
  doublebackwardCoef <- doublebackwardCoef[order(names(doublebackwardCoef))]
  
  ##### double backward
  fastdoublebackward <- VariableSelection(Fit, type = "fast double backWARD", bestmodels = 1, 
                                          metric = "AIC")
  fastdoublebackwardCoef <- coef(fastdoublebackward)
  fastdoublebackwardCoef <- fastdoublebackwardCoef[fastdoublebackwardCoef != 0, ]
  fastdoublebackwardCoef <- fastdoublebackwardCoef[order(names(fastdoublebackwardCoef))]
  
  
  #### step
  fullmodel <- glm(y ~ ., data = Data, family = binomial(link = "cloglog"))
  forwardStep <- step(glm(y ~ 1, data = Data, family = binomial(link = "cloglog")), 
                      direction = "forward", trace = 0, scope = formula(fullmodel))
  forwardCoefGLM <- coef(forwardStep)
  forwardCoefGLM <- forwardCoefGLM[order(names(forwardCoefGLM))]
  backwardStep <- step(fullmodel, direction = "backward", trace = 0)
  backwardCoefGLM <- coef(backwardStep)
  backwardCoefGLM <- backwardCoefGLM[order(names(backwardCoefGLM))]
  
  #### Checking results
  expect_equal(forwardCoef, forwardCoefGLM, tolerance = 1e-2)
  expect_equal(backwardCoef, backwardCoefGLM, tolerance = 1e-2)
  expect_equal(fastbackwardCoef, backwardCoef)
  expect_equal(fastdoublebackwardCoef, doublebackwardCoef)
})

## Poisson
test_that("Testing VS methods poisson", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 10), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(11, sd = 0.1)
  y <- rpois(n = 1000, lambda = exp(x %*% beta))
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ## Fitting upper model
  Fit <- BranchGLM(y ~ ., data = Data, family = "poisson", link = "sqrt")
  
  ### Exact Variable Selection
  BB <- VariableSelection(Fit, type = "Branch AND BOUnd", bestmodels = 1, 
                          metric = "AiC")
  BBB <- VariableSelection(Fit, type = "BackWArD Branch AND BOUnd", bestmodels = 1, 
                           metric = "aic")
  SBB <- VariableSelection(Fit, type = "SWITCH Branch AND BOUnd", bestmodels = 1, 
                           metric = "AIC")
  
  ### Checking results
  expect_equal(coef(BB), coef(BBB))
  expect_equal(coef(BB), coef(SBB))
  
  ### checking GLM fitting
  ind <- which(coef(BB) != 0)
  myCoefs <- rep(0, ncol(x))
  names(myCoefs) <- rownames(coef(BB))
  tempFit <- BranchGLM.fit(x[, ind, drop = FALSE], y, offset = Fit$offset,
                           family = Fit$family, link = Fit$link)
  myCoefs[ind] <- tempFit$coefficients[, 1]
  
  #### Testing results
  expect_equal(coef(BB)[, 1], myCoefs, tolerance = 1e-4)
  
  ### Checking that forward and backward are the same as step
  #### VariableSelection
  ##### forward
  forward <- VariableSelection(Fit, type = "FORWARD", bestmodels = 1, 
                               metric = "AIC")
  forwardCoef <- coef(forward)
  forwardCoef <- forwardCoef[forwardCoef != 0, ]
  forwardCoef <- forwardCoef[order(names(forwardCoef))]
  
  ##### backward
  backward <- VariableSelection(Fit, type = "backWARD", bestmodels = 1, 
                                metric = "AIC")
  backwardCoef <- coef(backward)
  backwardCoef <- backwardCoef[backwardCoef != 0, ]
  backwardCoef <- backwardCoef[order(names(backwardCoef))]
  
  ##### fast backward
  fastbackward <- VariableSelection(Fit, type = "fast backWARD", bestmodels = 1, 
                                    metric = "AIC")
  fastbackwardCoef <- coef(fastbackward)
  fastbackwardCoef <- fastbackwardCoef[fastbackwardCoef != 0, ]
  fastbackwardCoef <- fastbackwardCoef[order(names(fastbackwardCoef))]
  
  ##### double backward
  doublebackward <- VariableSelection(Fit, type = "double backWARD", bestmodels = 1, 
                                      metric = "AIC")
  doublebackwardCoef <- coef(doublebackward)
  doublebackwardCoef <- doublebackwardCoef[doublebackwardCoef != 0, ]
  doublebackwardCoef <- doublebackwardCoef[order(names(doublebackwardCoef))]
  
  ##### double backward
  fastdoublebackward <- VariableSelection(Fit, type = "fast double backWARD", bestmodels = 1, 
                                          metric = "AIC")
  fastdoublebackwardCoef <- coef(fastdoublebackward)
  fastdoublebackwardCoef <- fastdoublebackwardCoef[fastdoublebackwardCoef != 0, ]
  fastdoublebackwardCoef <- fastdoublebackwardCoef[order(names(fastdoublebackwardCoef))]
  
  
  #### step
  fullmodel <- glm(y ~ ., data = Data, family = poisson(link = "sqrt"))
  forwardStep <- step(glm(y ~ 1, data = Data, family = poisson(link = "sqrt")), 
                      direction = "forward", trace = 0, scope = formula(fullmodel))
  forwardCoefGLM <- coef(forwardStep)
  forwardCoefGLM <- forwardCoefGLM[order(names(forwardCoefGLM))]
  backwardStep <- step(fullmodel, direction = "backward", trace = 0)
  backwardCoefGLM <- coef(backwardStep)
  backwardCoefGLM <- backwardCoefGLM[order(names(backwardCoefGLM))]
  
  #### Checking results
  expect_equal(forwardCoef, forwardCoefGLM, tolerance = 1e-2)
  expect_equal(backwardCoef, backwardCoefGLM, tolerance = 1e-2)
  expect_equal(fastbackwardCoef, backwardCoef)
  expect_equal(fastdoublebackwardCoef, doublebackwardCoef)
})

## Testing formula with -
### This used to be problematic
test_that("Testing formula with -", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 10), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(11)
  y <- rnorm(n = 1000, mean = x %*% beta, sd = 2)
  Data <- cbind(y, x) |>
    as.data.frame()
  
  ## Fitting upper model
  Fit <- BranchGLM(y ~ . - 1, data = Data, family = "gaussian", link = "identity")
  
  ### Exact Variable Selection with -
  BB <- VariableSelection(Fit, type = "Branch AND BOUnd", bestmodels = 1, 
                          metric = "AiC")
  BBB <- VariableSelection(Fit, type = "BackWArD Branch AND BOUnd", bestmodels = 1, 
                           metric = "aic")
  SBB <- VariableSelection(Fit, type = "SWITCH Branch AND BOUnd", bestmodels = 1, 
                           metric = "AIC")
  
  ### Exact Variable Selection without -
  Data <- cbind(y, x[, -1]) |>
    as.data.frame()
  Fit <- BranchGLM(y ~ ., data = Data, family = "gaussian", link = "identity")
  
  BB2 <- VariableSelection(Fit, type = "Branch AND BOUnd", bestmodels = 1, 
                          metric = "AiC", keepintercept = FALSE)
  BBB2 <- VariableSelection(Fit, type = "BackWArD Branch AND BOUnd", bestmodels = 1, 
                           metric = "aic", keepintercept = FALSE)
  SBB2 <- VariableSelection(Fit, type = "SWITCH Branch AND BOUnd", bestmodels = 1, 
                           metric = "AIC", keepintercept = FALSE)
  
  ### Checking results
  expect_equal(unname(coef(BB)), unname(BB2$beta))
  expect_equal(unname(coef(BBB)), unname(BBB2$beta))
  expect_equal(unname(coef(SBB)), unname(SBB2$beta))
  
})