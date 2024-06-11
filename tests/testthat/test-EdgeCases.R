# Edge cases
## Perfect multicollinearity test
test_that("Perfect multicollinearity test", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 10), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(11)
  y <- rgamma(n = 1000, shape = 1, scale = exp(x %*% beta))
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ### Fitting models with perfect multicollinearity
  expect_error(BranchGLM(y ~ . + I(V2), data = Data, family = "gamma", link = "log", 
                         method = "Fisher") |> suppressWarnings())
  expect_error(BranchGLM(y ~ . + I(V2), data = Data, family = "gamma", link = "log", 
                         method = "BFGS") |> suppressWarnings())
  expect_error(BranchGLM(y ~ . + I(V2), data = Data, family = "gamma", link = "log", 
                         method = "LBFGS") |> suppressWarnings())
})

## Perfect regression test
test_that("Perfect regression test", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 10), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(11, sd = 0.1)
  y <- exp(x %*% beta)
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  colnames(Data)[1] <- "y"
  
  ### Fitting models with exact relationship
  #### Also checking tolower
  Fish <- BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", 
                    method = "FISHer")
  BFGS <- BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", 
                    method = "Bfgs")
  LBFGS <- BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", 
                     method = "LBfgs")
  
  ### Checking results
  names(beta) <- names(coef(Fish))
  expect_equal(coef(Fish), beta)
  expect_equal(coef(BFGS), beta)
  expect_equal(coef(LBFGS), beta)
  
  ### Variable Selection with exact relationship
  #### Also checking tolower and toupper
  BB <- VariableSelection(Fish, type = "Branch AND BOUnd", bestmodels = 1, 
                          metric = "AiC")
  BBB <- VariableSelection(Fish, type = "BackWArD Branch AND BOUnd", bestmodels = 1, 
                           metric = "aic")
  SBB <- VariableSelection(Fish, type = "SWITCH Branch AND BOUnd", bestmodels = 1, 
                           metric = "AIC")
  
  ### Checking results
  expect_equal(BB$beta, BBB$beta)
  expect_equal(BB$beta, SBB$beta)
  
  ### Checking that forward and backward don't error out
  expect_error(VariableSelection(Fish, type = "FORWARD", bestmodels = 1, 
                               metric = "AIC"), NA)
  expect_error(VariableSelection(Fish, type = "backWARD", bestmodels = 1, 
                               metric = "AIC"), NA)
})

## Testing empty model
test_that("Testing empty model", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 0), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(0)
  y <- rgamma(n = 0, shape = 1 / 2, scale = 0)
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ### Testing for errors
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", 
                            method = "Fisher"))
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", 
                            method = "BFGS"))
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", 
                             method = "LBFGS"))
  expect_error(VariableSelection(y ~ ., data = Data, family = "gamma", link = "log"))
})

## Testing model with all NAs
test_that("Testing model with all NAs", {
  library(BranchGLM)
  set.seed(8621)
  x <- matrix(NA, ncol = 10, nrow = 1000)
  y <- rep(NA, 1000)
  Data <- cbind(y, x) |>
    as.data.frame()
  
  ### Testing for errors
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", 
                         method = "Fisher"))
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", 
                         method = "BFGS"))
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", 
                         method = "LBFGS"))
  
  ### Testing for errors in variableselection
  expect_error(VariableSelection(y ~ ., data = Data, family = "gamma", link = "log"))
  
})

## Testing model with all NAs in design matrix
test_that("Testing model with all NAs in design matrix", {
  library(BranchGLM)
  set.seed(8621)
  x <- matrix(NA, ncol = 10, nrow = 1000)
  y <- rep(2, 1000)
  Data <- cbind(y, x) |>
    as.data.frame()
  
  ### Testing for errors
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", 
                         method = "Fisher"))
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", 
                         method = "BFGS"))
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", 
                         method = "LBFGS"))
  
  ### Testing for errors in variableselection
  expect_error(VariableSelection(y ~ ., data = Data, family = "gamma", link = "log"))
})

## Testing model with all NAs in y
test_that("Testing model with all NAs in y", {
  library(BranchGLM)
  set.seed(8621)
  x <- matrix(1, ncol = 10, nrow = 1000)
  y <- rep(NA, 1000)
  Data <- cbind(y, x) |>
    as.data.frame()
  
  ### Testing for errors
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", 
                         method = "Fisher"))
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", 
                         method = "BFGS"))
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", 
                         method = "LBFGS"))
  
  ### Testing for errors in variableselection
  expect_error(VariableSelection(y ~ ., data = Data, family = "gamma", link = "log"))
})

# Testing highly correlated covariates
test_that("Testing highly correlated covariates", {
  library(BranchGLM)
  set.seed(8621)
  sigma <- diag(.05, nrow = 10, ncol = 10) + matrix(.95, ncol = 10, nrow = 10)
  x <- sapply(rep(0, 10), rnorm, n = 1000, simplify = TRUE)
  x <- x %*% chol(sigma)
  x <- cbind(1, x)
  beta <- rnorm(11, sd = 0.1)
  y <- rgamma(1000, shape = 1, scale = exp(x %*% beta))
  Data <- cbind(y, x[, -1]) |>
    as.data.frame()
  
  ### Testing for errors
  expect_error(Fish <- BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", 
                         method = "Fisher"), NA)
  expect_error(BFGS <- BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", 
                         method = "BFGS"), NA)
  expect_error(LBFGS <- BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", 
                         method = "LBFGS"), NA)
  expect_error(Linear <- BranchGLM(y ~ ., data = Data, family = "gaussian", 
                         link = "identity"), NA)
  
  ### Variable Selection
  BBFish <- VariableSelection(Fish, bestmodels = 1, metric = "AiC")
  BBBFGS <- VariableSelection(BFGS, bestmodels = 1, metric = "aic")
  BBLBFGS <- VariableSelection(LBFGS, bestmodels = 1, metric = "AIC")
  BBLinear <- VariableSelection(Linear, bestmodels = 1, metric = "AIC")
  
  ### Checking results
  expect_equal(coef(BBFish), coef(BBBFGS), tolerance = 1e-2)
  expect_equal(coef(BBFish), coef(BBLBFGS), tolerance = 1e-2)
  
  ### checking GLM fitting
  #### Fisher
  ind <- which(coef(BBFish) != 0)
  FishCoefs <- rep(0, ncol(x))
  names(FishCoefs) <- rownames(coef(BBFish))
  tempFit <- BranchGLM.fit(x[, ind, drop = FALSE], y, offset = Fish$offset,
                           family = Fish$family, link = Fish$link)
  FishCoefs[ind] <- tempFit$coefficients[, 1]
  
  #### BFGS
  ind <- which(coef(BBBFGS) != 0)
  BFGSCoefs <- rep(0, ncol(x))
  names(BFGSCoefs) <- rownames(coef(BBBFGS))
  tempFit <- BranchGLM.fit(x[, ind, drop = FALSE], y, offset = BFGS$offset,
                           family = BFGS$family, link = BFGS$link)
  BFGSCoefs[ind] <- tempFit$coefficients[, 1]
  
  #### LBFGS
  ind <- which(coef(BBLBFGS) != 0)
  LBFGSCoefs <- rep(0, ncol(x))
  names(LBFGSCoefs) <- rownames(coef(BBLBFGS))
  tempFit <- BranchGLM.fit(x[, ind, drop = FALSE], y, offset = LBFGS$offset,
                           family = LBFGS$family, link = LBFGS$link)
  LBFGSCoefs[ind] <- tempFit$coefficients[, 1]
  
  #### Linear
  ind <- which(coef(BBLinear) != 0)
  LinearCoefs <- rep(0, ncol(x))
  names(LinearCoefs) <- rownames(coef(BBLinear))
  tempFit <- BranchGLM.fit(x[, ind, drop = FALSE], y, offset = Linear$offset,
                           family = Linear$family, link = Linear$link)
  LinearCoefs[ind] <- tempFit$coefficients[, 1]
  
  #### Testing results
  expect_equal(coef(BBFish)[, 1], FishCoefs, tolerance = 1e-4)
  expect_equal(coef(BBBFGS)[, 1], BFGSCoefs, tolerance = 1e-4)
  expect_equal(coef(BBLBFGS)[, 1], LBFGSCoefs, tolerance = 1e-4)
  expect_equal(coef(BBLinear)[, 1], LinearCoefs, tolerance = 1e-4)
  
})

# Testing gaussian log-likelihood with odd number of observations
## There was a bug in this due to integer division

test_that("Gaussian log-likelihood recovery test", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 10), rnorm, n = 1001, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(11)
  y <- rnorm(n = 1001, mean = x %*% beta, sd = 2)
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ### Fitting models
  Fit <- BranchGLM(y ~ ., data = Data, family = "gaussian", link = "identity")
  glmfit <- glm(y ~ ., data = Data)
  
  #### log-likelihood
  expect_equal(logLik(Fit), logLik(glmfit))
  
  ### Variable selection tests
  SBB <- VariableSelection(y ~ ., data = Data, family = "gaussian", link = "identity", 
                           showprogress = FALSE)
  temp <- summary(SBB)
  Fit <- BranchGLM(temp$formulas[[1]], data = Data, family = "gaussian", link = "identity")
  glmfit <- glm(temp$formulas[[1]], data = Data)
  
  #### log-likelihood
  expect_equal(AIC(Fit), AIC(glmfit))
  expect_equal(SBB$bestmetrics[1], AIC(Fit))
})
