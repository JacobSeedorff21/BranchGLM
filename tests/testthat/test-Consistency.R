# Testing consistency
## predict methods
test_that("Testing consistency of predict methods", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 10), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  x <- rbind(x, matrix(NA, nrow = 100, ncol = 11))
  beta <- rnorm(11, sd = 0.1)
  y <- exp(x %*% beta)
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  colnames(Data)[1] <- "y"
  Fit <- BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", 
                   offset = rep(0.01, 1100))
  VS <- VariableSelection(Fit)
  
  ### getting predictions with Fit
  preds <- predict(Fit)
  predsNew <- predict(Fit, newdata = Data, offset = rep(0.01, 1100))
  linpreds <- predict(Fit, type = "linpreds")
  linpredsNew <- predict(Fit, newdata = Data, type = "linpreds", offset = rep(0.01, 1100))
  predsNA <- predict(Fit, na.action = na.omit)
  predsNewNA <- predict(Fit, newdata = Data, offset = rep(0.01, 1100), na.action = na.omit)
  
  #### Testing for rownames and equality
  expect_equal(names(preds), rownames(Data))
  expect_equal(names(linpreds), rownames(Data))
  expect_equal(preds, predsNew)
  expect_equal(linpreds, linpredsNew)
  expect_equal(predsNA, predsNewNA)
  expect_equal(length(predsNA), 1000)
  
  ### getting predictions with VS
  preds <- predict(VS)
  predsNew <- predict(VS, newdata = Data, offset = rep(0.01, 1100))
  linpreds <- predict(VS, type = "linpreds")
  linpredsNew <- predict(VS, newdata = Data, type = "linpreds", offset = rep(0.01, 1100))
  predsNA <- predict(VS, na.action = na.omit)
  predsNewNA <- predict(VS, newdata = Data, offset = rep(0.01, 1100), na.action = na.omit)
  
  #### Testing for rownames and equality
  expect_equal(names(preds), rownames(Data))
  expect_equal(names(linpreds), rownames(Data))
  expect_equal(preds, predsNew)
  expect_equal(linpreds, linpredsNew)
  expect_equal(predsNA, predsNewNA)
  expect_equal(length(predsNA), 1000)
  
})

## Testing BranchGLM accessor functions
test_that("BranchGLM accessor functions", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 5), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  x <- rbind(x, matrix(NA, nrow = 100, ncol = 6))
  beta <- rnorm(6)
  y <- c(rgamma(n = 1000, shape = 1 / 2, scale = exp(x %*% beta) * 2), 
         rep(NA, 100))
  
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ## Fitting model while keeping data and y
  Fit <- BranchGLM(y ~ ., data = Data, family = "Gamma", link = "log")
  
  ### Testing nobs and formula
  expect_equal(formula(Fit), formula(y ~ .))
  expect_equal(nobs(Fit), 1000)
  expect_equal(nobs(Fit), attr(logLik(Fit), "nobs"))
  expect_equal(Fit$logLik, as.numeric(logLik(Fit)))
  expect_equal(length(coef(Fit)), 6)
  
  ### Performing variable selection
  BB <- VariableSelection(Fit)
  BB5 <- VariableSelection(Fit, bestmodels = 5)
  forward <- VariableSelection(Fit, type = "forward")
  
  ### Checking bestmodels
  expect_equal(is.matrix(forward$bestmodels), TRUE)
  expect_equal(is.matrix(BB$bestmodels), TRUE)
  expect_equal(nrow(forward$bestmodels), nrow(BB$bestmodels))
  expect_equal(is.matrix(BB5$bestmodels), TRUE)
  
  ### Checking beta
  expect_equal(is.matrix(forward$beta), TRUE)
  expect_equal(is.matrix(BB$beta), TRUE)
  expect_equal(nrow(forward$beta), nrow(BB$beta))
  expect_equal(is.matrix(BB5$beta), TRUE)
  
  ### Testing coef
  expect_equal(dim(coef(BB)), dim(coef(BB)))
  expect_equal(dim(coef(BB5)), dim(coef(forward)))
  expect_equal(dim(coef(BB5, which = "ALL")), c(6, 5))
  
  ## Fitting model without keeping data and y
  Fit <- BranchGLM(y ~ ., data = Data, family = "Gamma", link = "log", 
                   keepData = FALSE, keepY = FALSE)
  
  ### Testing functions
  expect_equal(formula(Fit), formula(y ~ .))
  expect_equal(nobs(Fit), 1000)
  expect_equal(nobs(Fit), attr(logLik(Fit), "nobs"))
  expect_equal(Fit$logLik, as.numeric(logLik(Fit)))
  expect_equal(length(coef(Fit)), 6)
  
})

# Testing predict function
## Gaussian and Gamma regression
test_that("Continuous regression tests", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(1, 10), rgamma, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rgamma(11, 1)
  y <- rgamma(n = 1000, shape = 1, scale = x %*% beta)
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  yInv <- rgamma(n = 1000, shape = 1, scale = 1 / (x %*% beta))
  DataInv <- cbind(y = yInv, x[,-1]) |>
    as.data.frame()
  
  ### Fitting models with BranchGLM
  #### Gaussian
  FitGausIden <- BranchGLM(y ~ ., data = Data, family = "gaussian", link = "identity")
  FitGausLog <- BranchGLM(y ~ ., data = Data, family = "gaussian", link = "log")
  FitGausInv <- BranchGLM(y ~ ., data = DataInv, family = "gaussian", link = "inverse")
  
  #### Gamma
  FitGamIden <- BranchGLM(y ~ ., data = Data, family = "gamma", link = "identity")
  FitGamLog <- BranchGLM(y ~ ., data = Data, family = "gamma", link = "log")
  FitGamInv <- BranchGLM(y ~ ., data = DataInv, family = "gamma", link = "inverse")
  
  ### Fitting models with glm
  #### Gaussian
  GLMGausIden <- glm(y ~ ., data = Data, family = gaussian(link = "identity"))
  GLMGausLog <- glm(y ~ ., data = Data, family = gaussian(link = "log"))
  GLMGausInv <- glm(y ~ ., data = DataInv, family = gaussian(link = "inverse"))
  
  #### Gamma
  ##### adding start since this didn't converge
  GLMGamIden <- glm(y ~ ., data = Data, family = Gamma(link = "identity"), 
                    start = beta)
  GLMGamLog <- glm(y ~ ., data = Data, family = Gamma(link = "log"))
  GLMGamInv <- glm(y ~ ., data = DataInv, family = Gamma(link = "inverse"))
  
  ### Checking results
  #### gaussian
  expect_equal(coef(FitGausIden), coef(GLMGausIden), tolerance = 1e-2)
  expect_equal(predict(FitGausIden), predict(GLMGausIden, type = "response"), 
               tolerance = 1e-2)
  expect_equal(coef(FitGausLog), coef(GLMGausLog), tolerance = 1e-2)
  expect_equal(predict(FitGausLog), predict(GLMGausLog, type = "response"), 
               tolerance = 1e-2)
  expect_equal(coef(FitGausInv), coef(GLMGausInv), tolerance = 1e-2)
  expect_equal(predict(FitGausInv), predict(GLMGausInv, type = "response"), 
               tolerance = 1e-2)
  
  #### gamma
  expect_equal(coef(FitGamIden), coef(GLMGamIden), tolerance = 1e-2)
  expect_equal(predict(FitGamIden), predict(GLMGamIden, type = "response"), 
               tolerance = 1e-2)
  expect_equal(coef(FitGamLog), coef(GLMGamLog), tolerance = 1e-2)
  expect_equal(predict(FitGamLog), predict(GLMGamLog, type = "response"), 
               tolerance = 1e-2)
  expect_equal(coef(FitGamInv), coef(GLMGamInv), tolerance = 1e-2)
  expect_equal(predict(FitGamInv), predict(GLMGamInv, type = "response"), 
               tolerance = 1e-2)
  
})


## Binomial regression
test_that("Binomial regression tests", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 10), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(11)
  y <- rbinom(n = 1000, size = 1, p = 1 / (1 + exp(-x %*% beta)))
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ### Fitting models with BranchGLM
  FitClog <- BranchGLM(y ~ ., data = Data, family = "binomial", link = "cloglog")
  FitLogit <- BranchGLM(y ~ ., data = Data, family = "binomial", link = "logit")
  FitProbit <- BranchGLM(y ~ ., data = Data, family = "binomial", link = "probit")
  
  ### Fitting models with glm
  GLMClog <- suppressWarnings(glm(y ~ ., data = Data, family = binomial(link = "cloglog")))
  GLMLogit <- glm(y ~ ., data = Data, family = binomial(link = "logit"))
  GLMProbit <- glm(y ~ ., data = Data, family = binomial(link = "probit"))
  
  ### Checking results
  expect_equal(coef(FitClog), coef(GLMClog), tolerance = 1e-2)
  expect_equal(predict(FitClog), predict(GLMClog, type = "response"), 
               tolerance = 1e-2)
  expect_equal(coef(FitLogit), coef(GLMLogit), tolerance = 1e-2)
  expect_equal(predict(FitLogit), predict(GLMLogit, type = "response"), 
               tolerance = 1e-2)
  expect_equal(coef(FitProbit), coef(GLMProbit), tolerance = 1e-2)
  expect_equal(predict(FitProbit), predict(GLMProbit, type = "response"), 
               tolerance = 1e-2)
  
})

## Poisson regression
test_that("Poisson regression tests", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(1, 10), rgamma, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rgamma(11, 1)
  y <- rpois(n = 1000, lambda = x %*% beta)
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ### Fitting models with BranchGLM
  FitIden <- BranchGLM(y ~ ., data = Data, family = "poisson", link = "identity")
  FitLog <- BranchGLM(y ~ ., data = Data, family = "poisson", link = "log")
  FitSqrt <- BranchGLM(y ~ ., data = Data, family = "poisson", link = "sqrt")
  
  ### Fitting models with glm
  GLMIden <- glm(y ~ ., data = Data, family = poisson(link = "identity"), 
                 start = beta)
  GLMLog <- glm(y ~ ., data = Data, family = poisson(link = "log"))
  GLMSqrt <- glm(y ~ ., data = Data, family = poisson(link = "sqrt"))
  
  ### Checking results
  expect_equal(coef(FitIden), coef(GLMIden), tolerance = 1e-2)
  expect_equal(predict(FitIden), predict(GLMIden, type = "response"), 
               tolerance = 1e-2)
  expect_equal(coef(FitLog), coef(GLMLog), tolerance = 1e-2)
  expect_equal(predict(FitLog), predict(GLMLog, type = "response"), 
               tolerance = 1e-2)
  expect_equal(coef(FitSqrt), coef(GLMSqrt), tolerance = 1e-2)
  expect_equal(predict(FitSqrt), predict(GLMSqrt, type = "response"), 
               tolerance = 1e-2)
  
})
