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
  Fit <- BranchGLM(y ~ ., data = Data, family = gaussian())
  
  ### Testing nobs and formula
  expect_equal(formula(Fit), formula(y ~ .))
  expect_equal(nobs(Fit), 1000)
  expect_equal(Fit$missing, 100)
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
  #### model.frame, and nobs
  expect_equal(model.frame(GLMGausIden), model.frame(FitGausIden))
  expect_equal(nobs(GLMGausIden), nobs(FitGausIden))
  
  #### gaussian
  ##### Checking log-likelihood
  expect_equal(logLik(FitGausIden)[[1]], 
               sum(dnorm(y, FitGausIden$pred, sigma(FitGausIden)[[1]], log = TRUE)))
  
  ##### Identity
  expect_equal(coef(FitGausIden), coef(GLMGausIden), tolerance = 1e-3)
  expect_equal(residuals(FitGausIden), residuals(GLMGausIden, type = "pearson"), 
               tolerance = 1e-3)
  expect_equal(predict(FitGausIden), predict(GLMGausIden, type = "response"), 
               tolerance = 1e-3)
  expect_equal(deviance(FitGausIden), deviance(GLMGausIden), tolerance = 1e-3)
  expect_equal(logLik(FitGausIden), logLik(GLMGausIden), tolerance = 1e-3)
  expect_equal(extractAIC(FitGausIden)[[2]], extractAIC(GLMGausIden)[[2]], tolerance = 1e-3)
  expect_equal(sigma(GLMGausIden), sigma(FitGausIden)[[2]], tolerance = 1e-3)
  expect_equal(as.data.frame(coef(summary(GLMGausIden))), 
                             coef(FitGausIden, type = "all"), tolerance = 1e-3)
  expect_equal(vcov(GLMGausIden), vcov(FitGausIden) * (1000 / (1000 - 11)), tolerance = 1e-3)
  expect_equal(family(GLMGausIden),  family(FitGausIden))
  
  ##### Log
  expect_equal(coef(FitGausLog), coef(GLMGausLog), tolerance = 1e-3)
  expect_equal(predict(FitGausLog), predict(GLMGausLog, type = "response"), 
               tolerance = 1e-3)
  expect_equal(residuals(FitGausLog), residuals(GLMGausLog, type = "pearson"), 
               tolerance = 1e-3)
  expect_equal(deviance(FitGausLog), deviance(GLMGausLog), tolerance = 1e-3)
  expect_equal(logLik(FitGausLog), logLik(GLMGausLog), tolerance = 1e-3)
  expect_equal(extractAIC(FitGausLog), extractAIC(GLMGausLog), tolerance = 1e-3)
  expect_equal(sigma(GLMGausLog), sigma(FitGausLog)[[2]], tolerance = 1e-3)
  expect_equal(as.data.frame(coef(summary(GLMGausLog))), 
                             coef(FitGausLog, type = "all"), tolerance = 1e-3)
  expect_equal(family(GLMGausLog),  family(FitGausLog))
  
  ##### Inverse
  expect_equal(coef(FitGausInv), coef(GLMGausInv), tolerance = 1e-3)
  expect_equal(predict(FitGausInv), predict(GLMGausInv, type = "response"), 
               tolerance = 1e-3)
  expect_equal(residuals(FitGausInv), residuals(GLMGausInv, type = "pearson"), 
               tolerance = 1e-3)
  expect_equal(deviance(FitGausInv), deviance(GLMGausInv), tolerance = 1e-3)
  expect_equal(logLik(FitGausInv), logLik(GLMGausInv), tolerance = 1e-3)
  expect_equal(extractAIC(FitGausInv), extractAIC(GLMGausInv), tolerance = 1e-3)
  expect_equal(sigma(GLMGausInv), sigma(FitGausInv)[[2]], tolerance = 1e-3)
  expect_equal(as.data.frame(coef(summary(GLMGausInv))), 
                             coef(FitGausInv, type = "all"), tolerance = 1e-2)
  expect_equal(family(GLMGausInv),  family(FitGausInv))
  
  #### gamma
  ##### Checking log-likelihood
  expect_equal(logLik(FitGamIden)[[1]], 
               sum(dgamma(y, 1 / sigma(FitGamIden)[[1]]^2, 
                          scale = FitGamIden$preds * sigma(FitGamIden)[[1]]^2, 
                          log = TRUE)))
  
  ##### Identity
  expect_equal(coef(FitGamIden), coef(GLMGamIden), tolerance = 1e-3)
  expect_equal(predict(FitGamIden), predict(GLMGamIden, type = "response"), 
               tolerance = 1e-3)
  expect_equal(residuals(FitGamIden), residuals(GLMGamIden, type = "pearson"), 
               tolerance = 1e-3)
  expect_equal(deviance(FitGamIden), deviance(GLMGamIden), tolerance = 1e-3)
  expect_equal(sigma(GLMGamIden), sigma(FitGamIden)[[2]], tolerance = 1e-3)
  expect_equal(as.data.frame(coef(summary(GLMGamIden))), 
                             coef(FitGamIden, type = "all"), tolerance = 1e-3)
  expect_equal(family(FitGamIden), family(GLMGamIden))
  
  ##### Log
  expect_equal(coef(FitGamLog), coef(GLMGamLog), tolerance = 1e-3)
  expect_equal(predict(FitGamLog), predict(GLMGamLog, type = "response"), 
               tolerance = 1e-3)
  expect_equal(residuals(FitGamLog), residuals(GLMGamLog, type = "pearson"), 
               tolerance = 1e-3)
  expect_equal(deviance(FitGamLog), deviance(GLMGamLog), tolerance = 1e-3)
  expect_equal(sigma(GLMGamLog), sigma(FitGamLog)[[2]], tolerance = 1e-3)
  expect_equal(as.data.frame(coef(summary(GLMGamLog))), 
                             coef(FitGamLog, type = "all"), tolerance = 1e-3)
  expect_equal(family(FitGamLog), family(GLMGamLog))
  
  ##### Inverse
  expect_equal(coef(FitGamInv), coef(GLMGamInv), tolerance = 1e-3)
  expect_equal(predict(FitGamInv), predict(GLMGamInv, type = "response"), 
               tolerance = 1e-3)
  expect_equal(residuals(FitGamInv), residuals(GLMGamInv, type = "pearson"), 
               tolerance = 1e-3)
  expect_equal(deviance(FitGamInv), deviance(GLMGamInv), tolerance = 1e-3)
  expect_equal(sigma(GLMGamInv), sigma(FitGamInv)[[2]], tolerance = 1e-3)
  expect_equal(as.data.frame(coef(summary(GLMGamInv))), 
               coef(FitGamInv, type = "all"), tolerance = 1e-3)
  expect_equal(family(FitGamInv), family(GLMGamInv))
  
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
  
  ##### Checking log-likelihood
  expect_equal(logLik(FitClog)[[1]], 
               sum(dbinom(y, 1, FitClog$preds, log = TRUE)))
  
  ### Checking results
  #### Cloglog
  expect_equal(coef(FitClog), coef(GLMClog), tolerance = 1e-3)
  expect_equal(predict(FitClog), predict(GLMClog, type = "response"), 
               tolerance = 1e-3)
  expect_equal(residuals(FitClog), residuals(GLMClog, type = "pearson"), 
               tolerance = 1e-3)
  expect_equal(deviance(FitClog), deviance(GLMClog), tolerance = 1e-3)
  expect_equal(logLik(FitClog), logLik(GLMClog), tolerance = 1e-3)
  expect_equal(extractAIC(FitClog), extractAIC(GLMClog), tolerance = 1e-3)
  expect_equal(vcov(FitClog), vcov(GLMClog), tolerance = 1e-3)
  expect_equal(coef(FitClog, type = "all"), as.data.frame(coef(summary(GLMClog))), 
               tolerance = 1e-3)
  expect_equal(unname(sigma(FitClog)), c(1, 1))
  expect_equal(family(FitClog), family(GLMClog), tolerance = 1e-3)
  
  #### Logit
  expect_equal(coef(FitLogit), coef(GLMLogit), tolerance = 1e-3)
  expect_equal(predict(FitLogit), predict(GLMLogit, type = "response"), 
               tolerance = 1e-3)
  expect_equal(residuals(FitLogit), residuals(GLMLogit, type = "pearson"), 
               tolerance = 1e-3)
  expect_equal(deviance(FitLogit), deviance(GLMLogit), tolerance = 1e-3)
  expect_equal(logLik(FitLogit), logLik(GLMLogit), tolerance = 1e-3)
  expect_equal(extractAIC(FitLogit), extractAIC(GLMLogit), tolerance = 1e-3)
  expect_equal(vcov(FitLogit), vcov(GLMLogit), tolerance = 1e-3)
  expect_equal(coef(FitLogit, type = "all"), as.data.frame(coef(summary(GLMLogit))), 
               tolerance = 1e-3)
  expect_equal(unname(sigma(FitLogit)), c(1, 1))
  expect_equal(family(FitLogit), family(GLMLogit), tolerance = 1e-3)
  
  #### Probit
  expect_equal(coef(FitProbit), coef(GLMProbit), tolerance = 1e-3)
  expect_equal(predict(FitProbit), predict(GLMProbit, type = "response"), 
               tolerance = 1e-3)
  expect_equal(residuals(FitProbit), residuals(GLMProbit, type = "pearson"), 
               tolerance = 1e-3)
  expect_equal(deviance(FitProbit), deviance(GLMProbit), tolerance = 1e-3)
  expect_equal(logLik(FitProbit), logLik(GLMProbit), tolerance = 1e-3)
  expect_equal(extractAIC(FitProbit), extractAIC(GLMProbit), tolerance = 1e-3)
  expect_equal(vcov(FitProbit), vcov(GLMProbit), tolerance = 1e-3)
  expect_equal(coef(FitProbit, type = "all"), as.data.frame(coef(summary(GLMProbit))), 
               tolerance = 1e-3)
  expect_equal(unname(sigma(FitProbit)), c(1, 1))
  expect_equal(family(FitProbit), family(GLMProbit), tolerance = 1e-3)
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
  
  ##### Checking log-likelihood
  expect_equal(logLik(FitLog)[[1]], 
               sum(dpois(y, FitLog$pred, log = TRUE)))
  
  ### Checking results
  #### Identity
  expect_equal(coef(FitIden), coef(GLMIden), tolerance = 1e-3)
  expect_equal(predict(FitIden), predict(GLMIden, type = "response"), 
               tolerance = 1e-3)
  expect_equal(residuals(FitIden), residuals(GLMIden, type = "pearson"), 
               tolerance = 1e-3)
  expect_equal(deviance(FitIden), deviance(GLMIden), tolerance = 1e-3)
  expect_equal(logLik(FitIden), logLik(GLMIden), tolerance = 1e-3)
  expect_equal(extractAIC(FitIden), extractAIC(GLMIden), tolerance = 1e-3)
  expect_equal(vcov(FitIden), vcov(GLMIden), tolerance = 1e-3)
  expect_equal(coef(FitIden, type = "all"), as.data.frame(coef(summary(GLMIden))), 
               tolerance = 1e-3)
  expect_equal(unname(sigma(FitIden)), c(1, 1))
  expect_equal(family(FitIden), family(GLMIden), tolerance = 1e-3)
  
  #### Log
  expect_equal(coef(FitLog), coef(GLMLog), tolerance = 1e-3)
  expect_equal(predict(FitLog), predict(GLMLog, type = "response"), 
               tolerance = 1e-3)
  expect_equal(residuals(FitLog), residuals(GLMLog, type = "pearson"), 
               tolerance = 1e-3)
  expect_equal(deviance(FitLog), deviance(GLMLog), tolerance = 1e-3)
  expect_equal(logLik(FitLog), logLik(GLMLog), tolerance = 1e-3)
  expect_equal(extractAIC(FitLog), extractAIC(GLMLog), tolerance = 1e-3)
  expect_equal(vcov(FitLog), vcov(GLMLog), tolerance = 1e-3)
  expect_equal(coef(FitLog, type = "all"), as.data.frame(coef(summary(GLMLog))), 
               tolerance = 1e-3)
  expect_equal(unname(sigma(FitLog)), c(1, 1))
  expect_equal(family(FitLog), family(GLMLog), tolerance = 1e-3)
  
  ## Square root
  expect_equal(coef(FitSqrt), coef(GLMSqrt), tolerance = 1e-3)
  expect_equal(predict(FitSqrt), predict(GLMSqrt, type = "response"), 
               tolerance = 1e-3)
  expect_equal(residuals(FitSqrt), residuals(GLMSqrt, type = "pearson"), 
               tolerance = 1e-3)
  expect_equal(deviance(FitSqrt), deviance(GLMSqrt), tolerance = 1e-3)
  expect_equal(logLik(FitSqrt), logLik(GLMSqrt), tolerance = 1e-3)
  expect_equal(extractAIC(FitSqrt), extractAIC(GLMSqrt), tolerance = 1e-3)
  expect_equal(vcov(FitSqrt), vcov(GLMSqrt), tolerance = 1e-3)
  expect_equal(coef(FitSqrt, type = "all"), as.data.frame(coef(summary(GLMSqrt))), 
               tolerance = 1e-3)
  expect_equal(unname(sigma(FitSqrt)), c(1, 1))
  expect_equal(family(FitSqrt), family(GLMSqrt), tolerance = 1e-3)
})
