# Confidence intervals tests
## Comparing to glm
### Linear
test_that("Confidence intervals tests linear", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 10), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(11)
  y <- rnorm(n = 1000, mean = x %*% beta, sd = 1)
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ### Getting confidence intervals
  CIs <- confint(BranchGLM(y ~ ., data = Data, family = "gaussian", 
                                       link = "identity"))
  CIsWald <- confint.default(BranchGLM(y ~ ., data = Data, family = "gaussian", 
                                       link = "identity"))
  
  ### Getting glm confints
  LMCIs <- confint(lm(y ~ ., data = Data))
  LMWald <- confint.default(lm(y ~ ., data = Data))
  
  ### Checking results
  expect_equal(unname(CIs$CIs), unname(LMCIs), tolerance = 1e-2)
  expect_equal(CIsWald, LMWald, tolerance = 1e-2)
})

### Gamma
test_that("Confidence intervals tests gamma", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 10), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(11)
  y <- rgamma(n = 1000, shape = 1 / 2, scale = exp(x %*% beta) * 2)
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ### Getting confidence intervals
  Fish <- BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", 
                            method = "Fisher")
  FishCIs <- confint(Fish)
  FishCIsWald <- confint.default(Fish)
  BFGS <- BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", 
                            method = "BFGS")
  BFGSCIs <- confint(BFGS)
  BFGSCIsWald <- confint.default(BFGS)
  LBFGS <- BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", 
                             method = "LBFGS")
  LBFGSCIs <- confint(LBFGS)
  LBFGSCIsWald <- confint.default(LBFGS)
  
  ### glm
  GLM <- glm(y ~ ., data = Data, family = Gamma(link = "log"))
  GLMCIsWald <- confint.default(GLM)
  
  ### Checking results
  expect_equal(FishCIs, BFGSCIs, tolerance = 1e-2)
  expect_equal(FishCIs, LBFGSCIs, tolerance = 1e-2)
  expect_equal(unname(FishCIs$CIs), unname(GLMCIsWald), tolerance = 1e-2)
  expect_equal(FishCIsWald, BFGSCIsWald, tolerance = 1e-2)
  expect_equal(FishCIsWald, LBFGSCIsWald, tolerance = 1e-2)
  expect_equal(FishCIsWald, GLMCIsWald, tolerance = 1e-2)
})

### Logistic
test_that("Confidence intervals tests logistic", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 10), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(11)
  y <- rbinom(n = 1000, size = 1, p = 1 / (1 + exp(-x %*% beta)))
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ### Getting confidence intervals
  Fish <- BranchGLM(y ~ ., data = Data, family = "binomial", link = "logit", 
                    method = "Fisher")
  FishCIs <- confint(Fish)
  FishCIsWald <- confint.default(Fish)
  BFGS <- BranchGLM(y ~ ., data = Data, family = "binomial", link = "logit", 
                    method = "BFGS")
  BFGSCIs <- confint(BFGS)
  BFGSCIsWald <- confint.default(BFGS)
  LBFGS <- BranchGLM(y ~ ., data = Data, family = "binomial", link = "logit", 
                     method = "LBFGS")
  LBFGSCIs <- confint(LBFGS)
  LBFGSCIsWald <- confint.default(LBFGS)
  
  ### glm
  GLM <- glm(y ~ ., data = Data, family = binomial)
  GLMCIsWald <- confint.default(GLM)
  
  ### Checking results
  expect_equal(FishCIs, BFGSCIs, tolerance = 1e-2)
  expect_equal(FishCIs, LBFGSCIs, tolerance = 1e-2)
  expect_equal(unname(FishCIs$CIs), unname(GLMCIsWald), tolerance = 1e-2)
  expect_equal(FishCIsWald, BFGSCIsWald, tolerance = 1e-2)
  expect_equal(FishCIsWald, LBFGSCIsWald, tolerance = 1e-2)
  expect_equal(FishCIsWald, GLMCIsWald, tolerance = 1e-2)
})

### Poisson
test_that("Confidence intervals tests methods", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 10), rnorm, n = 1000, simplify = TRUE, sd = 0.1)
  x <- cbind(1, x)
  beta <- rnorm(11)
  y <- rpois(n = 1000, lambda = exp(x %*% beta))
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ### Getting confidence intervals
  Fish <- BranchGLM(y ~ ., data = Data, family = "poisson", link = "log", 
                    method = "Fisher")
  FishCIs <- confint(Fish)
  FishCIsWald <- confint.default(Fish)
  BFGS <- BranchGLM(y ~ ., data = Data, family = "poisson", link = "log", 
                    method = "BFGS")
  BFGSCIs <- confint(BFGS)
  BFGSCIsWald <- confint.default(BFGS)
  LBFGS <- BranchGLM(y ~ ., data = Data, family = "poisson", link = "log", 
                     method = "LBFGS")
  LBFGSCIs <- confint(LBFGS)
  LBFGSCIsWald <- confint.default(LBFGS)
  
  ### glm
  GLM <- glm(y ~ ., data = Data, family = poisson)
  GLMCIsWald <- confint.default(GLM)
  
  ### Checking results
  expect_equal(FishCIs, BFGSCIs, tolerance = 1e-2)
  expect_equal(FishCIs, LBFGSCIs, tolerance = 1e-2)
  expect_equal(unname(FishCIs$CIs), unname(GLMCIsWald), tolerance = 1e-2)
  expect_equal(FishCIsWald, BFGSCIsWald, tolerance = 1e-2)
  expect_equal(FishCIsWald, LBFGSCIsWald, tolerance = 1e-2)
  expect_equal(FishCIsWald, GLMCIsWald, tolerance = 1e-2)
})

## Testing which and parm
test_that("Testing which and parm", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 10), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(11)
  y <- rnorm(n = 1000, mean = x %*% beta, sd = 1)
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ### Getting confidence intervals
  expect_error(confint(BranchGLM(y ~ ., data = Data, family = "gaussian", link = "identity"), 
                       parm = 11:1), NA)
  expect_error(CI1 <- confint(BranchGLM(y ~ ., data = Data, family = "gaussian", link = "identity"), 
                              parm = 1:11), NA)
  expect_error(CI2 <- confint(BranchGLM(y ~ ., data = Data, family = "gaussian", link = "identity"), 
                 parm = c("(Intercept)", colnames(Data)[-1])), NA)
  expect_equal(CI1, CI2)
  
  ### Testing which in plot
  expect_error(plot(CI1, which = c("(Intercept)", colnames(Data)[-1])), NA)
  expect_error(plot(CI1, which = 1:11), NA)
  expect_error(plot(CI1, which = "AlL"), NA)
})

## Testing errors in CI functions
test_that("Testing errors in CI functions", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 10), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(11)
  y <- rnorm(n = 1000, mean = x %*% beta, sd = 1)
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ### Testing confidence intervals
  expect_error(confint(BranchGLM(y ~ ., data = Data, family = "gaussian", link = "identity"), 
                       parm = 24))
  expect_error(confint(BranchGLM(y ~ ., data = Data, family = "gaussian", link = "identity"), 
                       parm = "apple"))
  CI <- confint(BranchGLM(y ~ ., data = Data, family = "gaussian", link = "identity"))
  expect_error(plot(CI, which = 24))
  expect_error(plot(CI, which = "apple"))
  expect_error(plotCI(t(CI$CIs)))
})
