# Parameter recovery 
## Gamma
test_that("Gamma parameter recovery test", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 10), rnorm, n = 10000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(11, sd = 0.1)
  y <- rgamma(n = 10000, shape = 1, scale = exp(x %*% beta))
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ### Fitting models
  familyFish <- BranchGLM(y ~ ., data = Data, family = Gamma(link = "log"), 
                          method = "Fisher")
  Fish <- BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", 
                    method = "Fisher")
  BFGS <- BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", 
                    method = "BFGS")
  LBFGS <- BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", 
                     method = "LBFGS")
  
  ### Checking results
  #### Beta parameters
  names(beta) <- names(coef(Fish))
  expect_equal(familyFish, Fish)
  expect_equal(coef(Fish), beta, tolerance = 0.1)
  expect_equal(coef(BFGS), beta, tolerance = 0.1)
  expect_equal(coef(LBFGS), beta, tolerance = 0.1)
  
  #### Dispersion parameter
  expect_equal(Fish$dispersion[[1]], 1, tolerance = 0.1)
  expect_equal(BFGS$dispersion[[1]], 1, tolerance = 0.1)
  expect_equal(LBFGS$dispersion[[1]], 1, tolerance = 0.1)
})

## Gaussian
test_that("Gaussian parameter recovery test", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 10), rnorm, n = 10000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(11)
  y <- rnorm(n = 10000, mean = x %*% beta, sd = 2)
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ### Fitting models
  familyFish <- BranchGLM(y ~ ., data = Data, family = gaussian())
  Fish <- BranchGLM(y ~ ., data = Data, family = "gaussian", link = "identity")
  
  ### Checking results
  #### Beta parameters
  names(beta) <- names(coef(Fish))
  expect_equal(familyFish, Fish)
  expect_equal(coef(Fish), beta, tolerance = 0.1)
  
  #### Dispersion parameter
  expect_equal(Fish$dispersion[[1]], 4, tolerance = 0.1)
})

## Binomial
test_that("Binomial parameter recovery test", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 10), rnorm, n = 10000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(11)
  y <- rbinom(n = 10000, size = 1, p = 1 / (1 + exp(-x %*% beta)))
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ### Fitting models
  familyFish <- BranchGLM(y ~ ., data = Data, family = binomial(), 
                    method = "Fisher")
  Fish <- BranchGLM(y ~ ., data = Data, family = "binomial", link = "logit", 
                    method = "Fisher")
  BFGS <- BranchGLM(y ~ ., data = Data, family = "binomial", link = "logit", 
                    method = "BFGS")
  LBFGS <- BranchGLM(y ~ ., data = Data, family = "binomial", link = "logit", 
                     method = "LBFGS")
  
  ### Checking results
  #### Beta parameters
  names(beta) <- names(coef(Fish))
  expect_equal(familyFish, Fish)
  expect_equal(coef(Fish), beta, tolerance = 0.1)
  expect_equal(coef(BFGS), beta, tolerance = 0.1)
  expect_equal(coef(LBFGS), beta, tolerance = 0.1)
})

## Poisson
test_that("Poisson parameter recovery test", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 10), rnorm, n = 10000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(11, sd = 0.1)
  y <- rpois(n = 10000, lambda = exp(x %*% beta))
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ### Fitting models
  familyFish <- BranchGLM(y ~ ., data = Data, family = poisson(), 
                    method = "Fisher")
  Fish <- BranchGLM(y ~ ., data = Data, family = "poisson", link = "log", 
                    method = "Fisher")
  BFGS <- BranchGLM(y ~ ., data = Data, family = "poisson", link = "log", 
                    method = "BFGS")
  LBFGS <- BranchGLM(y ~ ., data = Data, family = "poisson", link = "log", 
                     method = "LBFGS")
  
  ### Checking results
  #### Beta parameters
  names(beta) <- names(coef(Fish))
  expect_equal(familyFish, Fish)
  expect_equal(coef(Fish), beta, tolerance = 0.1)
  expect_equal(coef(BFGS), beta, tolerance = 0.1)
  expect_equal(coef(LBFGS), beta, tolerance = 0.1)
})
