# Testing variable importance
## Gamma regression
test_that("Testing VI methods gamma", {
  library(BranchGLM)
  set.seed(8621)
  
  ### Making sure x and beta are positive to use inverse link
  x <- sapply(rep(1, 3), rexp, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rexp(4, rate = 1)
  y <- rgamma(n = 1000, shape = 1, scale = 1 / (x %*% beta))
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ### Fitting upper model
  Fit <- BranchGLM(y ~ ., data = Data, family = "gamma", link = "inverse")
  VS <- VariableSelection(Fit, showprogress = FALSE)
  
  ### Getting VI
  VI1 <- VariableImportance(VS, showprogress = FALSE)
  VI2 <- VariableImportance(VS, VIMethod = "separate", showprogress = FALSE)
  
  ### Testing VI
  expect_equal(VI1$results, VI2$results)
  
  ### Testing VariableImportance.boot
  set.seed(123)
  VIB1 <- VariableImportance.boot(VI1, showprogress = FALSE, nboot = 10)
  set.seed(123)
  VIB2 <- VariableImportance.boot(VI2, showprogress = FALSE, nboot = 10)
  set.seed(123)
  VIB3 <- VariableImportance.boot(VS, showprogress = FALSE, nboot = 10)
  set.seed(123)
  VIB4 <- VariableImportance.boot(VS, VIMethod = "separate", showprogress = FALSE, nboot = 10)
  
  ### Testing VI
  expect_equal(VIB1$results, VIB2$results)
  expect_equal(VIB1, VIB3)
  expect_equal(VIB2, VIB4)
  
})

## Gaussian
test_that("Testing VI methods gaussian", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 3), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(4)
  y <- rnorm(n = 1000, mean = x %*% beta, sd = 2)
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ## Fitting upper model
  Fit <- BranchGLM(y ~ ., data = Data, family = "gaussian", link = "identity")
  VS <- VariableSelection(Fit, showprogress = FALSE)
  
  ### Getting VI
  VI1 <- VariableImportance(VS, showprogress = FALSE)
  VI2 <- VariableImportance(VS, VIMethod = "separate", showprogress = FALSE)
  
  ### Testing VI
  expect_equal(VI1$results, VI2$results)
  
  ### Testing barplot and print
  expect_error(barplot(VI1), NA)
  expect_error(barplot(VI1, horiz = FALSE), NA)
  expect_error(barplot(VI1, which = 1, horiz = FALSE))
  expect_error(barplot(VI1, which = c(-1, -4), horiz = FALSE), NA)
  expect_error(barplot(VI1, main = "apple"), NA)
  expect_error(barplot(VI1, modified = FALSE, decreasing = TRUE, which = 2), NA)
  expect_error(barplot(VI1), NA)
  expect_error(print(VI1), NA)
  
  ### Testing VariableImportance.boot
  set.seed(123)
  VIB1 <- VariableImportance.boot(VI1, showprogress = FALSE, nboot = 10)
  set.seed(123)
  VIB2 <- VariableImportance.boot(VI2, showprogress = FALSE, nboot = 10)
  set.seed(123)
  VIB3 <- VariableImportance.boot(VS, showprogress = FALSE, nboot = 10)
  set.seed(123)
  VIB4 <- VariableImportance.boot(VS, VIMethod = "separate", showprogress = FALSE, nboot = 10)
  
  #### Testing with VS object that had fit = false in BranchGLM
  Fit <- BranchGLM(y ~ ., data = Data, family = "gaussian", link = "identity", fit = FALSE)
  VS <- VariableSelection(Fit, showprogress = FALSE)
  
  ### Getting VI
  expect_error(VariableImportance(VS, showprogress = FALSE), NA)
  
  ### Testing VI
  expect_equal(VIB1$results, VIB2$results)
  expect_equal(VIB1, VIB3)
  expect_equal(VIB2, VIB4)
  
  ### Testing boxplot, hist, and print
  expect_error(boxplot(VIB1), NA)
  expect_error(boxplot(VIB1, which = 1))
  expect_error(boxplot(VIB1, which = c(-1, -4)), NA)
  expect_error(boxplot(VIB1, which = 2, linecol = "blue", linelwd = 3, horizontal = FALSE, 
                       las = 2), NA)
  expect_error(hist(VIB1), NA)
  expect_error(hist(VIB1, which = 1))
  expect_error(hist(VIB1, which = c(-1, -4)), NA)
  expect_error(hist(VIB1, linecol = "blue", linelwd = 3, which = 2:4), NA)
  expect_error(print(VIB1), NA)
  
  ## Testing with heuristic methods
  expect_error(VIFor <- VariableImportance(VariableSelection(Fit, type = "forward", 
                                          showprogress = FALSE), showprogress = FALSE), NA)
  expect_error(VIBack <- VariableImportance(VariableSelection(Fit, type = "backward", 
                                            showprogress = FALSE), showprogress = FALSE), NA)
  expect_error(VIFBack <- VariableImportance(VariableSelection(Fit, type = "fast backward", 
                                             showprogress = FALSE), showprogress = FALSE), NA)
  expect_error(VIDBack <- VariableImportance(VariableSelection(Fit, type = "double backward", 
                                             showprogress = FALSE), showprogress = FALSE), NA)
  expect_error(VIFDBack <- VariableImportance(VariableSelection(Fit, type = "fast double backward", 
                                              showprogress = FALSE), showprogress = FALSE), NA)
})

## Binomial
### Probit
test_that("Testing VI methods binomial", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 3), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(4)
  y <- rbinom(n = 1000, size = 1, p = pnorm(x %*% beta + rnorm(1000, sd = 3)))
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ## Fitting upper model
  Fit <- BranchGLM(y ~ ., data = Data, family = "binomial", link = "probit")
  VS <- VariableSelection(Fit, showprogress = FALSE)
  
  ### Getting VI
  VI1 <- VariableImportance(VS, showprogress = FALSE)
  VI2 <- VariableImportance(VS, VIMethod = "separate", showprogress = FALSE)
  
  ### Testing VI
  expect_equal(VI1$results, VI2$results)
  
  ### Testing VariableImportance.boot
  set.seed(123)
  VIB1 <- VariableImportance.boot(VI1, showprogress = FALSE, nboot = 10)
  set.seed(123)
  VIB2 <- VariableImportance.boot(VI2, showprogress = FALSE, nboot = 10)
  set.seed(123)
  VIB3 <- VariableImportance.boot(VS, showprogress = FALSE, nboot = 10)
  set.seed(123)
  VIB4 <- VariableImportance.boot(VS, VIMethod = "separate", showprogress = FALSE, nboot = 10)
  
  ### Testing VI
  expect_equal(VIB1$results, VIB2$results)
  expect_equal(VIB1, VIB3)
  expect_equal(VIB2, VIB4)
})

### cloglog
test_that("Testing VI methods binomial", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 3), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(4)
  y <- rbinom(n = 1000, size = 1, p = 1 - exp(-exp(x %*% beta + rnorm(1000, sd = 5))))
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ## Fitting upper model
  Fit <- BranchGLM(y ~ ., data = Data, family = "binomial", link = "cloglog")
  VS <- VariableSelection(Fit, showprogress = TRUE)
  
  ### Getting VI
  VI1 <- VariableImportance(VS, showprogress = FALSE)
  VI2 <- VariableImportance(VS, VIMethod = "separate", showprogress = FALSE)
  
  ### Testing VI
  expect_equal(VI1$results, VI2$results)
  
  ### Testing VariableImportance.boot
  set.seed(123)
  VIB1 <- VariableImportance.boot(VI1, showprogress = FALSE, nboot = 10)
  set.seed(123)
  VIB2 <- VariableImportance.boot(VI2, showprogress = FALSE, nboot = 10)
  set.seed(123)
  VIB3 <- VariableImportance.boot(VS, showprogress = FALSE, nboot = 10)
  set.seed(123)
  VIB4 <- VariableImportance.boot(VS, VIMethod = "separate", showprogress = FALSE, nboot = 10)
  
  ### Testing VI
  expect_equal(VIB1$results, VIB2$results)
  expect_equal(VIB1, VIB3)
  expect_equal(VIB2, VIB4)
})

## Poisson
test_that("Testing VI methods poisson", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 3), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(4, sd = 0.1)
  y <- rpois(n = 1000, lambda = exp(x %*% beta))
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ## Fitting upper model
  Fit <- BranchGLM(y ~ ., data = Data, family = "poisson", link = "sqrt")
  VS <- VariableSelection(Fit, showprogress = FALSE)
  
  ### Getting VI
  VI1 <- VariableImportance(VS, showprogress = FALSE)
  VI2 <- VariableImportance(VS, VIMethod = "separate", showprogress = FALSE)
  
  ### Testing VI
  expect_equal(VI1$results, VI2$results)
  
  ### Testing VariableImportance.boot
  set.seed(123)
  VIB1 <- VariableImportance.boot(VI1, showprogress = FALSE, nboot = 10)
  set.seed(123)
  VIB2 <- VariableImportance.boot(VI2, showprogress = FALSE, nboot = 10)
  set.seed(123)
  VIB3 <- VariableImportance.boot(VS, showprogress = FALSE, nboot = 10)
  set.seed(123)
  VIB4 <- VariableImportance.boot(VS, VIMethod = "separate", showprogress = FALSE, nboot = 10)
  
  ### Testing VI
  expect_equal(VIB1$results, VIB2$results)
  expect_equal(VIB1, VIB3)
  expect_equal(VIB2, VIB4)
  
})
