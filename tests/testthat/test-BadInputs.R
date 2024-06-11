# Testing bad inputs to functions
## BranchGLM
test_that("BranchGLM bad inputs", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 10), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(11, sd = 0.1)
  y <- exp(x %*% beta)
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  colnames(Data)[1] <- "y"
  
  ### formula
  expect_error(BranchGLM(1, data = Data, family = "gamma", link = "log"))
  expect_error(BranchGLM(1:2, data = Data, family = "gamma", link = "log"))
  expect_error(BranchGLM(~ ., data = Data, family = "gamma", link = "log"))
  expect_error(BranchGLM(y ~ apple, data = Data, family = "gamma", link = "log"))
  
  ### data
  expect_error(BranchGLM(y ~ ., data = cbind(y, x[, 1]), family = "gamma", link = "log"))
  
  ### family and link
  expect_error(BranchGLM(y ~ ., data = Data, family = 1:2, link = "log"))
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = 1:2))
  expect_error(BranchGLM(y ~ ., data = Data, family = "GamMA", link = "log"), NA)
  expect_error(BranchGLM(y ~ ., data = Data, family = "GamMA", link = "LOg"), NA)
  
  ### fitting parameters
  #### tol
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", tol = "apple"))
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", tol = 1:2))
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", tol = -1))
  
  #### grads
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", grads = "apple"))
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", grads = 1:2))
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", grads = -1))
  
  #### maxit
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", maxit = "apple"))
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", maxit = 1:2))
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", maxit = -1))
  
  #### nthreads
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", nthreads = "apple"))
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", nthreads = 1:2))
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", nthreads = -1))
  
  #### parallel
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", nthreads = "apple"))
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", nthreads = 1:2))
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", nthreads = -1))
  
  #### init
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", init = "apple"))
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", 
                         init = rep(NA_real_, 11)))
  
  #### fit
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", fit = "apple"))
  
  #### keepData
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", keepData = "apple"))
  
  #### keepY
  expect_error(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", keepY = "apple"))
  
  ## BranchGLM.fit
  ### x and y
  expect_error(BranchGLM.fit(NA, y, family = "gamma", link = "log"))
  expect_error(BranchGLM.fit(x, as.factor(round(y)), family = "gamma", link = "log"))
  expect_error(BranchGLM.fit(x, y[1:20], family = "gamma", link = "log"))
  
  ### offset
  expect_error(BranchGLM.fit(x, y, family = "gamma", link = "log", offset = x[1:20, 1]))
  expect_error(BranchGLM.fit(x, y, family = "gamma", link = "log", 
                             offset = as.factor(round(x[, 1]))))
  expect_error(BranchGLM.fit(x, y, family = "gamma", link = "log", offset = "apple"))
  
  ### family and link
  expect_error(BranchGLM.fit(x, y, family = 1:2, link = "log"))
  expect_error(BranchGLM.fit(x, y, family = "gamma", link = 1:2))
  expect_error(BranchGLM.fit(x, y, family = "GamMA", link = "log"), NA)
  expect_error(BranchGLM.fit(x, y, family = "GamMA", link = "LOg"), NA)
  
  ### fitting parameters
  #### tol
  expect_error(BranchGLM.fit(x, y, family = "gamma", link = "log", tol = "apple"))
  expect_error(BranchGLM.fit(x, y, family = "gamma", link = "log", tol = 1:2))
  expect_error(BranchGLM.fit(x, y, family = "gamma", link = "log", tol = -1))
  
  #### grads
  expect_error(BranchGLM.fit(x, y, family = "gamma", link = "log", grads = "apple"))
  expect_error(BranchGLM.fit(x, y, family = "gamma", link = "log", grads = 1:2))
  expect_error(BranchGLM.fit(x, y, family = "gamma", link = "log", grads = -1))
  
  #### maxit
  expect_error(BranchGLM.fit(x, y, family = "gamma", link = "log", maxit = "apple"))
  expect_error(BranchGLM.fit(x, y, family = "gamma", link = "log", maxit = 1:2))
  expect_error(BranchGLM.fit(x, y, family = "gamma", link = "log", maxit = -1))
  
  #### nthreads
  expect_error(BranchGLM.fit(x, y, family = "gamma", link = "log", nthreads = "apple"))
  expect_error(BranchGLM.fit(x, y, family = "gamma", link = "log", nthreads = 1:2))
  expect_error(BranchGLM.fit(x, y, family = "gamma", link = "log", nthreads = -1))
  
  #### parallel
  expect_error(BranchGLM.fit(x, y, family = "gamma", link = "log", nthreads = "apple"))
  expect_error(BranchGLM.fit(x, y, family = "gamma", link = "log", nthreads = 1:2))
  expect_error(BranchGLM.fit(x, y, family = "gamma", link = "log", nthreads = -1))
  
  #### init
  expect_error(BranchGLM.fit(x, y, family = "gamma", link = "log", init = "apple"))
  expect_error(BranchGLM.fit(x, y, family = "gamma", link = "log", 
                         init = rep(NA_real_, 11)))
  
  ### plot
  nofit <- BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", fit = FALSE)
  noy <- BranchGLM(y ~ ., data = Data, family = "gamma", link = "log", keepY = FALSE)
  fit <- BranchGLM(y ~ ., data = Data, family = "gamma", link = "log")
  expect_error(plot(nofit))
  expect_error(plot(noy))
  expect_error(plot(fit), NA)
  
})

## predict.BranchGLM
test_that("predict.BranchGLM bad inputs", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 10), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(11, sd = 0.1)
  y <- exp(x %*% beta)
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  colnames(Data)[1] <- "y"
  Fit <- BranchGLM(y ~ ., data = Data, family = "gaussian", link = "identity", 
                   offset = rep(0.01, 1000))
  
  ### newdata and offset
  expect_error(predict(Fit, newdata = as.matrix(Data)))
  expect_warning(predict(Fit, newdata = Data))
  expect_error(predict(Fit, newdata = Data, offset = rep(0.01, 100)))
  
  ### type
  expect_error(predict(Fit, type = "apple"))
  expect_error(predict(Fit, type = 1:2))
  expect_error(predict(Fit, newdata = Data, type = "apple"))
  expect_error(predict(Fit, newdata = Data, type = 1:2))
  
  ### na.action
  expect_error(predict(Fit, na.action = "helper"))
  expect_error(predict(Fit, na.action = 1:2))
  
})

## confidence intervals
test_that("confidence intervals bad inputs", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 10), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(11, sd = 0.1)
  y <- exp(x %*% beta)
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  colnames(Data)[1] <- "y"
  Fit <- BranchGLM(y ~ ., data = Data, family = "gaussian", link = "identity", 
                   offset = rep(0.01, 1000))
  CI <- confint(Fit)
  
  ### confint.BranchGLM
  #### parm
  expect_error(confint(Fit, parm = NA))
  expect_error(confint(Fit, parm = "apple"))
  expect_error(confint(Fit, parm = c(1, 1e6)))
  
  #### level
  expect_error(confint(Fit, level = -1))
  expect_error(confint(Fit, level = c(0.95, 0.99)))
  expect_error(confint(Fit, level = "apple"))
  
  #### nthreads
  expect_error(confint(Fit, nthreads = -1))
  expect_error(confint(Fit, nthreads = 1:2))
  expect_error(confint(Fit, nthreads = "apple"))
  
  #### parallel
  expect_error(confint(Fit, parallel = 1:2))
  expect_error(confint(Fit, parallel = "apple"))
  
  ### plot.BranchGLMCIs
  #### which
  expect_error(plot(CI, which = "apple"))
  expect_error(plot(CI, which = 1:100))
  
  #### mary
  expect_error(plot(CI, mary = -1))
  expect_error(plot(CI, mary = "apple"))
  expect_error(plot(CI, mary = 1:2))
  
  ### PlotCI
  points <- CI$MLE
  
  #### CIs and points
  expect_error(plotCI(t(CI$CIs), points = points))
  expect_error(plotCI("apple", points = points))
  expect_error(plotCI(CI$CIs[1:10, ], points = points))
  expect_error(plotCI(CI$CIs, points = points[1:2]))
  
  #### las
  expect_error(plotCI(CI$CIs, points = points, las = 100))
  expect_error(plotCI(CI$CIs, points = points, las = 1:2))
  expect_error(suppressWarnings(plotCI(CI$CIs, points = points, las = "apple")))
  
  #### cex.y
  expect_error(plotCI(CI$CIs, points, cex.y = -1))
  expect_error(suppressWarnings(plotCI(CI$CIs, points, cex.y = "apple")))
  expect_error(plotCI(CI$CIs, points, cex.y = 1:2))
  
  #### decreasing
  expect_error(plotCI(CI$CIs, points, decreasing = "apple"))
  expect_error(plotCI(CI$CIs, points, decreasing = c(TRUE, FALSE)))
  expect_error(plotCI(CI$CIs, points, decreasing = Data))
})

## VariableSelection
test_that("VariableSelection bad inputs", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 10), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(11, sd = 0.1)
  y <- exp(x %*% beta)
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  colnames(Data)[1] <- "y"
  Fit <- BranchGLM(y ~ ., data = Data, family = "gaussian", link = "identity")
  
  ### keep
  expect_error(VariableSelection(Fit, keep = c("apple", "diag", 1)))
  expect_error(VariableSelection(Fit, keep = NA_character_))
  
  ### keepintercept
  expect_error(VariableSelection(Fit, keepintercept = c("apple", "diag", 1)))
  expect_error(VariableSelection(Fit, keepintercept = NA_character_))
  
  ### metric
  expect_error(VariableSelection(Fit, metric = c("AIC", "BIC")))
  expect_error(VariableSelection(Fit, metric = 3))
  
  ### type
  expect_error(VariableSelection(Fit, type = c("AIC", "BIC")))
  expect_error(VariableSelection(Fit, type = NA_character_))
  
  ### bestmodels
  expect_error(VariableSelection(Fit, bestmodels = 0))
  expect_error(VariableSelection(Fit, bestmodels = 1:2))
  expect_error(VariableSelection(Fit, bestmodels = "apple"))
  
  ### type
  expect_error(VariableSelection(Fit, cutoff = -1))
  expect_error(VariableSelection(Fit, cutoff = 1:2))
  expect_error(VariableSelection(Fit, cutoff = "apple"))
  expect_error(VariableSelection(Fit, cutoff = 2, bestmodels = 10))
  
  ### nthreads
  expect_error(VariableSelection(Fit, nthreads = -1))
  expect_error(VariableSelection(Fit, nthreads = 1:2))
  expect_error(VariableSelection(Fit, nthreads = "apple"))
  
  ### parallel
  expect_error(VariableSelection(Fit, parallel = 1:2))
  expect_error(VariableSelection(Fit, parallel = "apple"))
  
  ### maxsize
  #### maxsize now defunct
  # expect_error(VariableSelection(Fit, maxsize = -1))
  # expect_error(VariableSelection(Fit, maxsize = 1:2))
  # expect_error(VariableSelection(Fit, maxsize = "apple"))
  
  ### showprogress
  expect_error(VariableSelection(Fit, showprogress = -1))
  expect_error(VariableSelection(Fit, showprogress = 1:2))
  expect_error(VariableSelection(Fit, showprogress = "apple"))
  
})
  
## BranchGLMVS methods
test_that("BranchGLMVS methods bad inputs", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 10), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(11, sd = 0.1)
  y <- exp(x %*% beta)
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  colnames(Data)[1] <- "y"
  Fit <- BranchGLM(y ~ ., data = Data, family = "gaussian", link = "identity", 
                   offset = rep(0.01, 1000))
  VS <- VariableSelection(Fit)
  
  ### predict.BranchGLMVS
  expect_error(predict(VS, newdata = as.matrix(Data)))
  expect_warning(predict(VS, newdata = Data))
  expect_error(predict(VS, newdata = Data, offset = rep(0.01, 100)))
  expect_error(predict(VS, which = 0))
  expect_error(predict(VS, which = 100))
  expect_error(predict(VS, which = 1:2))
  expect_error(predict(VS, which = "apple"))
  
  ### coef.BranchGLMVS
  expect_error(coef(VS, which = 0))
  expect_error(coef(VS, which = 100))
  expect_error(coef(VS, which = "apple"))
  
  ### plot.BranchGLMVS
  #### ptype
  expect_error(plot(VS, ptype = "apple"))
  expect_error(plot(VS, ptype = 1))
  expect_error(plot(VS, ptype = c("variables", "variables")))
  
  #### cols
  expect_error(plot(VS, cols = 1:3, ptype = "variables"))
  expect_error(plot(VS, cols = "red", ptype = "variables"))
  
  #### marnames
  expect_error(plot(VS, marnames = -1))
  expect_error(plot(VS, marnames = "apple"))
  expect_error(plot(VS, marnames = 1:2))
  
  #### cex
  expect_error(plot(VS, cex.axis = -1))
  expect_error(suppressWarnings(plot(VS, cex.axis = "apple")))
  expect_error(plot(VS, cex.axis = 1:2))
  expect_error(plot(VS, cex.names = -1))
  expect_error(suppressWarnings(plot(VS, cex.names = "apple")))
  expect_error(plot(VS, cex.names = 1:2))
  expect_error(plot(VS, cex.lab = -1))
  expect_error(suppressWarnings(plot(VS, cex.lab = "apple")))
  expect_error(plot(VS, cex.lab = 1:2))
  expect_error(plot(VS, cex.legend = -1))
  expect_error(suppressWarnings(plot(VS, cex.legend = "apple")))
  
})
