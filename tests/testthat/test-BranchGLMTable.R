# Testing for BranchGLMTable
## Testing consistency
test_that("Consistency tests for BranchGLMTable", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 10), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(11)
  y <- rbinom(n = 1000, size = 1, p = pnorm(x %*% beta + rnorm(1000, sd = 3)))
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ## Fitting model and getting ROC
  Fit <- BranchGLM(y ~ ., data = Data, family = "binomial", link = "logit")
  myROC <- ROC(Fit)
  
  ## Getting confusion matrix
  T1 <- Table(Fit)
  T2 <- Table(round(predict(Fit, type = "response")), y)
  T3 <- Table(round(predict(Fit, type = "response")), factor(y))
  T4 <- Table(round(predict(Fit, type = "response")), as.logical(y))
  
  ### Testing results
  expect_equal(T1, T2)
  expect_equal(T1$table, T3$table)
  expect_equal(T1$table, T4$table)
  
  ## Getting AUC/cindex
  AUC1 <- AUC(Fit)
  AUC2 <- AUC(predict(Fit, type = "response"), y)
  AUC3 <- AUC(predict(Fit, type = "response"), factor(y))
  AUC4 <- AUC(predict(Fit, type = "response"), as.logical(y))
  
  ### This one is an approximation from ROC curve, so it will not be exactly the same
  AUC5 <- AUC(myROC)
  
  ### Testing results
  expect_equal(AUC1, AUC2)
  expect_equal(AUC1, AUC3)
  expect_equal(AUC1, AUC4)
  expect_equal(AUC1, AUC5, tolerance = 1e-2)
  
  ## Getting ROC curve
  ROC1 <- ROC(Fit)
  ROC2 <- ROC(predict(Fit, type = "response"), y)
  ROC3 <- ROC(predict(Fit, type = "response"), factor(y))
  ROC4 <- ROC(predict(Fit, type = "response"), as.logical(y))
  
  ### Testing results
  expect_equal(ROC1, ROC2)
  expect_equal(ROC1, ROC3)
  expect_equal(ROC1, ROC4)
})

## Bad input tests
test_that("Bad input tests for BranchGLMTable", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 10), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(11, sd = 0.1)
  y <- exp(x %*% beta)
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  colnames(Data)[1] <- "y"
  
  ## Fitting model
  Fit <- BranchGLM(y ~ ., data = Data, family = "gamma", link = "log")
  
  ## Getting confusion matrix
  expect_error(Table(Fit))
  expect_error(Table(predict(Fit, type = "response"), y))
  expect_error(Table(runif(10), "apple"))
  expect_error(Table("apple", rep(1, 10)))
  
  ## Getting AUC/cindex
  expect_error(AUC(Fit))
  expect_error(AUC(predict(Fit, type = "response"), y))
  expect_error(AUC(runif(10), "apple"))
  expect_error(AUC("apple", rep(1, 10)))
  
  ## Getting ROC
  expect_error(ROC(Fit))
  expect_error(ROC(predict(Fit, type = "response"), y))
  expect_error(ROC(runif(10), "apple"))
  expect_error(ROC("apple", rep(1, 10)))
})

## Bad input for plot functions
test_that("Bad input tests for BranchGLMTable plots", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 10), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(11)
  y <- rbinom(n = 1000, size = 1, p = pnorm(x %*% beta + rnorm(1000, sd = 3)))
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ## Fitting model and getting ROC
  Fit <- BranchGLM(y ~ ., data = Data, family = "binomial", link = "logit")
  Fit2 <- BranchGLM(y ~ ., data = Data, family = "binomial", link = "probit")
  myROC <- ROC(Fit)
  myROC2 <- ROC(Fit2)
  
  ## MultipleROCCurves
  ### legendpos
  expect_error(MultipleROCCurves(myROC, myROC2, legendpos = "apple"))
  
  ### title
  expect_error(MultipleROCCurves(myROC, myROC2, title = 1:2))
  
  ### colors
  expect_error(MultipleROCCurves(myROC, myROC2, colors = 1:3))
  expect_error(MultipleROCCurves(myROC, myROC2, colors = "red"))
  expect_error(MultipleROCCurves(myROC, myROC2, colors = c(NA, "sltlrs")))
  
  ### names
  expect_error(MultipleROCCurves(myROC, myROC2, names = 1:3))
  expect_error(MultipleROCCurves(myROC, myROC2, names = "apple"))
  expect_error(MultipleROCCurves(myROC, myROC2, names = Data))
  
  ### lty
  expect_error(MultipleROCCurves(myROC, myROC2, lty = 1:3))
  expect_error(MultipleROCCurves(myROC, myROC2, lty = "red"))
  expect_error(MultipleROCCurves(myROC, myROC2, lty = -1))
  
  ### lwd
  expect_error(MultipleROCCurves(myROC, myROC2, lty = 1:3))
  expect_error(MultipleROCCurves(myROC, myROC2, lty = "red"))
  expect_error(MultipleROCCurves(myROC, myROC2, lty = -1))
  
})