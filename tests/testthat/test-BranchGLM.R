### Iris linear regression tests
test_that("linear regression works", {
  library(BranchGLM)
  Data <- iris
  
  ## Linear regression tests
  ### Fitting model
  LinearFit <- BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", 
                         link = "identity")
  
  ### Checking that number of iterations for linear regression is 1
  expect_equal(LinearFit$iterations, 1)
  
  ### Branch and bound with linear regression
  LinearVS <- VariableSelection(LinearFit, type = "branch and bound")
  LinearVS2 <- VariableSelection(Sepal.Length ~ ., data = Data, family = "gaussian", 
                                 link = "identity", type = "branch and bound", 
                                 parallel = TRUE, nthreads = 1)
  
  expect_equal(LinearVS$finalmodel$coefficients, LinearVS2$finalmodel$coefficients)
  
  ### Forward selection with linear regression
  LinearForward <- VariableSelection(LinearFit, type = "forward")
  LinearForward2 <- VariableSelection(Sepal.Length ~ ., data = Data, family = "gaussian", 
                                 link = "identity", type = "forward", 
                                 parallel = TRUE, nthreads = 1)
  
  expect_equal(LinearForward$finalmodel$coefficients, LinearForward2$finalmodel$coefficients)
  
  ### Backward elimination with linear regression
  LinearBackward <- VariableSelection(LinearFit, type = "backward")
  LinearBackward2 <- VariableSelection(Sepal.Length ~ ., data = Data, family = "gaussian", 
                                 link = "identity", type = "backward", 
                                 parallel = TRUE, nthreads = 1)
  
  expect_equal(LinearBackward$finalmodel$coefficients, LinearBackward2$finalmodel$coefficients)
  expect_equal(LinearBackward$finalmodel$AIC, AIC(LinearBackward$finalmodel))
  
  ### Predict should work even if not all levels are available in newdata
  #### Checking for object obtained via BranchGLM function
  newdata <- Data[1,]
  newdata$Species <- as.factor(as.character(newdata$Species))
  expect_equal(predict(LinearFit, newdata = newdata),
               predict(LinearFit, newdata = Data[1,]))
  
  #### Checking for object from VariableSelection function
  expect_equal(predict(LinearVS$finalmode, newdata = newdata), 
               predict(LinearVS$finalmode, newdata = Data[1,]))
  
})

### Toothgrowth regression tests
test_that("binomial regression and stuff works", {
  library(BranchGLM)
  Data <- ToothGrowth
  
  ## Linear regression tests
  ### Fitting model with BranchGLM
  LogitFit <- BranchGLM(supp ~ ., data = Data, family = "binomial", 
                         link = "logit")
  
  ### Fitting model with BranchGLM.fit
  x <- model.matrix(supp ~ ., data = Data)
  y <- Data$supp
  LogitFit2 <- BranchGLM.fit(x, as.numeric(y) - 1, family = "binomial", link = "logit")
  
  ### Checking that both approaches give same results
  LogitFitCoef <- LogitFit$coefficients
  row.names(LogitFitCoef) <- NULL
  expect_equal(LogitFitCoef, LogitFit2$coefficients)
  
  ### Branch and bound variable selection with logistic regression
  ### Checking that each branch and method gives same results
  LogitVS <- VariableSelection(LogitFit, type = "branch and bound")
  LogitVS2 <- VariableSelection(supp ~ ., data = Data, family = "binomial", 
                                 link = "logit", type = "backward branch and bound", 
                                 parallel = TRUE)
  
  expect_equal(LogitVS$finalmodel$coefficients, LogitVS2$finalmodel$coefficients)
  
  LogitVS3<- VariableSelection(supp ~ ., data = Data, family = "binomial", 
                                link = "logit", type = "switch branch and bound", 
                                parallel = TRUE)
  
  expect_equal(LogitVS$finalmodel$coefficients, LogitVS3$finalmodel$coefficients)
  
  ### Forward variable selection with logistic regression
  LogitVS <- VariableSelection(LogitFit, type = "forward")
  LogitVS2 <- VariableSelection(supp ~ ., data = Data, family = "binomial", 
                                link = "logit", type = "forward",
                                parallel = TRUE, nthreads = 1)
  
  expect_equal(LogitVS$finalmodel$coefficients, LogitVS2$finalmodel$coefficients)
  
  ### Backward variable selection with logistic regression
  LogitVS <- VariableSelection(LogitFit, type = "backward", metric = "BIC")
  LogitVS2 <- VariableSelection(supp ~ ., data = Data, family = "binomial", 
                                link = "logit", type = "backward", metric = "BIC", 
                                parallel = TRUE, nthreads = 1)
  
  expect_equal(LogitVS$finalmodel$coefficients, LogitVS2$finalmodel$coefficients)
  
  # Checking binomial utility functions
  ## Table
  ### BranchGLM
  LogitTable <- Table(LogitFit)
  expect_equal(LogitTable$table, matrix(c(17, 7, 13, 23), ncol = 2))
  
  ### Numeric preds
  preds <- predict(LogitFit)
  
  #### With factor y
  expect_error(Table(preds, Data$supp), NA)
  
  #### with numeric y
  expect_error(Table(preds, as.numeric(Data$supp) - 1), NA)
  
  #### with boolean y
  expect_error(Table(preds, Data$supp == "VC"), NA)
  
  ### ROC
  #### BranchGLM and factor y
  LogitROC <- ROC(LogitFit)
  LogitROC2 <- ROC(preds, Data$supp)
  expect_equal(LogitROC$Info, LogitROC2$Info)
  
  #### numeric y and boolean y
  expect_equal(ROC(preds, as.numeric(Data$supp) - 1)$Info, ROC(preds, Data$supp == "VC")$Info)
  
  ### AUC
  #### BranchGLM
  expect_equal(AUC(LogitFit), 0.71277778)
  
  #### BranchGLMROC
  expect_error(AUC(LogitROC), NA)
  
  #### factor y
  expect_equal(AUC(preds, Data$supp), 0.71277778)
  
  #### numeric y
  expect_equal(AUC(preds, as.numeric(Data$supp) - 1), 0.71277778)
  
  #### boolean y
  expect_equal(AUC(preds, Data$supp == "VC"), 0.71277778)
})

### Testing for fisher info errors
test_that("non-invertible info works", {
  library(BranchGLM)
  set.seed(199861)
  x <- sapply(1:25, rnorm, n = 10, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(26)
  y <- x %*% beta + rnorm(10)
  Data <- cbind(y, x) |>
          as.data.frame()
  
  ## Testing functions with non-invertible fisher info
  ### Testing BranchGLM
  expect_error(BranchGLM(V1 ~ ., data = Data, family = "gaussian", link = "identity") |>
                 suppressWarnings())
  expect_error(BranchGLM(V1 ~ ., data = Data, family = "gaussian", link = "identity",
                         parallel = TRUE, nthreads = 1) |>
                 suppressWarnings())
  ### Testing backward selection
  expect_error(VariableSelection(V1 ~ ., data = Data, family = "gaussian", link = "identity",
                                 type = "backward") |>
                 suppressWarnings())
  
  ### Testing forward selection for no error
  expect_error(VariableSelection(V1 ~ ., data = Data, family = "gaussian", link = "identity", type = "forward") |>
                suppressWarnings(), NA)
})

### Testing poisson regression
test_that("poisson regression works", {
  library(BranchGLM)
  set.seed(199861)
  x <- sapply(rep(0, 25), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(26, sd = .5)
  beta[4:26] <- 0
  y <- rpois(n = 1000, lambda = exp(x %*% beta))
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ## Fitting poisson regression
  expect_equal(BranchGLM(y ~ ., data = Data, family = "poisson", link = "log", parallel = TRUE)$coefficients, 
               BranchGLM(y ~ ., data = Data, family = "poisson", link = "log")$coefficients)
  
  ## Checking variable selection
  ### branch and bound
  expect_equal(VariableSelection(y ~ ., data = Data, family = "poisson", 
                                 link = "log", parallel = TRUE, type = "branch and bound")$finalmodel$coefficients, 
               VariableSelection(y ~ ., data = Data, family = "poisson", 
                                 link = "log", parallel = FALSE, type = "branch and bound")$finalmodel$coefficients)
  
  ### forward selection
  expect_equal(VariableSelection(y ~ ., data = Data, family = "poisson", 
                                 link = "log", parallel = TRUE, type = "forward")$finalmodel$coefficients, 
               VariableSelection(y ~ ., data = Data, family = "poisson", 
                                 link = "log", parallel = FALSE, type = "forward")$finalmodel$coefficients)
  ### backward selection
  expect_equal(VariableSelection(y ~ ., data = Data, family = "poisson", 
                                 link = "log", parallel = TRUE, type = "backward")$finalmodel$coefficients, 
               VariableSelection(y ~ ., data = Data, family = "poisson", 
                                 link = "log", parallel = FALSE, type = "backward")$finalmodel$coefficients)
  ### Checking offset predictions
  MyBranch <- BranchGLM(y ~ ., data = Data, family = "poisson", link = "log", offset = rep(1, nrow(Data)))
  
  expect_error(predict(MyBranch, newdata = Data), NA)
  expect_error(predict(MyBranch, newdata = Data, type = "linpreds"), NA)
  
  ### Checking offset predictions
  MyBranch <- VariableSelection(y ~ ., data = Data, family = "poisson", 
                                link = "log", parallel = TRUE, type = "forward", 
                                offset = rep(1, nrow(Data)))$finalmodel
  
  expect_error(predict(MyBranch, newdata = Data), NA)
  expect_error(predict(MyBranch, newdata = Data, type = "linpreds"), NA)
  })

### Testing gamma regression
test_that("gamma regression works", {
  library(BranchGLM)
  set.seed(862)
  x <- sapply(rep(0, 15), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(16)
  y <- rgamma(n = 1000, shape = 10, rate = exp(x %*% beta))
  
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ## Finding gamma regression
  expect_equal(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log")$dispersion, 
               0.097535718)
  
  ### forward selection should now work
  expect_error(VariableSelection(y ~ ., data = Data, family = "gamma", link = "log", 
                                 type = "forward", method = "Fisher"), NA)
  ### backward selection
  expect_equal(VariableSelection(y ~ ., data = Data, family = "gamma", 
                                  link = "log", parallel = TRUE, type = "backward")$finalmodel$coefficients, 
               VariableSelection(y ~ ., data = Data, family = "gamma", 
                                 link = "log", parallel = FALSE, type = "backward")$finalmodel$coefficients)
  
})
