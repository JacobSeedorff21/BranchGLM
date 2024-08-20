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
                                 parallel = TRUE, nthreads = 2)
  
  expect_equal(coef(LinearVS), coef(LinearVS2))
  
  ### Forward selection with linear regression
  LinearForward <- VariableSelection(LinearFit, type = "forward")
  LinearForward2 <- VariableSelection(Sepal.Length ~ ., data = Data, family = "gaussian", 
                                 link = "identity", type = "forward", 
                                 parallel = TRUE, nthreads = 2)
  
  expect_equal(coef(LinearForward), coef(LinearForward2))
  
  ### Backward elimination with linear regression
  LinearBackward <- VariableSelection(LinearFit, type = "backward")
  LinearBackward2 <- VariableSelection(Sepal.Length ~ ., data = Data, family = "gaussian", 
                                 link = "identity", type = "backward", 
                                 parallel = TRUE, nthreads = 2)
  
  expect_equal(coef(LinearBackward), coef(LinearBackward2))
  
  ### Predict should work even if not all levels are available in newdata
  #### Checking for object obtained via BranchGLM function
  newdata <- Data[1,]
  newdata$Species <- as.factor(as.character(newdata$Species))
  expect_equal(predict(LinearFit, newdata = newdata),
               predict(LinearFit, newdata = Data[1,]))
  
  #### Checking for object from VariableSelection function
  expect_equal(predict(LinearVS, newdata = newdata), 
               predict(LinearVS, newdata = Data[1,]))
  
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
                                 parallel = TRUE, nthreads = 2)
  
  expect_equal(coef(LogitVS), coef(LogitVS2))
  
  LogitVS3<- VariableSelection(supp ~ ., data = Data, family = "binomial", 
                                link = "logit", type = "switch branch and bound", 
                                parallel = TRUE, nthreads = 2)
  
  expect_equal(coef(LogitVS), coef(LogitVS3))
  
  ### Forward variable selection with logistic regression
  LogitVS <- VariableSelection(LogitFit, type = "forward")
  LogitVS2 <- VariableSelection(supp ~ ., data = Data, family = "binomial", 
                                link = "logit", type = "forward",
                                parallel = TRUE, nthreads = 2)
  
  expect_equal(coef(LogitVS), coef(LogitVS2))
  
  ### Backward variable selection with logistic regression
  LogitVS <- VariableSelection(LogitFit, type = "backward", metric = "BIC")
  LogitVS2 <- VariableSelection(supp ~ ., data = Data, family = "binomial", 
                                link = "logit", type = "backward", metric = "BIC", 
                                parallel = TRUE, nthreads = 2)
  
  expect_equal(coef(LogitVS), coef(LogitVS2))
  
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
                         parallel = TRUE, nthreads = 2) |>
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
  expect_equal(BranchGLM(y ~ ., data = Data, family = "poisson", link = "log", parallel = TRUE, 
                         nthreads = 2)$coefficients, 
               BranchGLM(y ~ ., data = Data, family = "poisson", link = "log")$coefficients, 
               tolerance = .Machine$double.eps^(.25))
  
  ## Checking variable selection
  ### branch and bound
  expect_equal(coef(VariableSelection(y ~ ., data = Data, family = "poisson", 
                                 link = "log", parallel = TRUE, nthreads = 2, type = "branch and bound")), 
               coef(VariableSelection(y ~ ., data = Data, family = "poisson", 
                                 link = "log", parallel = FALSE, type = "branch and bound")))
  
  ### forward selection
  expect_equal(coef(VariableSelection(y ~ ., data = Data, family = "poisson", 
                                 link = "log", parallel = TRUE, nthreads = 2, type = "forward")), 
               coef(VariableSelection(y ~ ., data = Data, family = "poisson", 
                                 link = "log", parallel = FALSE, type = "forward")))
  ### backward selection
  expect_equal(coef(VariableSelection(y ~ ., data = Data, family = "poisson", 
                                 link = "log", parallel = TRUE, nthreads = 2, type = "backward")), 
               coef(VariableSelection(y ~ ., data = Data, family = "poisson", 
                                 link = "log", parallel = FALSE, type = "backward")))
  ### Checking offset predictions
  MyBranch <- BranchGLM(y ~ ., data = Data, family = "poisson", link = "log", offset = rep(1, nrow(Data)))
  
  #### Don't expect error when offset is provided for new data
  expect_error(predict(MyBranch, newdata = Data, offset = rep(1, nrow(Data))), NA)
  expect_error(predict(MyBranch, newdata = Data, type = "linpreds", offset = rep(1, nrow(Data))), NA)
  
  #### Don't expect error when offset isn't provided for and no new data
  expect_error(predict(MyBranch), NA)
  expect_error(predict(MyBranch, type = "linpreds"), NA)
  
  
  #### Expect error when offset is not supplied for new data
  expect_warning(predict(MyBranch, newdata = Data))
  expect_warning(predict(MyBranch, newdata = Data, type = "linpreds"))
  
  ### Checking offset predictions
  MyBranch <- VariableSelection(y ~ ., data = Data, family = "poisson", 
                                link = "log", parallel = TRUE, nthreads = 2, type = "forward", 
                                offset = rep(1, nrow(Data)))
  
  expect_error(predict(MyBranch, newdata = Data, offset = rep(1, nrow(Data))), NA)
  expect_error(predict(MyBranch, newdata = Data, type = "linpreds", offset = rep(1, nrow(Data))), NA)
  })

### Testing gamma regression
test_that("gamma regression works", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 15), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(16)
  y <- rgamma(n = 1000, shape = 1 / 2, scale = exp(x %*% beta) * 2)
  
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  
  ## Finding gamma regression
  expect_equal(BranchGLM(y ~ ., data = Data, family = "gamma", link = "log")$dispersion[[1]], 
               2, tolerance = 1e-1)
  
  ### forward selection
  expect_error(VariableSelection(y ~ ., data = Data, family = "gamma", link = "log", 
                                 type = "forward", method = "Fisher"), NA)
  ### backward selection
  expect_equal(coef(VariableSelection(y ~ ., data = Data, family = "gamma", 
                                  link = "log", parallel = TRUE, type = "backward", 
                                  nthreads = 2)), 
               coef(VariableSelection(y ~ ., data = Data, family = "gamma", 
                                 link = "log", parallel = FALSE, type = "backward")))
  
})

### Testing variable selection with interactions
test_that("Interactions work", {
  library(BranchGLM)
  set.seed(8621)
  x <- sapply(rep(0, 5), rnorm, n = 1000, simplify = TRUE)
  x <- cbind(1, x)
  beta <- rnorm(6)
  y <- rgamma(n = 1000, shape = 1 / 2, scale = exp(x %*% beta) * 2)
  
  Data <- cbind(y, x[,-1]) |>
    as.data.frame()
  ### forward selection
  expect_error(VariableSelection(y ~ .*., data = Data, family = "gamma", link = "log", 
                                 type = "forward", method = "Fisher"), NA)
  ### backward selection
  expect_error(VariableSelection(y ~ .*., data = Data, family = "gamma", link = "log", 
                                 type = "backward", method = "Fisher"), NA)
  
  ### branch and bound selection
  expect_error(BB <- VariableSelection(y ~ .*., data = Data, family = "gamma", link = "log", 
                                 type = "branch and bound", method = "Fisher"), NA)
  
  ### backward branch and bound selection
  expect_error(BBB <- VariableSelection(y ~ .*., data = Data, family = "gamma", link = "log", 
                                 type = "backward branch and bound", method = "Fisher"), NA)
  
  ### switch branch and bound selection
  expect_error(SBB <-VariableSelection(y ~ .*., data = Data, family = "gamma", link = "log", 
                                 type = "switch branch and bound", method = "Fisher"), NA)
  
  ## Checking for same results
  expect_equal(coef(BB), coef(BBB))
  expect_equal(coef(BB), coef(SBB))
  
})
