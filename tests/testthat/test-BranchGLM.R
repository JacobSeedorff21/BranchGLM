### Iris linear regression tests
test_that("linear regression works", {
  library(BranchGLM)
  Data <- iris
  
  ## Linear regression tests
  ### Fitting model
  LinearFit <- BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", 
                         link = "identity")
  expect_equal(LinearFit$iterations, 1)
  
  ### Branch and bound with linear regression
  LinearVS <- VariableSelection(LinearFit, type = "branch and bound")
  LinearVS2 <- VariableSelection(Sepal.Length ~ ., data = Data, family = "gaussian", 
                                 link = "identity", type = "branch and bound", 
                                 parallel = TRUE)
  
  expect_equal(LinearVS$finalmodel$coefficients, LinearVS2$finalmodel$coefficients)
  
  ### Forward selection with linear regression
  LinearForward <- VariableSelection(LinearFit, type = "forward")
  LinearForward2 <- VariableSelection(Sepal.Length ~ ., data = Data, family = "gaussian", 
                                 link = "identity", type = "forward", 
                                 parallel = TRUE)
  
  expect_equal(LinearForward$finalmodel$coefficients, LinearForward2$finalmodel$coefficients)
  
  ### Backward elimination with linear regression
  LinearBackward <- VariableSelection(LinearFit, type = "backward")
  LinearBackward2 <- VariableSelection(Sepal.Length ~ ., data = Data, family = "gaussian", 
                                 link = "identity", type = "backward", 
                                 parallel = TRUE)
  
  expect_equal(LinearBackward$finalmodel$coefficients, LinearBackward2$finalmodel$coefficients)
  
})

### Toothgrowth regression tests
test_that("binomial regression and stuff works", {
  library(BranchGLM)
  Data <- ToothGrowth
  
  ## Linear regression tests
  ### Fitting model
  LogitFit <- BranchGLM(supp ~ ., data = Data, family = "binomial", 
                         link = "logit")
  
  ### Branch and bound variable selection with logistic regression
  LogitVS <- VariableSelection(LogitFit, type = "branch and bound")
  LogitVS2 <- VariableSelection(supp ~ ., data = Data, family = "binomial", 
                                 link = "logit", type = "branch and bound", 
                                 parallel = TRUE)
  
  expect_equal(LogitVS$finalmodel$coefficients, LogitVS2$finalmodel$coefficients)
  
  ### Forward variable selection with logistic regression
  LogitVS <- VariableSelection(LogitFit, type = "forward")
  LogitVS2 <- VariableSelection(supp ~ ., data = Data, family = "binomial", 
                                link = "logit", type = "forward",
                                parallel = TRUE)
  
  expect_equal(LogitVS$finalmodel$coefficients, LogitVS2$finalmodel$coefficients)
  
  ### Backward variable selection with logistic regression
  LogitVS <- VariableSelection(LogitFit, type = "backward", metric = "BIC")
  LogitVS2 <- VariableSelection(supp ~ ., data = Data, family = "binomial", 
                                link = "logit", type = "backward", metric = "BIC", 
                                parallel = TRUE)
  
  expect_equal(LogitVS$finalmodel$coefficients, LogitVS2$finalmodel$coefficients)
  
  # Checking binomial utility functions
  ## Table
  ### BranchGLM
  LogitTable <- Table(LogitFit)
  expect_equal(LogitTable$Table, matrix(c(17, 7, 13, 23), ncol = 2))
  
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
                         parallel = TRUE) |>
                 suppressWarnings())
  ### Testing backward selection
  expect_error(VariableSelection(V1 ~ ., data = Data, family = "gaussian", link = "identity",
                                 type = "backward") |>
                 suppressWarnings())
  expect_error(VariableSelection(V1 ~ ., data = Data, family = "gaussian", link = "identity",
                         parallel = TRUE, type = "backward") |>
                 suppressWarnings())
  
  ### Testing forward selection for no error
  expect_error(VariableSelection(V1 ~ ., data = Data, family = "gaussian", link = "identity", type = "forward") |>
                suppressWarnings(), NA)
  expect_error(VariableSelection(V1 ~ ., data = Data, family = "gaussian", link = "identity",
                                 parallel = TRUE, type = "forward") |>
                 suppressWarnings(), NA)
})

### Residual deviance tests
### Toothgrowth regression tests
test_that("residual deviance works", {
  library(BranchGLM)
  Data <- ToothGrowth
  
  ## Residual deviance tests
  ### Poisson
  PoissonFit <- BranchGLM(as.numeric(supp) ~ ., data = Data, family = "poisson", 
                        link = "log")
  GLMFit <- glm(as.numeric(supp) ~ ., data = Data, family = poisson)
  
  expect_equal(PoissonFit$resdev, GLMFit$residual.deviance)
  
  ## Logistic
  LogitFit <- BranchGLM(supp ~ ., data = Data, family = "binomial", 
                        link = "logit")
  GLMFit <- glm(as.numeric(supp) ~ ., data = Data, family = poisson)
  
  expect_equal(LogitFit$resdev, GLMFit$residual.deviance)
  
  ## Linear
  LinearFit <- BranchGLM(as.numeric(supp) ~ ., data = Data, family = "gaussian", 
                        link = "identity")
  GLMFit <- glm(as.numeric(supp) ~ ., data = Data, family = poisson)
  
  expect_equal(LinearFit$resdev, GLMFit$residual.deviance)
})
