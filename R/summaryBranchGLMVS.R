#' Summary Method for BranchGLMVS
#' @param object a \code{BranchGLMVS} object.
#' @param ... further arguments passed to other methods.
#' @return An object of class \code{summary.BranchGLMVS} which is a list with the 
#' following components
#' \item{\code{results}}{ a data frame which has the metric values for the best models along 
#' with the variables included in each model}
#' \item{\code{initmodel}}{ the initial \code{BranchGLM} object that was supplied to the 
#' \code{VariableSelection} function}
#' \item{\code{formulas}}{ a list containing the formulas of the best models}
#' \item{\code{metric}}{ the metric used to perform variable selection}
#' 
#' @examples
#' 
#' Data <- iris
#' Fit <- BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", link = "identity")
#' 
#' # Doing branch and bound selection 
#' VS <- VariableSelection(Fit, type = "branch and bound", metric = "BIC", 
#' bestmodels = 10, showprogress = FALSE)
#' VS
#' 
#' ## Getting summary of the process
#' Summ <- summary(VS)
#' Summ
#' 
#' ## Plotting the BIC of the best models
#' plot(Summ, type = "b")
#' 
#' ## Plotting the variables in the best models
#' plot(Summ, ptype = "variables")
#' 
#' ## Getting the best model according to BIC
#' FinalModel <- fit(Summ, which = 1)
#' FinalModel
#'  
#' @export

summary.BranchGLMVS <- function(object, ...){
  
  # Getting whether each variables is included in each model
  BestModels <- t(object$bestmodels)
  BestModels[BestModels == -1] <- "kept"
  BestModels[BestModels == 0] <- "no"
  BestModels[BestModels == 1] <- "yes"
  df <- data.frame(BestModels, object$bestmetrics)
  colnames(df) <- c(object$names, object$metric)
  Models <- object$bestmodels
  if(!is.matrix(Models)){
    # if Models is a vector, then change it to a matrix
    Models <- matrix(Models, ncol = 1)
  }
  
  # Generating formulas for each of the best models
  formulas <- apply(Models, 2, FUN = function(x){
    tempnames <- object$names[x != 0]
    tempnames <- tempnames[which(tempnames != "(Intercept)")]
    if(length(tempnames) > 0){
      MyFormula <- as.formula(paste0(object$initmodel$yname, " ~ ", paste0(tempnames, collapse = "+")))
      if(!any(object$names == "(Intercept)")){
        MyFormula <- deparse1(MyFormula) |>
          paste0(" - 1") |>
          as.formula()
      }
    }else{
      # We can do this since we only include non-null models in bestmodels
      MyFormula <- formula(paste0(object$initmodel$yname, " ~ 1")) 
    }
    MyFormula
  }
  )
  MyList <- list("results" = df, 
                 "initmodel" = object$initmodel, 
                 "formulas" = formulas, 
                 "metric" = object$metric)
  
  return(structure(MyList, class = "summary.BranchGLMVS"))
}

#' Fits GLMs for summary.BranchGLMVS objects
#' @name fit
#' @param object a \code{summary.BranchGLMVS} object.
#' @param which a positive integer indicating which model to fit, 
#' the default is to fit the first model .
#' @param keepData Whether or not to store a copy of data and design matrix, the default 
#' is TRUE. If this is FALSE, then the results from this cannot be used inside of \code{VariableSelection}.
#' @param keepY Whether or not to store a copy of y, the default is TRUE. If 
#' this is FALSE, then the binomial GLM helper functions may not work and this 
#' cannot be used inside of \code{VariableSelection}.
#' @param ... further arguments passed to other methods.
#' @details The information needed to fit the GLM is taken from the original information 
#' supplied to the \code{VariableSelection} function.
#' @examples
#' Data <- iris
#' Fit <- BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", link = "identity")
#' 
#' # Doing branch and bound selection 
#' VS <- VariableSelection(Fit, type = "branch and bound", metric = "BIC", 
#' bestmodels = 10, showprogress = FALSE)
#' 
#' ## Getting summary of the process
#' Summ <- summary(VS)
#' 
#' ## Getting the best model according to BIC
#' FinalModel <- fit(Summ, which = 1)
#' FinalModel
#' 
#' ## Getting the 8th best model according to BIC
#' EighthModel <- fit(Summ, which = 8)
#' EighthModel
#' 
#' @return An object of class \code{\link{BranchGLM}}. 
#' @export
#' 
fit <- function(object, ...) {
  UseMethod("fit")
}

#' @rdname fit 
#' @export
fit.summary.BranchGLMVS <- function(object, which = 1, keepData = TRUE, keepY = TRUE, ...){
  if(!is.numeric(which) || which < 0 || which > length(object$formulas) || 
     which != as.integer(which)){
    stop("which must be a positive integer denoting the rank of the model to fit")
  }
  Model <- object$initmodel
  FinalModel <- BranchGLM(object$formulas[[which]], data = Model$data, family = Model$family, 
                          link = Model$link, method = Model$method, 
                          tol = Model$tol, maxit = Model$maxit, 
                          keepData = keepData, keepY = keepY)
  
  return(FinalModel)
}

#' Print Method for summary.BranchGLMVS
#' @param x a \code{summary.BranchGLMVS} object.
#' @param digits number of digits to display for information in the table.
#' @param ... further arguments passed to other methods.
#' @return The supplied \code{summary.BranchGLMVS} object.
#' @export

print.summary.BranchGLMVS <- function(x, digits = 4, ...){
  temp <- x$results
  temp[, ncol(temp)] <- round(temp[ncol(temp)], digits = digits)
  print(temp)
  return(invisible(x))
}

#' Plot Method for summary.BranchGLMVS
#' @param x a \code{summary.BranchGLMVS} object.
#' @param ptype the type of plot to produce, look at details for more explanation.
#' @param marx value used to determine how large to make margin of x-axis, this is 
#' only for ptype = "variables". If variable names are cut-off, consider increasing this, 
#' the default value is 7.
#' @param ... arguments passed to the generic plot methods.
#' @details The different values for ptype are as follows
#' \itemize{
#'  \item "metrics" for a plot that displays the metric values ordered by rank
#'  \item "variables" for a plot that displays which variables are in each of the top models
#' }
#' @examples
#' Data <- iris
#' Fit <- BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", link = "identity")
#' 
#' # Doing branch and bound selection 
#' VS <- VariableSelection(Fit, type = "branch and bound", metric = "BIC", bestmodels = 10, 
#' showprogress = FALSE)
#' VS
#' 
#' ## Getting summary of the process
#' Summ <- summary(VS)
#' Summ
#' 
#' ## Plotting the BIC of the best models
#' plot(Summ, type = "b")
#' 
#' ## Plotting the variables in the best models
#' plot(Summ, ptype = "variables")
#' 
#' @return This only produces a plot, nothing is returned.
#' @export

plot.summary.BranchGLMVS <- function(x, ptype = "metrics", marx = 7, ...){
  if(ptype == "metrics"){
    plot(1:nrow(x$results), x$results[, x$metric], 
         xlab = "Rank", ylab = x$metric, 
         main = paste0("Best Models Ranked by ", x$metric), 
         ...)
  }else if(ptype == "variables"){
    # This is inspired by the plot.regsubsets function
    n <- length(x$formulas)
    Names <- colnames(x$results)[-ncol(x$results)]
    z <- x$results[, -ncol(x$results)]
    z[z == "kept"] <- 2
    z[z == "no"] <- 1
    z[z == "yes"] <- 0
    z <- apply(z, 2, as.numeric)
    if(is.matrix(z)){
    z <- t(z)
    }else{
      z <- matrix(z, nrow = length(z))
    }
    y <- 1:ncol(z)
    x1 <- 1:nrow(z)
    
    # Creating image  
    oldmar <- par("mar")
    par(mar = c(marx, 5, 3, 8) + 0.1)
    
    if(all(z != 2)){
      # Do this if there were no variable kept
      image(x1, y, z, ylab = "", 
            xaxt = "n", yaxt = "n", xlab = "", 
            main = paste0("Best Models Ranked by ", x$metric), 
            col = c("deepskyblue", "indianred"), ...)
      legend(grconvertX(1, from = "npc"), grconvertY(1, from = "npc"), 
             legend = c("Included", "Excluded"), 
             fill = c("deepskyblue", "indianred"), xpd = TRUE)
    }else{
      # Do this if there were any kept variables
      image(x1, y, z, ylab = "", 
            xaxt = "n", yaxt = "n", xlab = "", 
            main = paste0("Best Models Ranked by ", x$metric), 
            col = c("deepskyblue", "indianred", "forestgreen"))
      legend(grconvertX(1, from = "npc"), grconvertY(1, from = "npc"), 
             legend = c("Included", "Excluded", "Kept"), 
             fill = c("deepskyblue", "indianred", "forestgreen"), xpd = TRUE)
    }
    
    # Adding lines
    abline(v = x1 + 0.5, h = y + 0.5)
    
    # Adding axis labels
    axis(1, at = x1, labels = Names, line = 1, las = 2)
    axis(2, at = y, labels = y, line = 1, las = 2)
    
    # Adding y-axis title, this is used to avoid overlapping of axis title and labels
    mtext(paste0("Rank According to ", x$metric), side = 2, line = 4)
    
    # Resetting mar
    par(mar = oldmar)
    
  }else{
    stop("supplied ptype is not currently supported")
  }
}
