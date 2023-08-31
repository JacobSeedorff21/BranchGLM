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
    
  # Creating data frame with results
  df <- data.frame(BestModels, object$bestmetrics)
  colnames(df) <- c(object$names, object$metric)
  
  # Creating formulas for each model
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
      if(!("(Intercept)" %in% object$names[x != 0])){
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

#' Fits GLMs for summary.BranchGLMVS and BranchGLMVS objects
#' @name fit
#' @param object a \code{summary.BranchGLMVS} or \code{BranchGLMVS} object.
#' @param which a positive integer indicating which model to fit, 
#' the default is to fit the first model .
#' @param keepData Whether or not to store a copy of data and design matrix, the default 
#' is TRUE. If this is FALSE, then the results from this cannot be used inside of \code{VariableSelection}.
#' @param keepY Whether or not to store a copy of y, the default is TRUE. If 
#' this is FALSE, then the binomial GLM helper functions may not work and this 
#' cannot be used inside of \code{VariableSelection}.
#' @param useNA Whether or not to use observations that had missing values in the 
#' full model, but not for this specific model. The default is FALSE.
#' @param ... further arguments passed to other methods.
#' @details The information needed to fit the GLM is taken from the original information 
#' supplied to the \code{VariableSelection} function.
#' 
#' The fitted models do not have standard errors or p-values since these are 
#' biased due to the selection process.
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
#' @return An object of class \link{BranchGLM}. 
#' @export
#' 
fit <- function(object, ...) {
  UseMethod("fit")
}

#' @rdname fit 
#' @export
fit.summary.BranchGLMVS <- function(object, which = 1, keepData = TRUE, keepY = TRUE, 
                                    useNA = FALSE, ...){
  if(!is.numeric(which) || which < 0 || which > length(object$formulas) || 
     which != as.integer(which)){
    stop("which must be a positive integer denoting the rank of the model to fit")
  }
  FinalModel <- BranchGLM(object$formulas[[which]], data = object$initmodel$mf, 
                          family = object$initmodel$family, link = object$initmodel$link, 
                          offset = object$initmodel$offset,
                          method = object$initmodel$method, 
                          tol = object$initmodel$tol, maxit = object$initmodel$maxit, 
                          keepData = keepData, keepY = keepY)
  
  # Removing standard errors and p-values along with vcov
  FinalModel$coefficients[, 2:4] <- NA
  FinalModel$vcov <- NA
  FinalModel$numobs <- object$initmodel$numobs
  FinalModel$missing <- object$initmodel$missing
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

#' Plot Method for summary.BranchGLMVS and BranchGLMVS objects
#' @param x a \code{summary.BranchGLMVS} or \code{BranchGLMVS} object.
#' @param ptype the type of plot to produce, look at details for more explanation.
#' @param marnames value used to determine how large to make margin of axis with variable 
#' names, this is only for the "variables" plot. If variable names are cut-off, 
#' consider increasing this from the default value of 7.
#' @param addLines logical value to indicate whether or not to add black lines to 
#' separate the models for the "variables" plot. This is typically useful for smaller 
#' amounts of models, but can be annoying if there are many models.
#' @param type what type of plot to draw for the "metrics" plot, see more details at \link{plot.default}. 
#' @param horiz whether models should be displayed horizontally or vertically in the "variables" plot.
#' @param cex.names how big to make variable names in the "variables" plot.
#' @param cex.lab how big to make axis labels.
#' @param cex.axis how big to make axis annotation.
#' @param cex.legend how big to make legend labels.
#' @param ... arguments passed to the generic plot and image methods.
#' @details The different values for ptype are as follows
#' \itemize{
#'  \item "metrics" for a plot that displays the metric values ordered by rank
#'  \item "variables" for a plot that displays which variables are in each of the top models
#'  \item "both" for both plots
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
#' @return This only produces plots, nothing is returned.
#' @export

plot.summary.BranchGLMVS <- function(x, ptype = "both", marnames = 7, addLines = TRUE, 
                                     type = "b", horiz = FALSE,
                                     cex.names = 1, cex.lab = 1, 
                                     cex.axis = 1, cex.legend = 1,
                                     ...){
  if(!ptype %in% c("metrics", "both", "variables")){
    stop("supplied ptype is not supported")
  }
  if(ptype %in% c("metrics", "both")){
    plot(1:nrow(x$results), x$results[, ncol(x$results)], 
         xlab = "Rank", ylab = x$metric, 
         main = paste0("Best Models Ranked by ", x$metric),
         type = type, cex.lab = cex.lab, cex.axis = cex.axis,
         ...)
  }
  if(ptype %in% c("variables", "both") && !horiz){
    # This is inspired by the plot.regsubsets function
    n <- length(x$formulas)
    Names <- colnames(x$results)[-(ncol(x$results))]
    z <- x$results[, -(ncol(x$results))]
    z[z == "kept"] <- 2
    z[z == "no"] <- 1
    z[z == "yes"] <- 0
    z <- apply(z, 2, as.numeric)
    if(!is.matrix(z)){
      z <- matrix(z, ncol = length(z))
    }
    y <- 1:ncol(z)
    x1 <- 1:nrow(z)
    
    # Creating image  
    oldmar <- par("mar")
    on.exit(par(mar = oldmar))
    par(mar = c(5, marnames, 3, 6) + 0.1)
    
    if(all(z != 2)){
      # Do this if there were no variable kept
      image(x1, y, z, ylab = "", 
            xaxt = "n", yaxt = "n", xlab = "", 
            main = paste0("Best Models Ranked by ", x$metric), 
            col = c("deepskyblue", "indianred"), ...)
      legend(grconvertX(1, from = "npc"), grconvertY(1, from = "npc"), 
             legend = c("Included", "Excluded"), 
             fill = c("deepskyblue", "indianred"), 
             xpd = TRUE, cex = cex.legend)
    }else{
      # Do this if there were any kept variables
      image(x1, y, z, ylab = "", 
            xaxt = "n", yaxt = "n", xlab = "", 
            main = paste0("Best Models Ranked by ", x$metric), 
            col = c("deepskyblue", "indianred", "forestgreen"), ...)
      legend(grconvertX(1, from = "npc"), grconvertY(1, from = "npc"), 
             legend = c("Included", "Excluded", "Kept"), 
             fill = c("deepskyblue", "indianred", "forestgreen"), 
             xpd = TRUE, cex = cex.legend)
    }
    
    # Adding lines
    if(addLines){
      abline(h = y + 0.5, v = x1 + 0.5)
    }else{
      abline(h = y + 0.5)
    }
    
    # Adding axis labels
    axis(1, at = x1, labels = x1, line = 1, las = 1, cex.axis = cex.axis)
    axis(2, at = y, labels = Names, line = 1, las = 2, cex.axis = cex.names)
    
    # Adding y-axis title, this is used to avoid overlapping of axis title and labels
    mtext(paste0("Rank According to ", x$metric), side = 1, line = 4, cex = cex.lab)
    
  }else if(ptype %in% c("variables", "both") && horiz){
    # This is inspired by the plot.regsubsets function
    n <- length(x$formulas)
    Names <- colnames(x$results)[-(ncol(x$results))]
    z <- x$results[, -(ncol(x$results))]
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
    on.exit(par(mar = oldmar))
    par(mar = c(marnames, 5, 3, 6) + 0.1)
    
    if(all(z != 2)){
      # Do this if there were no variable kept
      image(x1, y, z, ylab = "", 
            xaxt = "n", yaxt = "n", xlab = "", 
            main = paste0("Best Models Ranked by ", x$metric), 
            col = c("deepskyblue", "indianred"), ...)
      legend(grconvertX(1, from = "npc"), grconvertY(1, from = "npc"), 
             legend = c("Included", "Excluded"), 
             fill = c("deepskyblue", "indianred"), 
             xpd = TRUE, cex = cex.legend)
    }else{
      # Do this if there were any kept variables
      image(x1, y, z, ylab = "", 
            xaxt = "n", yaxt = "n", xlab = "", 
            main = paste0("Best Models Ranked by ", x$metric), 
            col = c("deepskyblue", "indianred", "forestgreen"), ...)
      legend(grconvertX(1, from = "npc"), grconvertY(1, from = "npc"), 
             legend = c("Included", "Excluded", "Kept"), 
             fill = c("deepskyblue", "indianred", "forestgreen"),
             xpd = TRUE, cex = cex.legend)
    }
    
    # Adding lines
    if(addLines){
      abline(v = x1 + 0.5, h = y + 0.5)
    }else{
      abline(v = x1 + 0.5)
    }
    
    # Adding axis labels
    axis(1, at = x1, labels = Names, line = 1, las = 2, cex.axis = cex.names)
    axis(2, at = y, labels = y, line = 1, las = 2, cex.axis = cex.axis)
    
    # Adding y-axis title, this is used to avoid overlapping of axis title and labels
    mtext(paste0("Rank According to ", x$metric), side = 2, line = 4, cex = cex.lab)
    
  }
}
