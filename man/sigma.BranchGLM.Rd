% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/BranchGLM.R
\name{sigma.BranchGLM}
\alias{sigma.BranchGLM}
\title{Extract Square Root of the Dispersion Parameter Estimates}
\usage{
\method{sigma}{BranchGLM}(object, ...)
}
\arguments{
\item{object}{a \code{BranchGLM} object.}

\item{...}{further arguments passed to or from other methods.}
}
\value{
A numeric vector of length 2 with first and second elements giving
\item{\code{mle}}{ the MLE of the dispersion parameter}
\item{\code{pearson}}{ the Pearson estimator of the dispersion parameter}
}
\description{
Extracts the square root of the dispersion parameter estimates from \code{BranchGLM} objects.
}
\note{
The dispersion parameter for binomial and Poisson regression is always fixed to be 1.
The MLE of the dispersion parameter is used in the calculation of the log-likelihood
while the Pearson estimator of the dispersion parameter is used to calculate
standard errors for the coefficient estimates.
}
\examples{
Data <- iris
Fit <- BranchGLM(Sepal.Length ~ ., data = Data, family = "gaussian", link = "identity")
sigma(Fit)

}
