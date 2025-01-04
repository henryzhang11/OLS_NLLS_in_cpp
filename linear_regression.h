#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <Eigen/Dense>
#include <vector>
#include <set>
#include <tuple>

class LinearRegression {
public:
    // Constructor: Initialize X and Y
    LinearRegression(const Eigen::MatrixXd& X_input, const Eigen::VectorXd& Y_input);

    /* 
    Fit a linear model.
		- add a column of 1s to X
        - Checks: number of rows of X >= number of columns of X.
        - Checks: columns of X are linearly independent.
        - Returns: beta = (X^T X)^{-1} X^T Y.
    */
    Eigen::VectorXd fit(const Eigen::MatrixXd& x, const Eigen::VectorXd& y);

    /* 
    Calculate F-statistic (all fits assume an intercept).
        - Fits \hat{\beta}_0 using relevant_features.
        - Fits \hat{\beta}_1 using all_features.
        - Computes:
            RSS_0 = (Y - X_0 \hat{\beta}_0)^T (Y - X_0 \hat{\beta}_0) with X_0 = relevant_features in X.
            RSS_1 = (Y - X_1 \hat{\beta}_1)^T (Y - X_1 \hat{\beta}_1) with X_1 = all_features in X.
        - Computes F-statistic:
            q = size of all_features - size of relevant_features,
            n = number of rows of X,
            p = number of features in all_features.
            F = ((RSS_0 - RSS_1) / q) / (RSS_1 / (n - p - 1)).
        - Returns the corresponding probability using the CDF of an F-distribution.
    */
    double f_statistic(const std::set<int>& all_features, const std::set<int>& relevant_features) const;

    /*
    Calculate R^2 and RSE.
        - n = number of rows of X,
        - RSS = (Y - X \beta)^T (Y - X \beta),
        - RSE = sqrt(RSS / (n - p - 1)),
        - TSS = Y^T Y - (1 / n) * (sum(Y_i)^2 for i in 1 to n),
        - R^2 = 1 - (RSS / TSS).
        - Returns: R^2 and RSE.
    */
    std::tuple<double, double> standard_tests() const;

    /*
    Predict point estimate and prediction interval.
        - t = t_{alpha/2, n - p},
        - sigma = std::get<1>(standard_tests()),
        - For each sample in samples:
            lower_bound = sample^T * beta - t * sigma * sqrt(sample^T (X^T X)^{-1} sample),
            upper_bound = sample^T * beta + t * sigma * sqrt(sample^T (X^T X)^{-1} sample).
        - Returns: vector of (lower_bound, upper_bound) tuples for all samples.
    */
    std::vector<std::tuple<double, double>> predict(const std::vector<std::vector<double>>& samples, double alpha) const;

private:
    Eigen::MatrixXd X;              // Design matrix
    Eigen::VectorXd Y;              // Response vector
    Eigen::VectorXd beta;           // Coefficients
};

#endif
