// linear_regression.cpp

#include "linear_regression.h"

#include <Eigen/Dense>
#include <vector>
#include <set>
#include <tuple>
#include <stdexcept>
#include <cmath>
#include <numeric>

// Include Boost libraries for statistical distributions
#include <boost/math/distributions/fisher_f.hpp>
#include <boost/math/distributions/students_t.hpp>

LinearRegression::LinearRegression(const Eigen::MatrixXd& X_input, const Eigen::VectorXd& Y_input)
    : X(X_input), Y(Y_input), beta(Eigen::VectorXd()) {}

// Fit a linear model.
// - Adds a column of 1s to X for the intercept.
// - Checks if the number of rows is >= number of columns.
// - Checks if the columns of X are linearly independent.
// - Computes beta = (X^T X)^{-1} X^T Y.
Eigen::VectorXd LinearRegression::fit(const Eigen::MatrixXd& x, const Eigen::VectorXd& y) {
    // Add a column of ones for the intercept
    Eigen::MatrixXd X_augmented(x.rows(), x.cols() + 1);
    X_augmented << Eigen::VectorXd::Ones(x.rows()), x;

    // Check if the number of rows is >= number of columns
    if (X_augmented.rows() < X_augmented.cols()) {
        throw std::invalid_argument("Number of rows of X must be >= number of columns of X.");
    }

    // Check if columns of X are linearly independent by checking the rank
    Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(X_augmented);
    if (lu_decomp.rank() < X_augmented.cols()) {
        throw std::invalid_argument("Columns of X are not linearly independent.");
    }

    // Compute beta = (X^T X)^{-1} X^T Y
    Eigen::MatrixXd XtX = X_augmented.transpose() * y;
    Eigen::MatrixXd XtX_inv = (X_augmented.transpose() * X_augmented).inverse();
    Eigen::VectorXd beta_estimate = XtX_inv * X_augmented.transpose() * y;

    // Store the augmented X and beta
    X = X_augmented;
    Y = y;
    beta = beta_estimate;

    return beta;
}

// Calculate F-statistic and return the corresponding probability using the CDF of an F-distribution.
double LinearRegression::f_statistic(const std::set<int>& all_features, const std::set<int>& relevant_features) const {
    // Number of observations and total features
    int n = X.rows();
    int p_all = all_features.size();
    int p_relevant = relevant_features.size();

    if (p_relevant >= p_all) {
        throw std::invalid_argument("Number of relevant features must be less than the number of all features.");
    }

    // Helper lambda to extract columns based on feature indices
    auto extract_columns = [&](const std::set<int>& features) -> Eigen::MatrixXd {
        int cols = features.size() + 1; // +1 for intercept
        Eigen::MatrixXd X_subset(n, cols);
        X_subset.col(0) = Eigen::VectorXd::Ones(n); // Intercept
        int col_idx = 1;
        for (const auto& feature : features) {
            if (feature < 0 || feature >= (X.cols() - 1)) { // -1 because first column is intercept
                throw std::out_of_range("Feature index out of range.");
            }
            X_subset.col(col_idx++) = X.col(feature + 1); // +1 to skip intercept
        }
        return X_subset;
    };

    // Extract relevant and all features
    Eigen::MatrixXd X0 = extract_columns(relevant_features);
    Eigen::MatrixXd X1 = extract_columns(all_features);

    // Compute beta0 and beta1
    Eigen::VectorXd beta0 = (X0.transpose() * X0).inverse() * X0.transpose() * Y;
    Eigen::VectorXd beta1 = (X1.transpose() * X1).inverse() * X1.transpose() * Y;

    // Compute RSS0 and RSS1
    Eigen::VectorXd residual0 = Y - X0 * beta0;
    Eigen::VectorXd residual1 = Y - X1 * beta1;
    double RSS0 = residual0.squaredNorm();
    double RSS1 = residual1.squaredNorm();

    // Compute F-statistic
    int q = p_all - p_relevant;
    return ((RSS0 - RSS1) / q) / (RSS1 / (n - p_all - 1));
}

// Calculate R^2 and RSE.
std::tuple<double, double> LinearRegression::standard_tests() const {
    int n = X.rows();
    int p = X.cols() - 1; // Exclude intercept

    // Compute RSS
    Eigen::VectorXd residual = Y - X * beta;
    double RSS = residual.squaredNorm();

    // Compute RSE
    double RSE = std::sqrt(RSS / (n - p - 1));

    // Compute TSS
    double Y_sum = Y.sum();
    double TSS = Y.squaredNorm() - (Y_sum * Y_sum) / n;

    // Compute R^2
    double R_squared = 1.0 - (RSS / TSS);

    return std::make_tuple(R_squared, RSE);
}

// Predict point estimate and prediction interval.
std::vector<std::tuple<double, double>> LinearRegression::predict(const std::vector<std::vector<double>>& samples, double alpha) const {
    int n = X.rows();
    int p = X.cols() - 1; // Exclude intercept

    // Get RSE from standard tests
    double sigma = std::get<1>(standard_tests());

    // Compute (X^T X)^{-1}
    Eigen::MatrixXd XtX_inv = (X.transpose() * X).inverse();

    // Degrees of freedom for t-distribution
    double df = static_cast<double>(n - p - 1);

    // Compute t critical value
    boost::math::students_t dist_t(df);
    double t_crit = boost::math::quantile(boost::math::complement(dist_t, alpha / 2));

    std::vector<std::tuple<double, double>> intervals;
    intervals.reserve(samples.size());

    for (const auto& sample : samples) {
        if (sample.size() != static_cast<size_t>(p)) {
            throw std::invalid_argument("Sample size does not match number of features.");
        }

        // Create augmented sample with intercept
        Eigen::VectorXd x_sample(p + 1);
        x_sample(0) = 1.0; // Intercept
        for (int i = 0; i < p; ++i) {
            x_sample(i + 1) = sample[i];
        }

        // Compute point estimate
        double y_pred = x_sample.dot(beta);

        // Compute standard error
        double se = sigma * std::sqrt(x_sample.transpose() * XtX_inv * x_sample);

        // Compute prediction interval
        double lower = y_pred - t_crit * se;
        double upper = y_pred + t_crit * se;

        intervals.emplace_back(std::make_tuple(lower, upper));
    }

    return intervals;
}
