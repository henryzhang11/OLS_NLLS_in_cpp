// main.cpp

#include "linear_regression.h"
#include <eigen3/Eigen/Dense>
#include <vector>
#include <set>
#include <iostream>

int main() {
    try {
        // Sample data
        Eigen::MatrixXd X_input(5, 2);
        X_input << 1, 2,
                   2, 3,
                   3, 4,
                   4, 5,
                   5, 6;
        Eigen::VectorXd Y_input(5);
        Y_input << 2, 3, 5, 7, 11;

        // Initialize LinearRegression
        LinearRegression lr(X_input, Y_input);

        // Fit the model
        Eigen::VectorXd beta = lr.fit(X_input, Y_input);
        std::cout << "Coefficients:\n" << beta << std::endl;

        // Compute standard tests
        auto [R2, RSE] = lr.standard_tests();
        std::cout << "R^2: " << R2 << ", RSE: " << RSE << std::endl;

        // Define feature sets for F-statistic
        std::set<int> all_features = {0, 1}; // Assuming 0 and 1 are feature indices
        std::set<int> relevant_features = {0};

        // Compute F-statistic
        double p_value = lr.f_statistic(all_features, relevant_features);
        std::cout << "F-statistic p-value: " << p_value << std::endl;

        // Predict with new samples
        std::vector<std::vector<double>> new_samples = {
            {6, 7},
            {7, 8}
        };
        double alpha = 0.05;
        auto predictions = lr.predict(new_samples, alpha);
        for (size_t i = 0; i < predictions.size(); ++i) {
            std::cout << "Sample " << i+1 << ": [" 
                      << std::get<0>(predictions[i]) << ", " 
                      << std::get<1>(predictions[i]) << "]\n";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
    }
    
    return 0;
}

