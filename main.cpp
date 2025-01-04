#include "linear_regression.h"
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>

void loadCSV(const std::string& filename, Eigen::MatrixXd& X, Eigen::VectorXd& Y) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> data;
    std::string line, cell;

    // Skip the first line (header)
    std::getline(file, line); // This reads the header line and ignores it.

    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::vector<double> row;

        while (std::getline(lineStream, cell, ',')) {
            row.push_back(std::stod(cell));
        }
        data.push_back(row);
    }

    int rows = data.size();
    int cols = data[0].size() - 1; // Last column is the response vector

    X = Eigen::MatrixXd(rows, cols);
    Y = Eigen::VectorXd(rows);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            X(i, j) = data[i][j];
        }
        Y(i) = data[i][cols]; // Last column is the response
    }
}

int main() {
    // Load data from CSV
    Eigen::MatrixXd X;
    Eigen::VectorXd Y;
    loadCSV("train_data.csv", X, Y);

    // Create LinearRegression object
    LinearRegression lr(X, Y);

    // Fit the model
    Eigen::VectorXd beta = lr.fit(X, Y);
    std::cout << "Fitted Coefficients (Beta):\n" << beta << std::endl;

    // Perform standard tests (R^2 and RSE)
    auto [r_squared, rse] = lr.standard_tests();
    std::cout << "R^2: " << r_squared << ", RSE: " << rse << std::endl;

	// Perform f tests for all variables
	std::set<int> all_variables;
	for (int i = 0; i < X.cols(); ++i) {
		all_variables.insert(i);
	}
	for (int i = 0; i < X.cols(); ++i) {
		std::set<int> selected_variables;
		for (int j = 0; j < X.cols(); ++j) {
			if (j != i) {
				selected_variables.insert(j);
			}
		}
		std::cout << "f-statitic of varibale " << i << " equals " << lr.f_statistic(all_variables, selected_variables) << std::endl;
	}
    // Predict with new samples
	Eigen::MatrixXd X_test;
	Eigen::VectorXd y_test;
	loadCSV("test_data.csv", X_test, y_test);	    
	std::vector<std::vector<double>> samples;
	std::vector<double> sample;
	for (int i = 0; i < X_test.cols(); ++i) {
		sample.push_back(static_cast<double>(X_test(0, i)));
	}
	samples.push_back(sample); 
	double alpha = 0.05; // 95% confidence level
	auto predictions = lr.predict(samples, alpha);
    for (const auto& [lower, upper] : predictions) {
        std::cout << "Prediction Interval: [" << lower << ", " << upper << "]\n";
    }

    return 0;
}
