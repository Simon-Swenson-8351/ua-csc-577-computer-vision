function [] = hw3()
    line_data = importdata('line_data.txt');
    size(line_data)
    line_data_mean = mean(line_data);
    building_img = imread('building.jpeg');
    size(building_img)
    chandelier_img = imread('chandelier.tiff');
    size(chandelier_img)
    non_homo_slope_int = non_homogeneous_linear_least_squares(line_data(:, 1), line_data(:, 2))
    non_homo_a_b = non_homogeneous_to_homogeneous_model(non_homo_slope_int, line_data_mean(1), line_data_mean(2))
    homo_a_b = homogeneous_linear_least_squares(line_data)
    homo_slope_int = homogeneous_to_non_homogeneous_model(homo_a_b, line_data_mean(1), line_data_mean(2))
    non_homo_rmse_non_homo = rmse_non_homogeneous(line_data(:, 1), line_data(:, 2), non_homo_slope_int)
    non_homo_rmse_homo = rmse_homogeneous(line_data(:, 1), line_data(:, 2), non_homo_a_b)
    homo_rmse_non_homo = rmse_non_homogeneous(line_data(:, 1), line_data(:, 2), homo_slope_int)
    homo_rmse_homo = rmse_homogeneous(line_data(:, 1), line_data(:, 2), homo_a_b)
    figure();
    scatter(line_data(:, 1), line_data(:, 2), 3, 'filled');
    hold on;
    plot(-0.5:0.1:3.5, line(-0.5:0.1:3.5, non_homo_slope_int));
    plot(-0.5:0.1:3.5, line(-0.5:0.1:3.5, homo_slope_int));
    legend('Original points', 'Non-homogeneous fit', 'Homogeneous fit');
end

% This is very similar to what we did last homework, by using the pseudo-
% inverse.
function [model_params] = non_homogeneous_linear_least_squares(X, y)
    X_size = size(X);
    % ones needed to account for the y-intercept.
    U = [X ones(X_size(1), 1)];
    model_params = pseudoinv(U) * y;
end

% Note that, as opposed to non homo, homo treats y as just another x
% dimension, so we don't have a y parameter here.
function [model_params] = homogeneous_linear_least_squares(X)
    % Different U this time
    X_mean = mean(X);
    U = X - X_mean;
    [U_eigvecs, U_eigvals] = eig(U' * U);
    % Assume that the first column of the eigenvector matrix corresponds to
    % the lowest eigenvalue.
    % TODO: bad assumption, but it's probably fine for this assignment.
    model_params = U_eigvecs(:, 1);
end

% Assumes each row is a data point
function [out] = rmse_non_homogeneous(X, y, slope_int)
    X_size = size(X);
    % Sum each row together.
    out = sqrt(sum((y - [X ones(X_size(1), 1)] * slope_int) .^ 2) ./ X_size(1));
end

function [out] = rmse_homogeneous(X, y, a_b)
    d = homogeneous_get_d(a_b, mean(X), mean(y));
    X_size = size(X);
    out = sqrt(sum((d - a_b(1) .* X - a_b(2) .* y) .^ 2) / X_size(1));
end

function [d] = homogeneous_get_d(a_b, X_mean, y_mean)
    d = a_b(1) * X_mean + a_b(2) * y_mean;
end

function [slope_int] = homogeneous_to_non_homogeneous_model(a_b, X_mean, y_mean)
    % Need to do some math to get U * x = 0 into the form X * m = y. I
    % derived this by hand.
    slope_int = [-a_b(1)/a_b(2) ; X_mean * a_b(1) / a_b(2) + y_mean];
end

function [a_b] = non_homogeneous_to_homogeneous_model(slope_int, X_mean, y_mean)
    b = sqrt(1/(slope_int(1)^2 + 1));
    a_b = [-b * slope_int(1) ; b];
end

function [out] = pseudoinv(in)
    out = (in' * in)^(-1) * in';
end

function [y] = line(X, theta)
    X_size = size(X);
    U = [X' ones(X_size(2), 1)];
    y = U * theta;
end