function [] = hw4()

    % Uncomment this if you care about verifying coordinate transform for
    % part A
    %tent_image = imread('tent.jpg');
    %tent_image(100, 200, :) = [255 0 0];
    %figure();
    %imshow(tent_image);
    
    calibration_image = imread('IMG_0862.jpeg');
    figure();
    imshow(calibration_image);
    
    world_coords = importdata('world_coords.txt');
    image_coords = importdata('image_coords.txt');
    
    camera_matrix_1 = importdata('camera_matrix_1.txt');
    camera_matrix_2 = importdata('camera_matrix_2.txt');
    
    % Convert to homogeneous
    world_coords(:, 4) = 1.0;
    
    % Run the tests
    camera_matrix_1_output = camera_matrix_1 * world_coords';
    camera_matrix_1_output = camera_matrix_1_output ./ camera_matrix_1_output(3, :);
    camera_matrix_1_output = camera_matrix_1_output(1:2, :);
    camera_matrix_2_output = camera_matrix_2 * world_coords';
    camera_matrix_2_output = camera_matrix_2_output ./ camera_matrix_2_output(3, :);
    camera_matrix_2_output = camera_matrix_2_output(1:2, :);
    
    % Plot the points
    calibration_image = draw_points(calibration_image,          image_coords', [255   0   0], 4);
    calibration_image = draw_points(calibration_image, camera_matrix_1_output, [  0 255   0], 4);
    calibration_image = draw_points(calibration_image, camera_matrix_2_output, [  0   0 255], 4);
    figure();
    imshow(calibration_image);
    
    m1_rmse = rmse(image_coords, camera_matrix_1_output')
    m2_rmse = rmse(image_coords, camera_matrix_2_output')
end


function [] = test_image_pixel_coordinate_transformation()
    extents = [800.0 600.0];
    % Expected results:
    %   -0.5000   -0.5000
    %   399.5000   -0.5000
    %   799.5000   -0.5000
    %   -0.5000  299.5000
    %   399.5000  299.5000
    %   799.5000  299.5000
    %   -0.5000  599.5000
    %   399.5000  599.5000
    %   799.5000  599.5000
    image_coords = [-1.0  1.0;
                     0.0  1.0;
                     1.0  1.0;
                    -1.0  0.0;
                     0.0  0.0;
                     1.0  0.0;
                    -1.0 -1.0;
                     0.0 -1.0;
                     1.0 -1.0];
    pixel_coords = [  0.0   0.0;
                    399.0   0.0;
                    799.0   0.0;
                      0.0 299.0;
                    399.0 299.0;
                    799.0 299.0;
                      0.0 599.0;
                    399.0 599.0;
                    799.0 599.0];
    image_coordinates_to_pixel_coordinates(image_coords, extents)
    pixel_coordinates_to_image_coordinates(pixel_coords, extents)
end

% Note that pixel coordinates are considered to be centered, so an x value
% of 0 will not map exactly to -1, but some value very close to it, like
% -0.995, for example.
% pixel_coordinates a n x 2-element vector (x, y)
% pixel_extents a 2-element vector (x, y)
% image_coordinates a n x 2-element vector (x, y)
function [image_coordinates] = pixel_coordinates_to_image_coordinates(pixel_coordinates, pixel_extents)
    image_coordinates = pixel_coordinates + 0.5;
    image_coordinates = image_coordinates .* 2.0 ./ pixel_extents - 1.0;
    image_coordinates(:, 2) = image_coordinates(:, 2) * -1;
end

% Note that pixel coordinates are considered to be centered, so an x value
% of -1, for example, will map to the left edge of the left most pixel, or
% -0.5
% image_coordinates a n x 2-element vector
% pixel_extents a 2-element vector
% pixel_coordinates a n x 2-element vector
function [pixel_coordinates] = image_coordinates_to_pixel_coordinates(image_coordinates, pixel_extents)
    pixel_coordinates = (image_coordinates + 1.0) ./ 2 .* pixel_extents;
    pixel_coordinates(:, 2) = (pixel_coordinates(:, 2) .* -1) + pixel_extents(:, 2);
    pixel_coordinates = pixel_coordinates - 0.5;
end

% The point, as well as the r pixels around the point in a square, will be
% colored. The end result will be a colored square of w, h = 2 * r + 1
% centered at position.
function [image] = draw_points(image, points, color, r)
    points_size = size(points);
    points_rounded = round(points);
    for i = 1:points_size(2)
        image = draw_point(image, points_rounded(:, i), color, r);
    end
end

function [image] = draw_point(image, point, color, r)
    image(max((point(1) - r):(point(1) + r), 1), max((point(2) - r):(point(2) + r), 1), 1) = color(1);
    image(max((point(1) - r):(point(1) + r), 1), max((point(2) - r):(point(2) + r), 1), 2) = color(2);
    image(max((point(1) - r):(point(1) + r), 1), max((point(2) - r):(point(2) + r), 1), 3) = color(3);
end

% Uses a distance metric
% Assumes each row is a data point
function [out] = rmse(actual, guesses)
    actual_size = size(actual);
    actual_rows = actual_size(1);
    % Sum each row together.
    out = sqrt(sum(sum((actual - guesses) .^ 2)) ./ actual_rows);
end