function [] = hw5()

    % Part A: Homogeneous Least Squares between image coordinates and world coordinates
    world_coords = importdata('world_coords.txt');
    world_coords_size = size(world_coords);
    world_coords_homogeneous = normal_coords_to_homogeneous_coords(world_coords);
    matlab_pixel_index_coords = importdata('image_coords.txt');
    matlab_pixel_index_coords_homogeneous = normal_coords_to_homogeneous_coords(matlab_pixel_index_coords);
    calibration_image = imread('IMG_0862.jpeg');
    calibration_image_size = size(calibration_image);

    % First, convert the image coordinates to the standard coordinate system with 
    % (0, 0) in the center and the smaller of the two dimensions ranging from -1 to 
    % 1.
    % Note that the image is 1600 x 1200.

    % Scale, rotate -90 degrees, translate.
    %standard_image_coords_to_matlab_pixel_index_coords = get_standard_image_coords_to_matlab_pixel_index_coords_matrix([calibration_image_size(2) calibration_image_size(1)]);
    %matlab_pixel_index_coords_to_standard_image_coords = standard_image_coords_to_matlab_pixel_index_coords^(-1);

    % Just to verify, should be (0, 1400, 1)
    %standard_image_coords_to_matlab_pixel_index_coords * [1; 1; 1]

    %standard_image_coords = matlab_pixel_index_coords_homogeneous * matlab_pixel_index_coords_to_standard_image_coords';

    U = compute_homogeneous_U_matrix(world_coords_homogeneous, matlab_pixel_index_coords);
    camera_vector = find_homogeneous_least_squared_error_solution(U);
    camera_matrix = camera_vector_to_matrix(camera_vector)

    predicted_points_homogeneous = world_coords_homogeneous * camera_matrix'
    %predicted_points_homogeneous = predicted_points_homogeneous * standard_image_coords_to_matlab_pixel_index_coords';
    predicted_points = homogeneous_coords_to_normal_coords(predicted_points_homogeneous);


    calibration_image_camera_dots = calibration_image;
    
    matlab_pixel_index_coords
    calibration_image_camera_dots = draw_points(calibration_image_camera_dots, matlab_pixel_index_coords, [  0 255 0], 4);
    predicted_points
    calibration_image_camera_dots = draw_points(calibration_image_camera_dots,          predicted_points, [255   0 0], 4);
    figure();
    imshow(calibration_image_camera_dots);
    
    our_rmse = rmse(matlab_pixel_index_coords, predicted_points)

    % Part B: Rendering a sphere into the mini-fig image
    mini_fig_image = imread('IMG_0861.jpeg');
    mini_fig_image_s1 = mini_fig_image;
    camera_loc = [9 14 11];
    light_loc = [33 29 44];
    sphere_loc = [3 2 3];
    sphere_r = 0.5;
    mini_fig_image_s1 = render_sphere(mini_fig_image_s1, camera_matrix, camera_loc, light_loc, sphere_loc, sphere_r);
    figure();
    imshow(mini_fig_image_s1);
    % Additional exploration, since the sphere seems off-centered to me.
    vertical_points = zeros(7, 3);
    for i = 0:1:6
        vertical_points(i + 1, :) = [3 2 i];
    end
    vertical_points_px = homogeneous_coords_to_normal_coords(normal_coords_to_homogeneous_coords(vertical_points) * camera_matrix');
    mini_fig_image_dbl_check = mini_fig_image_s1;
    mini_fig_image_dbl_check = draw_points(mini_fig_image_dbl_check, vertical_points_px, [0 0 255], 4);
    figure();
    imshow(mini_fig_image_dbl_check);
    
    % This got me interested in rendering a grid, as well.
    % (Okay this is visually indecipherable. Too many points. Don't put
    % this in the report.)
    %all_points = zeros(6.^3, 3);
    %for i = 0:6
    %    for j = 0:6
    %        for k = 0:6
    %            all_points(i * 36 + j * 6 + k + 1, :) = [i j k];
    %        end
    %    end
    %end
    %all_points_px = homogeneous_coords_to_normal_coords(normal_coords_to_homogeneous_coords(all_points) * camera_matrix');
    %mini_fig_image_triple_check = mini_fig_image_s1;
    %mini_fig_image_triple_check = draw_points(mini_fig_image_triple_check, all_points_px, [0 0 255], 4);
    %figure();
    %imshow(mini_fig_image_triple_check);
    
    % Separating intrinsics and extrinsics
    % The first matrix basically sends standard image coordinates to pixel
    % coordinates, so we derive that matrix similar to how we did so in
    % HW4.
    
    % Synthetic matrices
    intrinsic_t = [1   0 1200 / 2.0; ...
                   0   1 1600 / 2.0; ...
                   0   0          1];
    % Why do we need to flip all axes? Well, our points actually end up
    % with negative w values if they're viewable (the camera coordinate
    % system points away from the scene, the other direction). To make
    % things easier to reason about, we just take care of that, here.
    intrinsic_r = [ 0 -1 0; ...
                    1  0 0; ...
                    0  0 1];
    intrinsic_s = [1200 / 2.0          0 0; ...
                   0          1200 / 2.0 0; ...
                   0                   0 1];
    intrinsic_f = [1 0 0; ...
                   0 1 0; ...
                   0 0 -1];
    proj = [1 0 0 0; ...
            0 1 0 0; ...
            0 0 1 0];
    extrinsic_r = rand(3);
    extrinsic_r = extrinsic_r * extrinsic_r';
    [extrinsic_r, D] = eig(extrinsic_r);
    if det(extrinsic_r) < 0
        extrinsic_r(:, 1) = -extrinsic_r(:, 1);
    end
    extrinsic_r(4, 1:3) = 0;
    extrinsic_r(1:3, 4) = 0;
    extrinsic_r(4, 4) = 1;
    extrinsic_r
    extrinsic_t = [1 0 0  5; ...
                   0 1 0 -2; ...
                   0 0 1  3; ...
                   0 0 0  1];
    synthetic_camera_matrix = intrinsic_t * intrinsic_r * intrinsic_s * intrinsic_f * proj * extrinsic_r * extrinsic_t
    % Normally, to convert from standard image coordinate points to pixel
    % points, we would want to map 1, 1, 1 to 0, 1400. However,
    % We would want 1, 1, -1 to map to 0, 1400, since every point visible
    % by the camera (barring FoV) is negative in the k direction. This is a
    % consequence of the camera being a right-handed system, so the k axis
    % points in toward the camera, rather than out toward the scene. So, we
    % just need to make the focal length negative 1, essentially. Thus, we
    % have one last flip matrix that just flips the last element (w).
    % Should be 0, 1400
    intrinsic_t * intrinsic_r * intrinsic_s * intrinsic_f * [1; 1; -1]
    
    [i_t, i_rf, i_s, e_r, e_t] = decompose_camera(synthetic_camera_matrix)
    
    intrinsic_inv = intrinsic^(-1)
    % Remove the intrinsic from the matrix
    projection_extrinsic = intrinsic_inv * camera_matrix;
    % Because of how the projection matrix acts (it lops off the last row
    % of the extrinsics), we just add that back to invert it.
    extrinsic = [camera_matrix;
                 0 0 0 1]
    % Rotation is also pretty easy. It turns out a rotation matrix
    % multiplied by a translation matrix yeilds the original rotation for
    % the top-left 3x3 sub-matrix.
    extrinsic_rotation = extrinsic;
    extrinsic_rotation(1:3, 4) = 0
    extrinsic_translation = extrinsic_rotation^(-1) * extrinsic
    camera_matrix
    camera_matrix_dbl_check = intrinsic * [1 0 0 0; ...
                                           0 1 0 0; ...
                                           0 0 1 0] * extrinsic_rotation * extrinsic_translation;
                                       
    camera_matrix_dbl_check = camera_matrix_dbl_check ./ sqrt(sum(sum(camera_matrix_dbl_check .^ 2)))
end

% Input is assumed to be a matrix where each row is a point in R^(n - 1).
function [out] = homogeneous_coords_to_normal_coords(in)
    in_size = size(in);
    out = (in ./ in(:, in_size(2)));
    out = out(:, 1:in_size(2) - 1);
end

% Input is assumed to be a matrix where each row is a point in R^n.
function [out] = normal_coords_to_homogeneous_coords(in)
    in_size = size(in);
    out = zeros(in_size(1), in_size(2) + 1);
    out(:, 1:in_size(2)) = in;
    out(:, in_size(2) + 1) = 1;
end

% computes U where U * x = 0.
% world_coords is assumed to be in homogeneous form. standard_image_coords is 
% assumed to be in standard image coordinates or have w = 1.
function [out] = compute_homogeneous_U_matrix(world_coords, standard_image_coords)
    world_coords_size = size(world_coords);
    out = zeros(world_coords_size(1) * 2, 12);
    % Filling out the matrix based on slide deck 10, page 16.
    out(1:2:(world_coords_size(1) * 2 - 1), 1:4) = world_coords;
    out(2:2:(world_coords_size(1) * 2), 5:8) = world_coords;
    out(1:2:(world_coords_size(1) * 2 - 1), 9:12) = -standard_image_coords(:, 1) .* world_coords;
    out(2:2:(world_coords_size(1) * 2), 9:12) = -standard_image_coords(:, 2) .* world_coords;
end

function [out] = find_homogeneous_least_squared_error_solution(U)
    [eig_vecs, eig_vals] = eig(U' * U);
    % TODO: this assumes the first eigenvector has the lowest eigenvalue. It's 
    % probably not super portable.
    out = eig_vecs(:, 1);
end

function [out] = camera_vector_to_matrix(camera_vector)
    out = zeros(3, 4);
    out(1, :) = camera_vector(1:4);
    out(2, :) = camera_vector(5:8);
    out(3, :) = camera_vector(9:12);
end

% Assumes each row is a point
function [image] = draw_points(image, points, color, r)
    points_size = size(points);
    points_rounded = round(points);
    for i = 1:points_size(1)
        image = draw_point(image, points_rounded(i, :), color, r);
    end
end

function [image] = draw_point(image, point, color, r)
    image_size = size(image);
    y_indices = (point(1) - r):(point(1) + r);
    y_indices = y_indices(find(y_indices > 0 & y_indices <= image_size(1)));
    x_indices = (point(2) - r):(point(2) + r);
    x_indices = x_indices(find(x_indices > 0 & x_indices <= image_size(2)));
    % Matlab is stupid and doesn't broadcast assignments to arrays, so have
    % to do this in three steps.
    image(y_indices, x_indices, 1) = color(1);
    image(y_indices, x_indices, 2) = color(2);
    image(y_indices, x_indices, 3) = color(3);
end



% Uses a distance metric
% Assumes each row is a data point
function [out] = rmse(actual, guesses)
    actual_size = size(actual);
    actual_rows = actual_size(1);
    % Sum each row together.
    out = sqrt(sum(sum((actual - guesses) .^ 2)) ./ actual_rows);
end

function [out] = make_unit(vec)
    out = vec / sqrt(sum(vec.^2));
end

function [out] = render_sphere(image, camera, camera_loc, light_loc, sphere_loc, r)
    out = image;
    for phi = (-pi / 2):(pi / 512):(pi / 2)
        for theta = 0:(pi / 512):(2 * pi)
            pt = [];
            pt(1) = sphere_loc(1) + cos(phi) * cos(theta) * r;
            pt(2) = sphere_loc(2) + cos(phi) * sin(theta) * r;
            pt(3) = sphere_loc(3) + sin(phi) * r;
            norm = make_unit(pt - sphere_loc);
            if (camera_loc - pt) * norm' > 0
                light_dir = make_unit(light_loc - pt) * 255;
                shade = max([0 round(light_dir * norm')]);
                pt_homo = normal_coords_to_homogeneous_coords(pt);
                image_coord = homogeneous_coords_to_normal_coords(pt_homo * camera');
                out(round(image_coord(1)), round(image_coord(2)), :) = shade;
            end
        end
    end
end

% dims is assumed to be a 2-d vector, first entry width, second entry
% height.
function [out] = get_standard_image_coords_to_matlab_pixel_index_coords_matrix(dims)
    less = min(dims);
    l_h = less/2.0;
    out = [1   0 dims(1) / 2.0;
           0   1 dims(2) / 2.0;
           0   0             1] * ...
         [ 0  -1             0;
           1   0             0;
           0   0             1] * ...
        [l_h   0             0;
           0 l_h             0;
           0   0             1];
end

function [i_t, i_rf, i_s, e_r, e_t] = decompose_camera(camera)
    function F = parameters_eqns(all_unks)
        rho = all_unks(1);
        w = all_unks(2);
        h = all_unks(3);
        x = all_unks(4);
        y = all_unks(5);
        z = all_unks(6);
        f1 = all_unks(7);
        f2 = all_unks(8);
        f3 = all_unks(9);
        g1 = all_unks(10);
        g2 = all_unks(11);
        g3 = all_unks(12);
        h1 = all_unks(13);
        h2 = all_unks(14);
        h3 = all_unks(15);
        F(1)  = -h * h1 / 2 - w * g1 - rho * camera(1, 1);
        F(2)  = -h * h2 / 2 - w * g2 - rho * camera(1, 2);
        F(3)  = -h * h3 / 2 - w * g3 - rho * camera(1, 3);
        F(4)  = x * (h * h1 / 2 + g1 * w) + y * (h * h2 / 2 + g2 * w) + z * (h * h3 / 2 + g3 * w) - rho * camera(1, 4);
        F(5)  = -w * h1 / 2 + w * f1 - rho * camera(2, 1);
        F(6)  = -w * h2 / 2 + w * f2 - rho * camera(2, 2);
        F(7)  = -w * h3 / 2 + w * f3 - rho * camera(2, 3);
        F(8)  = x * (w * h1 / 2 - f1 * w) + y * (w * h2 / 2 - f2 * w) + z * (w * h3 / 2 - f3 * w) - rho * camera(2, 4);
        F(9)  = -h1 - rho * camera(3, 1);
        F(10) = -h2 - rho * camera(3, 2);
        F(11) = -h3 - rho * camera(3, 3);
        F(12) = h1 * x + h2 * y + h3 * z - rho * camera(3, 4);

        % Dot products
        F(13) =    f1^2 +    f2^2 + f3^2    - 1;
        F(14) =    g1^2 +    g2^2 + g3^2    - 1;
        F(15) = f1 * g1 + f2 * g2 + f3 * g3;

        % Cross product of f x g = h
        F(16) = f2 * g3 - f3 * g2 - h1;
        F(17) = f1 * g3 - f3 * g1 - h2;
        F(18) = f1 * g2 - f2 * g1 - h3;
    end
    x0 = [1000, 800, 600, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1];
    %options = optimoptions('fsolve', 'Algorithm', 'Levenberg-Marquardt');
    %options = optimoptions(options, 'MaxIterations', 10000000);
    %options = optimoptions(options, 'MaxFunctionEvaluations', 10000000);
    %options = optimoptions(options, 'OptimalityTolerance', 1);
    sln = fsolve(@parameters_eqns, x0)%, options)
    i_t = [1 0 sln(3);
           0 1 sln(2);
           0 0      1];
    i_rf = [0 -1  0;
            1  0  0;
            0  0 -1];
    i_s = [sln(2)      0 0;
                0 sln(2) 0;
                0      0 1];
    e_r = [ sln(7)  sln(8)  sln(9) 0;
           sln(10) sln(11) sln(12) 0;
           sln(13) sln(14) sln(15) 0;
                 0       0       0 1];
    e_t = [1 0 0 -sln(4);
           0 1 0 -sln(5);
           0 0 1 -sln(6);
           0 0 0       1];
end