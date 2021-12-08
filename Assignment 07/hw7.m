function [] = hw7()

    swatch_white_int = imread('color_constancy_images/macbeth_syl-50MR16Q.tif');
    swatch_white_dbl = int_to_dbl_img(swatch_white_int);

    swatch_bluish_int = imread('color_constancy_images/macbeth_solux-4100.tif');
    swatch_bluish_dbl = int_to_dbl_img(swatch_bluish_int);

    % The range of pixels where the white square of the swatch is.
    white_square_range_row = 329:379;
    white_square_range_col = 106:148;

    white_mean_dbl = mean(swatch_white_dbl(white_square_range_row, white_square_range_col, :), [1 2]);
    white_mean_int = dbl_to_int_img(white_mean_dbl);
    white_mean_int_scaled = scale_int_img_to_250(white_mean_int)

    bluish_mean_dbl = mean(swatch_bluish_dbl(white_square_range_row, white_square_range_col, :), [1 2]);
    bluish_mean_int = dbl_to_int_img(bluish_mean_dbl);
    bluish_mean_int_scaled = scale_int_img_to_250(bluish_mean_int)

    ang_err = angular_error(white_mean_dbl, bluish_mean_dbl)

    figure();
    imshow(scale_int_img_to_250(swatch_bluish_int));

    % Instead of representing this as a matrix in matlab, we can use
    % broadcasting to instead represent it as a vector.
    transform = white_mean_dbl ./ bluish_mean_dbl;
    predicted = scale_dbl_img_to_1(swatch_bluish_dbl .* transform);
    
    figure();
    imshow(scale_int_img_to_250(dbl_to_int_img(predicted)));

    figure();
    imshow(scale_int_img_to_250(swatch_white_int));
    
    swatch_original_rmse_rg = rmse_rg(swatch_bluish_dbl, swatch_white_dbl)
    swatch_predicted_rmse_rg = rmse_rg(predicted, swatch_white_dbl)
    
    
    
    
    
    apples_white_int = imread('color_constancy_images/apples2_syl-50MR16Q.tif');
    apples_white_dbl = int_to_dbl_img(apples_white_int);
    
    apples_bluish_int = imread('color_constancy_images/apples2_solux-4100.tif');
    apples_bluish_dbl = int_to_dbl_img(apples_bluish_int);
    
    
    
    ball_white_int = imread('color_constancy_images/ball_syl-50MR16Q.tif');
    ball_white_dbl = int_to_dbl_img(ball_white_int);
    
    ball_bluish_int = imread('color_constancy_images/ball_solux-4100.tif');
    ball_bluish_dbl = int_to_dbl_img(ball_bluish_int);
    
    
    
    blocks_white_int = imread('color_constancy_images/blocks1_syl-50MR16Q.tif');
    blocks_white_dbl = int_to_dbl_img(blocks_white_int);
    
    blocks_bluish_int = imread('color_constancy_images/blocks1_solux-4100.tif');
    blocks_bluish_dbl = int_to_dbl_img(blocks_bluish_int);
    
    
    
    % Show before images
    figure();
    imshow(scale_int_img_to_250(dbl_to_int_img(apples_bluish_dbl)));
    figure();
    imshow(scale_int_img_to_250(dbl_to_int_img(ball_bluish_dbl)));
    figure();
    imshow(scale_int_img_to_250(dbl_to_int_img(blocks_bluish_dbl)));
    
    
    
    apples_max_rgb_bluish_light_dbl = max_rgb(apples_bluish_dbl);
    apples_max_rgb_bluish_light_int = scale_int_img_to_250(dbl_to_int_img(apples_max_rgb_bluish_light_dbl))
    apples_angular_error = angular_error(apples_max_rgb_bluish_light_dbl, bluish_mean_dbl)
    
    ball_max_rgb_bluish_light_dbl = max_rgb(ball_bluish_dbl);
    ball_max_rgb_bluish_light_int = scale_int_img_to_250(dbl_to_int_img(ball_max_rgb_bluish_light_dbl))
    ball_angular_error = angular_error(ball_max_rgb_bluish_light_dbl, bluish_mean_dbl)
    
    blocks_max_rgb_bluish_light_dbl = max_rgb(blocks_bluish_dbl);
    blocks_max_rgb_bluish_light_int = scale_int_img_to_250(dbl_to_int_img(blocks_max_rgb_bluish_light_dbl))
    blocks_angular_error = angular_error(blocks_max_rgb_bluish_light_dbl, bluish_mean_dbl)
    
    % Use the oracle white light when computing the transform
    apples_transform = white_mean_dbl ./ apples_max_rgb_bluish_light_dbl;
    apples_predicted_dbl = scale_dbl_img_to_1(apples_bluish_dbl .* apples_transform);
    figure();
    imshow(scale_int_img_to_250(dbl_to_int_img(apples_predicted_dbl)));
    apples_max_rgb_rmse_rg = rmse_rg(apples_predicted_dbl, apples_white_dbl)
    
    ball_transform = white_mean_dbl ./ ball_max_rgb_bluish_light_dbl;
    ball_predicted_dbl = scale_dbl_img_to_1(ball_bluish_dbl .* ball_transform);
    figure();
    imshow(scale_int_img_to_250(dbl_to_int_img(ball_predicted_dbl)));
    ball_max_rgb_rmse_rg = rmse_rg(ball_predicted_dbl, ball_white_dbl)
    
    blocks_transform = white_mean_dbl ./ blocks_max_rgb_bluish_light_dbl;
    blocks_predicted_dbl = scale_dbl_img_to_1(blocks_bluish_dbl .* blocks_transform);
    figure();
    imshow(scale_int_img_to_250(dbl_to_int_img(blocks_predicted_dbl)));
    blocks_max_rgb_rmse_rg = rmse_rg(blocks_predicted_dbl, blocks_white_dbl)
    
    
    
    apples_gray_world_bluish_light_dbl = gray_world(apples_bluish_dbl);
    apples_gray_world_bluish_light_int = scale_int_img_to_250(dbl_to_int_img(apples_gray_world_bluish_light_dbl))
    apples_angular_error = angular_error(apples_gray_world_bluish_light_dbl, bluish_mean_dbl)
    
    ball_gray_world_bluish_light_dbl = gray_world(ball_bluish_dbl);
    ball_gray_world_bluish_light_int = scale_int_img_to_250(dbl_to_int_img(ball_gray_world_bluish_light_dbl))
    ball_angular_error = angular_error(ball_gray_world_bluish_light_dbl, bluish_mean_dbl)
    
    blocks_gray_world_bluish_light_dbl = gray_world(blocks_bluish_dbl);
    blocks_gray_world_bluish_light_int = scale_int_img_to_250(dbl_to_int_img(blocks_gray_world_bluish_light_dbl))
    blocks_angular_error = angular_error(blocks_gray_world_bluish_light_dbl, bluish_mean_dbl)
    
    % Use the oracle white light when computing the transform
    apples_transform = white_mean_dbl ./ apples_gray_world_bluish_light_dbl;
    apples_predicted_dbl = scale_dbl_img_to_1(apples_bluish_dbl .* apples_transform);
    figure();
    imshow(scale_int_img_to_250(dbl_to_int_img(apples_predicted_dbl)));
    apples_gray_world_rmse_rg = rmse_rg(apples_predicted_dbl, apples_white_dbl)
    
    ball_transform = white_mean_dbl ./ ball_gray_world_bluish_light_dbl;
    ball_predicted_dbl = scale_dbl_img_to_1(ball_bluish_dbl .* ball_transform);
    figure();
    imshow(scale_int_img_to_250(dbl_to_int_img(ball_predicted_dbl)));
    ball_gray_world_rmse_rg = rmse_rg(ball_predicted_dbl, ball_white_dbl)
   
    blocks_transform = white_mean_dbl ./ blocks_gray_world_bluish_light_dbl;
    blocks_predicted_dbl = scale_dbl_img_to_1(blocks_bluish_dbl .* blocks_transform);
    figure();
    imshow(scale_int_img_to_250(dbl_to_int_img(blocks_predicted_dbl)));
    blocks_gray_world_rmse_rg = rmse_rg(blocks_predicted_dbl, blocks_white_dbl)
    
    
    
    % Show after images
    figure();
    imshow(scale_int_img_to_250(dbl_to_int_img(apples_white_dbl)));
    figure();
    imshow(scale_int_img_to_250(dbl_to_int_img(ball_white_dbl)));
    figure();
    imshow(scale_int_img_to_250(dbl_to_int_img(blocks_white_dbl)));
    
    % Error minimization result
    apple_optimal_lse_light_dbl = compute_optimal_lse_light(apples_bluish_dbl, apples_white_dbl, white_mean_dbl);
    apple_optimal_lse_light_int = scale_int_img_to_250(dbl_to_int_img(apple_optimal_lse_light_dbl))
    apples_angular_error = angular_error(apple_optimal_lse_light_dbl, bluish_mean_dbl)
    
    apples_better_predicted_dbl = scale_dbl_img_to_1(apples_bluish_dbl .* (white_mean_dbl ./ apple_optimal_lse_light_dbl));
    apples_optimal_lse_rmse_rg = rmse_rg(apples_better_predicted_dbl, apples_white_dbl)
    
    figure();
    imshow(scale_int_img_to_250(dbl_to_int_img(apples_better_predicted_dbl)));
    
    ball_optimal_lse_light_dbl = compute_optimal_lse_light(ball_bluish_dbl, ball_white_dbl, white_mean_dbl);
    ball_optimal_lse_light_int = scale_int_img_to_250(dbl_to_int_img(ball_optimal_lse_light_dbl))
    ball_angular_error = angular_error(ball_optimal_lse_light_dbl, bluish_mean_dbl)
    
    ball_better_predicted_dbl = scale_dbl_img_to_1(ball_bluish_dbl .* (white_mean_dbl ./ ball_optimal_lse_light_dbl));
    ball_optimal_lse_rmse_rg = rmse_rg(ball_better_predicted_dbl, ball_white_dbl)
    
    figure();
    imshow(scale_int_img_to_250(dbl_to_int_img(ball_better_predicted_dbl)));
    
    blocks_optimal_lse_light_dbl = compute_optimal_lse_light(blocks_bluish_dbl, blocks_white_dbl, white_mean_dbl);
    blocks_optimal_lse_light_int = scale_int_img_to_250(dbl_to_int_img(blocks_optimal_lse_light_dbl))
    blocks_angular_error = angular_error(blocks_optimal_lse_light_dbl, bluish_mean_dbl)
    
    blocks_better_predicted_dbl = scale_dbl_img_to_1(blocks_bluish_dbl .* (white_mean_dbl ./ blocks_optimal_lse_light_dbl));
    blocks_optimal_lse_rmse_rg = rmse_rg(blocks_better_predicted_dbl, blocks_white_dbl)
    
    figure();
    imshow(scale_int_img_to_250(dbl_to_int_img(blocks_better_predicted_dbl)));
    
    my_simulanneal_result = reshape(my_simulanneal(apples_bluish_dbl, apples_white_dbl, white_mean_dbl), 1, 1, 3);
    my_simulanneal_result_int = scale_int_img_to_250(dbl_to_int_img(my_simulanneal_result))
    my_simulanneal_ang_err = angular_error(my_simulanneal_result, bluish_mean_dbl)
    my_simulanneal_rmse = rmse_rg(apples_bluish_dbl .* (white_mean_dbl ./ reshape(my_simulanneal_result, 1, 1, 3)), apples_white_dbl)
    
    my_simulanneal_result = reshape(my_simulanneal(ball_bluish_dbl, ball_white_dbl, white_mean_dbl), 1, 1, 3);
    my_simulanneal_result_int = scale_int_img_to_250(dbl_to_int_img(my_simulanneal_result))
    my_simulanneal_ang_err = angular_error(my_simulanneal_result, bluish_mean_dbl)
    my_simulanneal_rmse = rmse_rg(ball_bluish_dbl .* (white_mean_dbl ./ reshape(my_simulanneal_result, 1, 1, 3)), ball_white_dbl)
    
    my_simulanneal_result = reshape(my_simulanneal(blocks_bluish_dbl, blocks_white_dbl, white_mean_dbl), 1, 1, 3);
    my_simulanneal_result_int = scale_int_img_to_250(dbl_to_int_img(my_simulanneal_result))
    my_simulanneal_ang_err = angular_error(my_simulanneal_result, bluish_mean_dbl)
    my_simulanneal_rmse = rmse_rg(blocks_bluish_dbl .* (white_mean_dbl ./ reshape(my_simulanneal_result, 1, 1, 3)), blocks_white_dbl)
end


% Recall that the dot product between a, b is equal to the magnitude of a 
% times the magnitude of b times cosine of theta, where theta is the angle
% between the two. This allows us to do some math where we come up with
% theta: theta = cos^-1((a dot b)/(mag(a) * mag(b)))
% in1 and in2 are assumed to be 2-d matrices of the same size, with each 
% column representing a vector. They need not be units.
function [result] = angular_error(in1, in2)
    % dot products
    result = in1 .* in2;
    result = sum(result);
    
    % magnitudes
    in1_mag = sqrt(sum(in1 .^ 2));
    in2_mag = sqrt(sum(in2 .^ 2));
    result = acos(result ./ (in1_mag .* in2_mag));
end

% Assumed that predicted and target are both images of the same size in 
% double-space (with values normed between 0 and 1).
% This will also first scale by 255 so that the error is presented in terms
% of the correct domain (uint8, 0-255).
function [result] = rmse_rg(predicted, target)
    predicted = predicted * 255.0;
    target = target * 255.0;
    predicted_size = size(predicted);
    total_elems = predicted_size(1) * predicted_size(2) * 2;
    
    p_rg = zeros(predicted_size(1), predicted_size(2), 2);
    predicted_sum = sum(predicted, 3);
    p_rg(:, :, 1) = predicted(:, :, 1) ./ predicted_sum;
    p_rg(:, :, 2) = predicted(:, :, 2) ./ predicted_sum;
    
    t_rg = zeros(predicted_size(1), predicted_size(2), 2);
    target_sum = sum(target, 3);
    t_rg(:, :, 1) = target(:, :, 1) ./ target_sum;
    t_rg(:, :, 2) = target(:, :, 2) ./ target_sum;
    
    elemwise_se = (p_rg - t_rg) .^ 2;
    
    % Tragically, we need a loop here, since some entries must be
    % discarded, I'm sure there's a more idiomatic way to do this, but not
    % going to bother for now.
    result = 0.0;
    e_N = 0;
    for i = 1:predicted_size(1)
        for j = 1:predicted_size(2)
            for k = 1:2
                if predicted_sum(i, j) > 10.0 && target_sum(i, j) > 10.0
                    result = result + elemwise_se(i, j, k);
                    e_N = e_N + 1;
                end
            end
        end
    end
    result = sqrt(result / e_N);
end

% Will return the light for the corresponding image using the Max RGB
% algorithm.
function [light] = max_rgb(image)
    light = max(image, [], [1 2]);
end

% Will return the light for the corresponding image using the Gray world
% algorithm.
function [light] = gray_world(image)
    light = mean(image, [1 2]);
end

function [int_img] = dbl_to_int_img(dbl_img)
    int_img = uint8(dbl_img * 255);
end

function [dbl_img] = int_to_dbl_img(int_img)
    dbl_img = double(int_img) / 255.0;
end

% Scales an image so that the brightest scalar that appears in image x, y,
% c is scaled to 250.
% This function is not invertable, since we lose information about what,
% exactly, the max value was.
function [scaled_int_img] = scale_int_img_to_250(unscaled_int_img)
    % Could get away with int32 or uint16 here, but I'm not feeling it.
    scaled_int_img = uint8(int64(unscaled_int_img) * 250 / int64(max(unscaled_int_img, [], 'all')));
end

function [scaled_dbl_img] = scale_dbl_img_to_1(unscaled_dbl_img)
    scaled_dbl_img = unscaled_dbl_img / max(unscaled_dbl_img, [], 'all');
end

% src_img, targ_img, and targ_light are assumed to be double.
% We are minimizing the mean squared error metric here, solving
% analytically for when the gradient is 0 in all dimensions.
function [src_light] = compute_optimal_lse_light(src_img, targ_img, targ_light)
    src_light = targ_light .* sum(src_img .^ 2, [1 2]) ./ sum(src_img .* targ_img, [1 2]);
end

function [best_result] = my_simulanneal(src_img, targ_img, targ_light)
    function [err] = fn_candidate(X)
        X = reshape(X, 1, 1, 3);
        predicted_img = src_img .* (targ_light ./ X);
        err = rmse_rg(predicted_img, targ_img);
    end
    best_result = simulannealbnd(@fn_candidate, [0.5 0.5 0.5], [0.0 0.0 0.0], [1.0 1.0 1.0]);
end