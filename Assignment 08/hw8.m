function [] = hw8()

    pkg load statistics;
    part_b();
    part_c2();

    'finished'
    input('');
end

function [] = part_b()
    climber_im_int = imread('climber.tiff');
    climber_im_dbl = int_to_dbl_img(climber_im_int);
    climber_im_size = size(climber_im_dbl);
    figure();
    imshow(climber_im_dbl);
    imwrite(climber_im_dbl, 'figs/climber_im.png');

    # Note, since this is estimating gradient values, it's actually sampling 
    # between x pixels. The image size is (w_old + 1, h_old).
    finite_differences_x_filter = [1.0 -1.0;
                                   1.0 -1.0];
    finite_differences_x_response = conv2(climber_im_dbl, finite_differences_x_filter);

    figure();
    imshow(scale_dbl_img_to_1(finite_differences_x_filter));
    imwrite(scale_dbl_img_to_1(finite_differences_x_filter), 'figs/finite_differences_x_filter.png');

    figure();
    imshow(scale_dbl_img_to_1(finite_differences_x_response));
    imwrite(scale_dbl_img_to_1(finite_differences_x_response), 'figs/finite_differences_x_response.png');

    finite_differences_y_filter = [1.0  1.0;
                                  -1.0 -1.0];
    finite_differences_y_response = conv2(climber_im_dbl, finite_differences_y_filter);

    figure();
    imshow(scale_dbl_img_to_1(finite_differences_y_filter));
    imwrite(scale_dbl_img_to_1(finite_differences_y_filter), 'figs/finite_differences_y_filter.png');

    figure();
    imshow(scale_dbl_img_to_1(finite_differences_y_response));
    imwrite(scale_dbl_img_to_1(finite_differences_y_response), 'figs/finite_differences_y_response.png');

    gradient_magnitude = sqrt(finite_differences_x_response .^ 2 + finite_differences_y_response .^ 2);

    figure();
    imshow(scale_dbl_img_to_1(gradient_magnitude));
    imwrite(scale_dbl_img_to_1(gradient_magnitude), 'figs/gradient_magnitude.png');

    threshold_above = 0.35;
    gradient_magnitude_size = size(gradient_magnitude);
    threshold_response = ones(gradient_magnitude_size);
    threshold_response(find(gradient_magnitude > threshold_above)) = 0.0;
    
    figure();
    imshow(threshold_response);
    imwrite(threshold_response, 'figs/threshold_response.png');

    gaussian_filter = gen_gaussian_filter(2.0, 13);

    figure();
    imshow(scale_dbl_img_to_1(gaussian_filter));
    imwrite(scale_dbl_img_to_1(gaussian_filter), 'figs/gaussian_filter.png');

    smoothed = conv2(climber_im_dbl, gaussian_filter);

    figure();
    imshow(scale_dbl_img_to_1(smoothed));
    imwrite(scale_dbl_img_to_1(smoothed), 'figs/smoothed.png');

    smoothed_dx = conv2(smoothed, finite_differences_x_filter);
    smoothed_dy = conv2(smoothed, finite_differences_y_filter);
    smoothed_gradient_magnitude = sqrt(smoothed_dx .^ 2 + smoothed_dy .^ 2);

    figure();
    imshow(scale_dbl_img_to_1(smoothed_gradient_magnitude));
    imwrite(scale_dbl_img_to_1(smoothed_gradient_magnitude), 'figs/smoothed_gradient_magnitude.png');

    smoothed_threshold_above = 0.08;
    smoothed_gradient_magnitude_size = size(smoothed_gradient_magnitude);
    smoothed_threshold_response = ones(smoothed_gradient_magnitude_size);
    smoothed_threshold_response(find(smoothed_gradient_magnitude > smoothed_threshold_above)) = 0.0;
    
    figure();
    imshow(smoothed_threshold_response);
    imwrite(smoothed_threshold_response, 'figs/smoothed_threshold_response.png');

    gaussian_filter_2 = gen_gaussian_filter(4.0, 23);

    figure();
    imshow(scale_dbl_img_to_1(gaussian_filter_2));
    imwrite(scale_dbl_img_to_1(gaussian_filter_2), 'figs/gaussian_filter_2.png');

    blur_filter_x = conv2(gaussian_filter_2, finite_differences_x_filter);

    figure();
    imshow(scale_dbl_img_to_1(blur_filter_x));
    imwrite(scale_dbl_img_to_1(blur_filter_x), 'figs/blur_filter_x.png');

    blur_filter_y = conv2(gaussian_filter_2, finite_differences_y_filter);

    figure();
    imshow(scale_dbl_img_to_1(blur_filter_y));
    imwrite(scale_dbl_img_to_1(blur_filter_y), 'figs/blur_filter_y.png');

    smoothed_dx_2 = conv2(climber_im_dbl, blur_filter_x);
    smoothed_dy_2 = conv2(climber_im_dbl, blur_filter_y);
    smoothed_gradient_magnitude_2 = sqrt(smoothed_dx_2 .^ 2 + smoothed_dy_2 .^ 2);

    figure();
    imshow(scale_dbl_img_to_1(smoothed_gradient_magnitude_2));
    imwrite(scale_dbl_img_to_1(smoothed_gradient_magnitude_2), 'figs/smoothed_gradient_magnitude_2.png');

    smoothed_threshold_above_2 = 0.05;
    smoothed_gradient_magnitude_size_2 = size(smoothed_gradient_magnitude_2);
    smoothed_threshold_response_2 = ones(smoothed_gradient_magnitude_size_2);
    smoothed_threshold_response_2(find(smoothed_gradient_magnitude_2 > smoothed_threshold_above_2)) = 0.0;
    
    figure();
    imshow(smoothed_threshold_response_2);
    imwrite(smoothed_threshold_response_2, 'figs/smoothed_threshold_response_2.png');

    gaussian_filter_1d_x = reshape(gen_gaussian_filter_1d(2.0, 13), 1, 13);
    gaussian_filter_1d_y = reshape(gaussian_filter_1d_x, 13, 1);

    figure();
    imshow(scale_dbl_img_to_1(conv2(gaussian_filter_1d_x, gaussian_filter_1d_y)));
    imwrite(scale_dbl_img_to_1(conv2(gaussian_filter_1d_x, gaussian_filter_1d_y)), 'figs/combined_1d_gaussians.png');

    smoothed_separate = conv2(climber_im_dbl, gaussian_filter_1d_x);
    smoothed_separate = conv2(smoothed, gaussian_filter_1d_y);
    
    figure();
    imshow(scale_dbl_img_to_1(smoothed_separate));
    imwrite(scale_dbl_img_to_1(smoothed_separate), 'figs/smoothed_separate.png');
end

function [] = part_c2()
    num_a = 155;
    num_e = 195;
    num_i = 205;
    num_o = 79;
    num_u = 166;
    
    lorem_ipsum_im_int = imread('lorem_ipsum.tiff');
    lorem_ipsum_im_dbl = int_to_dbl_img(lorem_ipsum_im_int);

    # We assume each of these filter images are the same size to simplify 
    # things.
    a_filter_int = imread('character-filters/a.tiff');
    a_filter_dbl = int_to_dbl_img(a_filter_int);
    a_filter_dbl = filter_from_im(a_filter_dbl, [20 20]);

    e_filter_int = imread('character-filters/e.tiff');
    e_filter_dbl = int_to_dbl_img(e_filter_int);
    e_filter_dbl = filter_from_im(e_filter_dbl, [20 20]);

    i_filter_int = imread('character-filters/i.tiff');
    i_filter_dbl = int_to_dbl_img(i_filter_int);
    i_filter_dbl = filter_from_im(i_filter_dbl, [20 20]);

    o_filter_int = imread('character-filters/o.tiff');
    o_filter_dbl = int_to_dbl_img(o_filter_int);
    o_filter_dbl = filter_from_im(o_filter_dbl, [20 20]);

    u_filter_int = imread('character-filters/u.tiff');
    u_filter_dbl = int_to_dbl_img(u_filter_int);
    u_filter_dbl = filter_from_im(u_filter_dbl, [20 20]);

    imwrite(scale_dbl_img_to_1(a_filter_dbl), 'a_filter.tiff');
    imwrite(scale_dbl_img_to_1(e_filter_dbl), 'e_filter.tiff');
    imwrite(scale_dbl_img_to_1(i_filter_dbl), 'i_filter.tiff');
    imwrite(scale_dbl_img_to_1(o_filter_dbl), 'o_filter.tiff');
    imwrite(scale_dbl_img_to_1(u_filter_dbl), 'u_filter.tiff');

    imwrite(scale_dbl_img_to_1(conv2(i_filter_dbl, i_filter_dbl)), 'i_test.tiff');

    a_response = conv2(lorem_ipsum_im_dbl, a_filter_dbl);
    e_response = conv2(lorem_ipsum_im_dbl, e_filter_dbl);
    i_response = conv2(lorem_ipsum_im_dbl, i_filter_dbl);
    o_response = conv2(lorem_ipsum_im_dbl, o_filter_dbl);
    u_response = conv2(lorem_ipsum_im_dbl, u_filter_dbl);


    imwrite(scale_dbl_img_to_1(a_response), 'a_response.tiff');
    imwrite(scale_dbl_img_to_1(e_response), 'e_response.tiff');
    imwrite(scale_dbl_img_to_1(i_response), 'i_response.tiff');
    imwrite(scale_dbl_img_to_1(o_response), 'o_response.tiff');
    imwrite(scale_dbl_img_to_1(u_response), 'u_response.tiff');

    a_color = reshape([1.0 0.0 0.0], 1, 1, 3);
    e_color = reshape([0.0 1.0 0.0], 1, 1, 3);
    i_color = reshape([0.0 0.0 1.0], 1, 1, 3);
    o_color = reshape([1.0 1.0 0.0], 1, 1, 3);
    u_color = reshape([0.0 1.0 1.0], 1, 1, 3);

    response_size = size(a_response);
    color_coded = ones(response_size(1), response_size(2), 3);

    a_threshold_val = binary_search_threshold(a_response, num_a, 3, 0.0001)#max(max(a_response)) .* 0.85;
    e_threshold_val = binary_search_threshold(e_response, num_e, 3, 0.0001)#max(max(e_response)) .* 0.85;
    i_threshold_val = binary_search_threshold(i_response, num_i, 3, 0.0001)#max(max(i_response)) .* 0.875;
    o_threshold_val = binary_search_threshold(o_response, num_o, 3, 0.0001)#max(max(o_response)) .* 0.85;
    u_threshold_val = binary_search_threshold(u_response, num_u, 3, 0.0001)#max(max(u_response)) .* 0.875;

    color_coded = draw_boxes(color_coded, find(a_response > a_threshold_val), a_color, 9);
    color_coded = draw_boxes(color_coded, find(e_response > e_threshold_val), e_color, 9);
    color_coded = draw_boxes(color_coded, find(i_response > i_threshold_val), i_color, 9);
    color_coded = draw_boxes(color_coded, find(o_response > o_threshold_val), o_color, 9);
    color_coded = draw_boxes(color_coded, find(u_response > u_threshold_val), u_color, 9);

    imwrite(color_coded, 'color_coded.png');
end

function [threshold] = binary_search_threshold(response_img, target_num_hits, distance_to_consider_same, min_delta)
    response_img_size = size(response_img);
    max_response = max(max(response_img));
    min_response = min(min(response_img));
    mid_response = (max_response + min_response) / 2;
    num_hits = -1;
    while (max_response - min_response > min_delta && (num_hits != target_num_hits))
        indices = find(response_img > mid_response);
        indices_size = size(indices);
        #nonclose_indices = [];
        #for i = 1:indices_size(1)
        #    loc_cur = indices(i);
        #    [y, x] = ind2sub(response_img_size, loc_cur);
        #    nonclose_indices_size = size(nonclose_indices);
        #    nonclose = true;
        #    for j = 1:nonclose_indices_size(1)
        #        [y_o, x_o] = ind2sub(response_img_size, nonclose_indices(j));
        #        if abs(y_o - y) <= distance_to_consider_same || abs(x_o - x) <= distance_to_consider_same
        #            nonclose = false;
        #            break;
        #        end
        #    end
        #    if nonclose
        #        nonclose_indices = [nonclose_indices(:); loc_cur];
        #    end
        #end
        #nonclose_indices_size = size(nonclose_indices);
        #num_hits = nonclose_indices_size(1)
        num_hits = indices_size(1);
        if num_hits < target_num_hits
            # We didn't find enough, so we need to lower the threshold
            max_response = mid_response;
            mid_response = (max_response + min_response) / 2;
        elseif num_hits > target_num_hits
            # We found too many, so we need to raise the threshold
            min_response = mid_response;
            mid_response = (max_response + min_response) / 2;
        endif
    end
    threshold = mid_response;
end

# color should be 1x1x3
function [im_out] = draw_boxes(im_in, locs, color, box_size)
    im_in_size = size(im_in);
    color_block = zeros(box_size, box_size, 3);
    color_block(:, :, 1) = color(1, 1, 1);
    color_block(:, :, 2) = color(1, 1, 2);
    color_block(:, :, 3) = color(1, 1, 3);
    im_out = im_in;
    locs_size = size(locs);
    for i = 1:locs_size(1)
        loc_cur = locs(i);
        [loc_y, loc_x] = ind2sub(im_in_size, loc_cur);
        y_start = round(loc_y - box_size/2);
        y_range = y_start:(y_start + box_size - 1);
        x_start = round(loc_x - box_size/2);
        x_range = x_start:(x_start + box_size - 1);
        im_out(y_range, x_range, :) = color_block;
    end
end

function [filter_out] = filter_from_im(im, padding_size)
    tmp = scale_dbl_img_to_1(im);
    tmp = tmp - mean(mean(tmp));
    tmp = tmp .* max([-min(min(tmp)); max(max(tmp))]);
    tmp_size = size(tmp);
    filter_out = zeros(padding_size);
    start_y = round(padding_size(1)/2 - tmp_size(1)/2);
    range_y = start_y:(start_y + tmp_size(1) - 1);
    start_x = round(padding_size(2)/2 - tmp_size(2)/2);
    range_x = start_x:(start_x + tmp_size(2) - 1);
    filter_out(range_y, range_x) = tmp;
    # Recall that convolutions flip the filter vertically and horizontally. 
    # Counteract that by flipping here.
    filter_out = flip(flip(filter_out, 1), 2);
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
    scaled_dbl_img = unscaled_dbl_img - min(min(unscaled_dbl_img));
    scaled_dbl_img = scaled_dbl_img ./ max(max(scaled_dbl_img));
end

# This is a lot slower than conv2, but I implemented it anyway. Used in earlier 
# iterations, but now just use conv2.
function [convolved] = conv_2d(a, b)
    b_flipped = flip(flip(b, 1), 2);
    a_size = size(a);
    b_size = size(b);
    # Ensure there are enough 0's on the edges 
    a_padded = zeros(a_size(1) + 2 * b_size(1) - 2, a_size(1) + 2 * b_size(1) - 2);
    a_padded_size = size(a_padded);
    a_padded(b_size(1):(b_size(1) + a_size(1) - 1), b_size(2):(b_size(2) + a_size(2) - 1)) = a;
    convolved = zeros(b_size(1) + a_size(1) - 1, b_size(2) + a_size(2) - 1);
    convolved_size = size(convolved);
    for i = 1:convolved_size(1)
        for j = 1:convolved_size(2)
            convolved(i, j) = sum(sum(a_padded(i:(i + b_size(1) - 1), j:(j + b_size(2) - 1)) .* b_flipped));
        end
    end
end

function [gaussian_filter] = gen_gaussian_filter(sigma, window)
    size_candidate = window;
    if mod(size_candidate, 2) == 0
        size_candidate += 1;
    end
    gaussian_filter = zeros(size_candidate);
    gaussian_filter_size = size(gaussian_filter);
    samples = (-(gaussian_filter_size(1) - 1) / 2):((gaussian_filter_size(1) - 1) / 2);
    [X1, X2] = meshgrid(samples, samples);
    mvn_z = mvnpdf([X1(:) X2(:)], [0; 0], [(sigma .^ 2) 0.0; 0.0 (sigma .^ 2)]);

    figure();
    surf(samples, samples, reshape(mvn_z, size_candidate, size_candidate));
    figure();

    gaussian_filter = reshape(mvn_z, size_candidate, size_candidate);
    gaussian_filter = gaussian_filter ./ sum(sum(gaussian_filter));
end

# A multivariate gaussian can be represented as:
# g(x) = (2 * pi)^(-1/2) * sigma ^ -1 * e ^ (-x^2 / (2 sigma^2))
# h(y) = (2 * pi)^(-1/2) * sigma ^ -1 * e ^ (-y^2 / (2 sigma^2))

function [gaussian_filter] = gen_gaussian_filter_1d(sigma, window)
    size_candidate = window;
    if mod(size_candidate, 2) == 0
        size_candidate += 1;
    end
    gaussian_filter = zeros(size_candidate, 1);
    gaussian_filter_size = size(gaussian_filter);
    samples = (-(gaussian_filter_size(1) - 1) / 2):((gaussian_filter_size(1) - 1) / 2);

    gaussian_filter = normpdf(samples, 0, sigma);
    gaussian_filter = gaussian_filter ./ sum(gaussian_filter);
end