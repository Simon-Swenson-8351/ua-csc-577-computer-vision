function [] = hw9()
    part_a();
    part_b();

    'finished'
    input('');
end

function [] = part_a()
    sunset_im = int_to_dbl_img(imread('sunset.tiff'));
    tiger_1_im = int_to_dbl_img(imread('tiger-1.tiff'));
    tiger_2_im = int_to_dbl_img(imread('tiger-2.tiff'));

    figure();
    imshow(sunset_im);

    sunset_5_im = reshape(sunset_im, size(sunset_im)(1) * size(sunset_im)(2), size(sunset_im)(3));
    [sunset_5_assignments, sunset_5_means] = k_means(sunset_5_im, 5, 0.0001);
    sunset_5_im = map_means(sunset_5_im, sunset_5_assignments, sunset_5_means);
    sunset_5_im = reshape(sunset_5_im, size(sunset_im)(1), size(sunset_im)(2), size(sunset_im)(3));
    figure();
    imshow(sunset_5_im);

    sunset_10_im = reshape(sunset_im, size(sunset_im)(1) * size(sunset_im)(2), size(sunset_im)(3));
    [sunset_10_assignments, sunset_10_means] = k_means(sunset_10_im, 10, 0.0001);
    sunset_10_im = map_means(sunset_10_im, sunset_10_assignments, sunset_10_means);
    sunset_10_im = reshape(sunset_10_im, size(sunset_im)(1), size(sunset_im)(2), size(sunset_im)(3));
    figure();
    imshow(sunset_10_im);
end

function [] = part_b()
end

function [dbl_img] = int_to_dbl_img(int_img)
    dbl_img = double(int_img) / 255.0;
end

function [out_im] = map_means(in_im, assignments, means)
    out_im = in_im;
    for i = 1:size(means)(3)
        idc = find(assignments == i);
        out_im(idc, :) = repmat(means(1, :, i), size(idc)(1), 1);
    end
end

% data is assumed to be a matrix with each row a different data point.
function [cluster_assignments, means] = k_means(data, k, epsilon)
    data_size = size(data);
    % This shape seems weird, but it lets us broadcast means along the 3rd 
    % dimension of the data tensor.
    means = zeros(1, data_size(2), k);
    for i = 1:k
        means(1, :, i) = data(ceil(rand(1) * size(data)(1)), :) + rand(1, 3) .* 0.01;
    end

    % Easy way to compute the distances: broadcast data - means, a |D| x |x| x k 
    % matrix.
    cur_distances = repmat(data, 1, 1, k);
    % |D| x 1 x k matrix
    cur_distances = sqrt(sum((cur_distances - means) .^ 2, 2));
    [cur_distances_min, cluster_assignments] = min(cur_distances, [], 3);
    prev_err = Inf;
    cur_err = k_means_err(data, cluster_assignments, means)
    while prev_err - cur_err > epsilon

        for i = 1:k
            cur_cluster_data = data(find(cluster_assignments == i), :);
            means(1, :, i) = sum(cur_cluster_data, 1) ./ size(cur_cluster_data)(1);
        end
        cur_distances = repmat(data, 1, 1, k);
        cur_distances = sqrt(sum((cur_distances - means) .^ 2, 2));
        [cur_distances_min, cluster_assignments] = min(cur_distances, [], 3);

        prev_err = cur_err;
        cur_err = k_means_err(data, cluster_assignments, means)
    end
end

function [err] = k_means_err(data, cluster_assignments, means)
    err = 0.0;
    for i = 1:size(means)(3)
        cur_cluster_data = data(find(cluster_assignments == i), :);
        err += sum(sum(cur_cluster_data - means(1, :, i) .^ 2, 2), 1);
    end
end