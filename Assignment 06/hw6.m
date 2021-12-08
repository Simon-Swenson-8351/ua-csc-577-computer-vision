light_directions = importdata('light_directions.txt');
im1 = imread('4-1.tiff');
im2 = imread('4-2.tiff');
im3 = imread('4-3.tiff');
im4 = imread('4-4.tiff');
im5 = imread('4-5.tiff');
im6 = imread('4-6.tiff');
im7 = imread('4-7.tiff');
im_size = size(im1);

ims_stacked = zeros(im_size(1), im_size(2), 7);
ims_stacked(:, :, 1) = im1(:, :, 1);
ims_stacked(:, :, 2) = im2(:, :, 1);
ims_stacked(:, :, 3) = im3(:, :, 1);
ims_stacked(:, :, 4) = im4(:, :, 1);
ims_stacked(:, :, 5) = im5(:, :, 1);
ims_stacked(:, :, 6) = im6(:, :, 1);
ims_stacked(:, :, 7) = im7(:, :, 1);

normals = zeros(im_size(1), im_size(2), 3);
intensities = zeros(1, 7);

for i = 1:im_size(1)
    for j = 1:im_size(2)
        intensities(1, :) = ims_stacked(i, j, :);
        normals(i, j, :) = (light_directions' * light_directions)^(-1) * light_directions' * intensities';
    end
end

test_light = reshape([0, 0, 1], 1, 1, 3);

normals_lens = sqrt(sum(normals .^ 2, 3));
normals_units = normals ./ normals_lens;

normals_units .* test_light;
test_output = sum(normals_units .* test_light, 3) .* (250 / 255);
figure();
imshow(test_output);

% And the albedo
normals_lens = normals_lens ./ max(max(normals_lens));
figure();
imshow(normals_lens);

% Test light with the albedo
test_output = sum(normals .* test_light, 3);
test_output = test_output ./ max(max(test_output));
figure();
imshow(test_output);

% Shape derivation
% Since we're working in discrete space (pixels), the task is substantially
% easier. We simply need to add up all the partial derivatives along a
% path. ("Walk" the pixel space, from top-left, tacking on partials as we 
% go.)

partials_y = -normals(:, :, 1) ./ normals(:, :, 3);
partials_x = -normals(:, :, 2) ./ normals(:, :, 3);
z = zeros(im_size(1), im_size(2));

% Now for the "integration"
for i = 1:im_size(1)
    for j = 1:im_size(2)
        if i == 1 && j == 1
            z(i, j) = 0;
        elseif i == 1
            z(i, j) = partials_x(i, j - 1) + z(i, j - 1);
        elseif j == 1
            z(i, j) = partials_y(i - 1, j) + z(i - 1, j);
        else
            z(i, j) = partials_x(i - 1, j - 1) + partials_y(i - 1, j - 1) + z(i - 1, j - 1);
        end
    end
end

figure();
surf(10:10:im_size(2), 10:10:im_size(1), z(10:10:im_size(1), 10:10:im_size(2)));

% Part 2 (grad students)

im = imread('color_photometric_stereo_1.tiff');
light_dirs = importdata('color_light_directions_1.txt');
light_colors = importdata('color_light_colors_1.txt');

im_r = im(:, :, 1);
figure()
imshow(im_r);
im_g = im(:, :, 2);
figure()
imshow(im_g);
im_b = im(:, :, 3);
figure()
imshow(im_b);

im = imread('color_photometric_stereo_2.tiff');
light_dirs = importdata('color_light_directions_2.txt');
light_colors = importdata('color_light_colors_2.txt');


im_r = im(:, :, 1);
figure()
imshow(im_r);
im_g = im(:, :, 2);
figure()
imshow(im_g);
im_b = im(:, :, 3);
figure()
imshow(im_b);

light_colors_sum = sum(light_colors, 1)
light_colors_percentages = light_colors ./ light_colors_sum

light_dir_r = sum(light_dirs .* light_colors_percentages(:, 1))
light_dir_g = sum(light_dirs .* light_colors_percentages(:, 2))
light_dir_b = sum(light_dirs .* light_colors_percentages(:, 3))
new_light_dirs = [ light_dir_r ; light_dir_g ; light_dir_b ]

normals = zeros(im_size(1), im_size(2), 3);
intensities = zeros(1, 3);

for i = 1:im_size(1)
    for j = 1:im_size(2)
        intensities(1, :) = im(i, j, :);
        normals(i, j, :) = new_light_dirs ^ -1 * intensities';
    end
end

test_light = reshape([0, 0, 1], 1, 1, 3);

normals_lens = sqrt(sum(normals .^ 2, 3));
normals_units = normals ./ normals_lens;

normals_units .* test_light;
test_output = sum(normals_units .* test_light, 3) .* (250 / 255);
figure();
imshow(test_output);

% And the albedo
normals_lens = normals_lens ./ max(max(normals_lens));
figure();
imshow(normals_lens);

% Test light with the albedo
test_output = sum(normals .* test_light, 3);
test_output = test_output ./ max(max(test_output));
figure();
imshow(test_output);

% Shape derivation
% Since we're working in discrete space (pixels), the task is substantially
% easier. We simply need to add up all the partial derivatives along a
% path. ("Walk" the pixel space, from top-left, tacking on partials as we 
% go.)

partials_x = -normals(:, :, 1) ./ normals(:, :, 3);
partials_y = -normals(:, :, 2) ./ normals(:, :, 3);
z = zeros(im_size(1), im_size(2));

% Now for the "integration"
for i = 1:im_size(1)
    for j = 1:im_size(2)
        if i == 1 && j == 1
            z(i, j) = 0;
        elseif i == 1
            z(i, j) = partials_x(i, j) + z(i, j - 1);
        elseif j == 1
            z(i, j) = partials_y(i, j) + z(i - 1, j);
        else
            z(i, j) = partials_x(i, j) + partials_y(i, j) + z(i - 1, j - 1);
        end
    end
end

figure();
surf(10:10:im_size(2), 10:10:im_size(1), z(10:10:im_size(1), 10:10:im_size(2)));



normals = zeros(im_size(1), im_size(2), 3);
intensities = zeros(1, 3);

for i = 1:im_size(1)
    for j = 1:im_size(2)
        intensities(1, :) = im(i, j, :);
        cur_mat = [new_light_dirs intensities'];
        [eig_vecs, eig_vals] = eig(cur_mat * cur_mat');
        normals(i, j, :) = eig_vecs(1, :);
    end
end

test_light = reshape([0, 0, 1], 1, 1, 3);

normals_lens = sqrt(sum(normals .^ 2, 3));
normals_units = normals ./ normals_lens;

normals_units .* test_light;
test_output = sum(normals_units .* test_light, 3) .* (250 / 255);
figure();
imshow(test_output);

% And the albedo
normals_lens = normals_lens ./ max(max(normals_lens));
figure();
imshow(normals_lens);

% Test light with the albedo
test_output = sum(normals .* test_light, 3);
test_output = test_output ./ max(max(test_output));
figure();
imshow(test_output);

% Shape derivation
% Since we're working in discrete space (pixels), the task is substantially
% easier. We simply need to add up all the partial derivatives along a
% path. ("Walk" the pixel space, from top-left, tacking on partials as we 
% go.)

partials_x = -normals(:, :, 1) ./ normals(:, :, 3);
partials_y = -normals(:, :, 2) ./ normals(:, :, 3);
z = zeros(im_size(1), im_size(2));

% Now for the "integration"
for i = 1:im_size(1)
    for j = 1:im_size(2)
        if i == 1 && j == 1
            z(i, j) = 0;
        elseif i == 1
            z(i, j) = partials_x(i, j) + z(i, j - 1);
        elseif j == 1
            z(i, j) = partials_y(i, j) + z(i - 1, j);
        else
            z(i, j) = partials_x(i, j) + partials_y(i, j) + z(i - 1, j - 1);
        end
    end
end

figure();
surf(10:10:im_size(2), 10:10:im_size(1), z(10:10:im_size(1), 10:10:im_size(2)));



normals = zeros(im_size(1), im_size(2), 3);
intensities = zeros(1, 3);

for i = 1:im_size(1)
    for j = 1:im_size(2)
        intensities(1, :) = im(i, j, :);
        cur_mat = [new_light_dirs intensities'];
        cur_mat = cur_mat - mean(cur_mat);
        [eig_vecs, eig_vals] = eig(cur_mat * cur_mat');
        normals(i, j, :) = eig_vecs(1, :);
    end
end

test_light = reshape([0, 0, 1], 1, 1, 3);

normals_lens = sqrt(sum(normals .^ 2, 3));
normals_units = normals ./ normals_lens;

normals_units .* test_light;
test_output = sum(normals_units .* test_light, 3) .* (250 / 255);
figure();
imshow(test_output);

% And the albedo
normals_lens = normals_lens ./ max(max(normals_lens));
figure();
imshow(normals_lens);

% Test light with the albedo
test_output = sum(normals .* test_light, 3);
test_output = test_output ./ max(max(test_output));
figure();
imshow(test_output);

% Shape derivation
% Since we're working in discrete space (pixels), the task is substantially
% easier. We simply need to add up all the partial derivatives along a
% path. ("Walk" the pixel space, from top-left, tacking on partials as we 
% go.)

partials_x = -normals(:, :, 1) ./ normals(:, :, 3);
partials_y = -normals(:, :, 2) ./ normals(:, :, 3);
z = zeros(im_size(1), im_size(2));

% Now for the "integration"
for i = 1:im_size(1)
    for j = 1:im_size(2)
        if i == 1 && j == 1
            z(i, j) = 0;
        elseif i == 1
            z(i, j) = partials_x(i, j) + z(i, j - 1);
        elseif j == 1
            z(i, j) = partials_y(i, j) + z(i - 1, j);
        else
            z(i, j) = partials_x(i, j) + partials_y(i, j) + z(i - 1, j - 1);
        end
    end
end

figure();
surf(10:10:im_size(2), 10:10:im_size(1), z(10:10:im_size(1), 10:10:im_size(2)));



light_colors_sum_max = max(light_colors_sum);
light_colors_sum_normed = light_colors_sum ./ light_colors_sum_max;
light_colors_sum_normed = reshape(light_colors_sum_normed, 1, 1, 3);

im = cast(im, 'double');
im_max = max(max(im))
%im = im ./ im_max;
im = im ./ light_colors_sum_normed;

normals = zeros(im_size(1), im_size(2), 3);
intensities = zeros(1, 3);

for i = 1:im_size(1)
    for j = 1:im_size(2)
        intensities(1, :) = im(i, j, :);
        normals(i, j, :) = new_light_dirs ^ -1 * intensities';
    end
end

test_light = reshape([0, 0, 1], 1, 1, 3);

normals_lens = sqrt(sum(normals .^ 2, 3));
normals_units = normals ./ normals_lens;

normals_units .* test_light;
test_output = sum(normals_units .* test_light, 3) .* (250 / 255);
figure();
imshow(test_output);

% And the albedo
normals_lens = normals_lens ./ max(max(normals_lens));
figure();
imshow(normals_lens);

% Test light with the albedo
test_output = sum(normals .* test_light, 3);
test_output = test_output ./ max(max(test_output));
figure();
imshow(test_output);

% Shape derivation
% Since we're working in discrete space (pixels), the task is substantially
% easier. We simply need to add up all the partial derivatives along a
% path. ("Walk" the pixel space, from top-left, tacking on partials as we 
% go.)

partials_x = -normals(:, :, 1) ./ normals(:, :, 3);
partials_y = -normals(:, :, 2) ./ normals(:, :, 3);
z = zeros(im_size(1), im_size(2));

% Now for the "integration"
for i = 1:im_size(1)
    for j = 1:im_size(2)
        if i == 1 && j == 1
            z(i, j) = 0;
        elseif i == 1
            z(i, j) = partials_x(i, j) + z(i, j - 1);
        elseif j == 1
            z(i, j) = partials_y(i, j) + z(i - 1, j);
        else
            z(i, j) = partials_x(i, j) + partials_y(i, j) + z(i - 1, j - 1);
        end
    end
end

figure();
surf(10:10:im_size(2), 10:10:im_size(1), z(10:10:im_size(1), 10:10:im_size(2)));



im = imread('color_photometric_stereo_1.tiff');
light_dirs = importdata('color_light_directions_1.txt');
light_colors = importdata('color_light_colors_1.txt');

light_colors_sum = sum(light_colors, 1)
light_colors_percentages = light_colors ./ light_colors_sum

light_dir_r = sum(light_dirs .* light_colors_percentages(:, 1))
light_dir_g = sum(light_dirs .* light_colors_percentages(:, 2))
light_dir_b = sum(light_dirs .* light_colors_percentages(:, 3))
new_light_dirs = [ light_dir_r ; light_dir_g ; light_dir_b ]

light_colors_sum_max = max(light_colors_sum);
light_colors_sum_normed = light_colors_sum ./ light_colors_sum_max;
light_colors_sum_normed = reshape(light_colors_sum_normed, 1, 1, 3);

im = cast(im, 'double');
im_max = max(max(im))
%im = im ./ im_max;
im = im ./ light_colors_sum_normed;

normals = zeros(im_size(1), im_size(2), 3);
intensities = zeros(1, 3);

for i = 1:im_size(1)
    for j = 1:im_size(2)
        intensities(1, :) = im(i, j, :);
        normals(i, j, :) = new_light_dirs ^ -1 * intensities';
    end
end

test_light = reshape([0, 0, 1], 1, 1, 3);

normals_lens = sqrt(sum(normals .^ 2, 3));
normals_units = normals ./ normals_lens;

normals_units .* test_light;
test_output = sum(normals_units .* test_light, 3) .* (250 / 255);
figure();
imshow(test_output);

% And the albedo
normals_lens = normals_lens ./ max(max(normals_lens));
figure();
imshow(normals_lens);

% Test light with the albedo
test_output = sum(normals .* test_light, 3);
test_output = test_output ./ max(max(test_output));
figure();
imshow(test_output);

% Shape derivation
% Since we're working in discrete space (pixels), the task is substantially
% easier. We simply need to add up all the partial derivatives along a
% path. ("Walk" the pixel space, from top-left, tacking on partials as we 
% go.)

partials_x = -normals(:, :, 1) ./ normals(:, :, 3);
partials_y = -normals(:, :, 2) ./ normals(:, :, 3);
z = zeros(im_size(1), im_size(2));

% Now for the "integration"
for i = 1:im_size(1)
    for j = 1:im_size(2)
        if i == 1 && j == 1
            z(i, j) = 0;
        elseif i == 1
            z(i, j) = partials_x(i, j) + z(i, j - 1);
        elseif j == 1
            z(i, j) = partials_y(i, j) + z(i - 1, j);
        else
            z(i, j) = partials_x(i, j) + partials_y(i, j) + z(i - 1, j - 1);
        end
    end
end

figure();
surf(10:10:im_size(2), 10:10:im_size(1), z(10:10:im_size(1), 10:10:im_size(2)));