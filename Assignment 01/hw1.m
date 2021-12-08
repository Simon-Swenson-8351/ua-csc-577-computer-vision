function [num_questions] = hw1(infile)
%HW1 Performs all commands needed for HW1.
%   The return value is the number of completed homework questions. First, 
%   HW1 reads in an image, displays it in a figure, then saves the
%   image with the extension '.out.jpg'. Then, it displays statistics (min, 
%   max) for each channel in the image. Then, the image is converted to a
%   grayscale. This grayscale image is displayed as a figure. After that,
%   each separate channel is taken as a grayscale image and displayed. To
%   further experiment with channels, a new 3-channel image is produced
%   which permutes the channels of the original image. To experiment with
%   indexing, the grayscale image from earlier is modified so that each
%   fifth pixel along the rows and columns are turned white. This is also
%   displayed in a figure. In addition to the statistics gathered earlier,
%   a histogram is then generated for each channel and shown in a figure.
%   After that, to experiment with function plotting, the sin and cosine
%   function are plotted from the domain -pi to pi. To experiment with
%   matrix operations, an arbitrary matrix equation is solved by using
%   inv, linsolve, and '\'. The topic of modifying certain pixel values in
%   the grayscale image is then returned to, and a one-liner is used to
%   perform a similar operation to the lattice operation that the nested 
%   loop performed earlier. A similar method is then used to set all pixel
%   values over a certain threshold to black. Finally, the function
%   explores PCA and transforms a set of data points based on their PCA.
    num_questions = 0;
    img = imread(infile);
    img_fig = figure;
    imshow(img);
    imwrite(img, strcat(infile, '.out.jpg'));
    
    % #6
    num_questions = num_questions + 1;
    
    % #7
    num_questions = num_questions + 1;
    whos img;
    [num_rows, num_cols, num_channels] = size(img)
    img_red = img(:, :, 1);
    % 0
    strcat('min(min(img_red)) = ', int2str(min(min(img_red))))
    % 251
    strcat('max(max(img_red)) = ', int2str(max(max(img_red))))
    img_green = img(:, :, 2);
    % 0
    strcat('min(min(img_green)) = ', int2str(min(min(img_green))))
    % 248
    strcat('max(max(img_green)) = ', int2str(max(max(img_green))))
    img_blue = img(:, :, 3);
    % 0
    strcat('min(min(img_blue)) = ', int2str(min(min(img_blue))))
    % 253
    strcat('max(max(img_blue)) = ', int2str(max(max(img_blue))))
    img_gray = rgb2gray(img);
    'size(img_gray) = '
    size(img_gray)
    img_gray_fig = figure;
    imshow(img_gray);
    
    % #8
    num_questions = num_questions + 1;
    img_red_fig = figure;
    imshow(img_red);
    img_green_fig = figure;
    imshow(img_green);
    img_blue_fig = figure;
    imshow(img_blue);
    
    img_permuted = zeros(size(img), 'uint8');
    img_permuted(:, :, 1) = img_green;
    img_permuted(:, :, 2) = img_blue;
    img_permuted(:, :, 3) = img_red;
    img_permuted_fig = figure;
    imshow(img_permuted);
    
    % #9
    num_questions = num_questions + 1;
    img_gray_scaled = cast(img_gray, 'double') ./ 255.0;
    img_gray_size = size(img_gray_scaled);
    for y = 5:5:img_gray_size(1)
        for x = 5:5:img_gray_size(2)
            img_gray_scaled(y, x) = 1.0;
        end
    end
    
    img_gray_lattice_imagesc_fig = figure;
    imagesc(img_gray_scaled);
    img_gray_lattice_imgshow_fig = figure;
    imshow(img_gray_scaled);
    
    % #10
    num_questions = num_questions + 1;
    img_red_1d_fig = figure;
    histogram(img_red(:), 20);
    img_green_1d_fig = figure;
    histogram(img_green(:), 20);
    img_blue_1d_fig = figure;
    histogram(img_blue(:), 20);
    
    % #11
    num_questions = num_questions + 1;
    sin_fig = figure;
    X = linspace(-pi, pi);
    plot(X, sin(X), 'b');
    hold on;
    plot(X, cos(X), 'r');
    
    % #12
    num_questions = num_questions + 1;
    mat_12 = [3 4 1; 2 -1 2; 1 1 -1];
    vec_12 = [9; 8; 0];
    mat_12_inv = inv(mat_12);
    ans_12_1 = mat_12_inv * vec_12
    mat_12 * ans_12_1
    
    ans_12_2 = linsolve(mat_12, vec_12)
    ans_12_3 = mat_12 \ vec_12
    diff_12 = ans_12_2 - ans_12_1
    
    % #15
    num_questions = num_questions + 1;
    img_gray_15 = cast(img_gray, 'double') ./ 255;
    img_gray_15_size = size(img_gray_15)
    img_gray_15(5:5:(img_gray_15_size(1)), 5:5:(img_gray_15_size(2))) = 0.0;
    img_gray_15_fig_1 = figure;
    imshow(img_gray_15);
    
    img_gray_15(find(img_gray_15 > 0.5)) = 0.0;
    img_gray_15_fig_2 = figure;
    imshow(img_gray_15);
    
    % #16
    num_questions = num_questions + 1;
    pca_mat = importdata('pca.txt')
    pca_mat_size = size(pca_mat);
    pca_fig = figure;
    scatter(pca_mat(:, 1), pca_mat(:, 2), 3, 'filled');
    axis equal;
    covar = cov(pca_mat)
    
    % #17
    num_questions = num_questions + 1;
    pca_mean = sum(pca_mat, 1) / pca_mat_size(1)
    pca_mat_mean = pca_mat - pca_mean
    covar_mean = cov(pca_mat_mean)
    [pca_cov_eigvecs, pca_cov_eigvals] = eig(cov(pca_mat_mean))
    pca_cov_eigvecs(:, 1)' * pca_cov_eigvecs(:, 2)
    % Essentially, we have a new basis defined by our eigenvectors, and we
    % just need to multiply each (x, y) coordinate against that new basis
    % to get that point in terms of that basis. This requires some
    % transpose shenanagans.
    pca_performed = pca_mat * pca_cov_eigvecs(:, [2 1]);
    covar_pca = cov(pca_performed)
    pca_fig_2 = figure;
    scatter(pca_performed(:, 1), pca_performed(:, 2), 3, 'filled');
    axis equal;
    cov_sum_before = covar(1, 1) + covar(2, 2)
    cov_sum_after = covar_pca(1, 1) + covar_pca(2, 2)
end