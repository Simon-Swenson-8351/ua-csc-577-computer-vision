function [] = hw11()

close all;
format long g;
num_questions = 0;
questionSet = {'1', '2', '3', '4'};
flagSet = [0, 0, 1, 0];
questionFlag = containers.Map(questionSet,flagSet);

%%
% flag_line_or_homography is true for line, false for homography
function [final_inliers, theta_value, best_inliers, root_mean_squared_error, orthogonal_error, estimated_Y_values] = ransac_line_model(data, point_num, iter_count, threshold, inlier_count)
    
    data_count = size(data,2);
    best_inliers = 0;
    
    for i = 1:iter_count
        % Find the random `point_num` points from dataset
        random_column = randperm(data_count, point_num);
        random_data = data(:, random_column);
        
        % Finding line parameters
        slope = (random_data(2,2)-random_data(2,1))/(random_data(1,2)-random_data(1,1));
        y_intercept = random_data(2,1)-(slope*random_data(1,1));
        distance = data(2,:)' - (slope*data(1,:)'+y_intercept);
        abs_distance = find((abs(distance)<=threshold));
        
        inliers = data(:,abs_distance);
        
%         figure;
%         scatter(inliers(1,:), inliers(2,:))
        
        inlier_length = length(inliers);
        if inlier_length >= inlier_count && inlier_length > best_inliers
            best_inliers = inlier_length;
            final_inliers = inliers;
            [theta_value, root_mean_squared_error, orthogonal_error, estimated_Y_values] = total_least_squares(final_inliers);
        end
    end
end

%%
% flag_line_or_homography is true for line, false for homography
function [final_inliers_X, final_inliers_Y] = ransac_homography_model(data_X, data_Y, point_num, iter_count, threshold, inlier_count)
    
    data_count = size(data_X,1);
    best_inliers = 0;
    
    for i = 1:iter_count
        % Find the random `point_num` points from dataset
        random_row = randperm(data_count, point_num);
        random_data_X = data_X(random_row, :);
        random_data_Y = data_Y(random_row, :);
        
        % Finding line parameters
        homography = calc_homography(random_data_X, random_data_Y);
        data_predictions = (homography * data_X')';
        data_predictions = data_predictions ./ data_predictions(:, 3);
        distance =  sqrt(sum((data_Y - data_predictions) .^ 2, 2));
        abs_distance = find(distance<=threshold);
        
        inliers_X = data_X(abs_distance, :);
        inliers_Y = data_Y(abs_distance, :);
        
%         figure;
%         scatter(inliers(1,:), inliers(2,:))
        
        inlier_length = length(inliers_X);
        if inlier_length >= inlier_count && inlier_length > best_inliers
            best_inliers = inlier_length;
            final_inliers_X = inliers_X;
            final_inliers_Y = inliers_Y;
        end
    end
end

%%
function[homography] = calc_homography(points, matching_points)
    points_size = size(points);
    homogeneous_matrix = zeros(points_size(1) * 3, 9);
    homogeneous_matrix_size = size(homogeneous_matrix);
    homogeneous_matrix(1:3:homogeneous_matrix_size(1), 4:6) = -points;
    homogeneous_matrix(1:3:homogeneous_matrix_size(1), 7:9) = matching_points(:, 2) .* points;
    homogeneous_matrix(2:3:homogeneous_matrix_size(1), 1:3) = points;
    homogeneous_matrix(2:3:homogeneous_matrix_size(1), 7:9) = -matching_points(:, 1) .* points;
    homogeneous_matrix(3:3:homogeneous_matrix_size(1), 1:3) = -matching_points(:, 2) .* points;
    homogeneous_matrix(3:3:homogeneous_matrix_size(1), 4:6) = matching_points(:, 1) .* points;
    homography_vector = calc_hlse_sln(homogeneous_matrix);
    homography = [homography_vector(1:3)'; homography_vector(4:6)'; homography_vector(7:9)'];
    
end

if questionFlag('1')
    
    %% Reading in the text file
    line_dataId = fopen("line_data_2.txt", "r");
    formatSpec = "%f %f";
    sizeA = [2 Inf];
    line_data = fscanf(line_dataId, formatSpec, sizeA);

    %% Calling the RANSAC function
    [final_inliers, theta, max_inliers, rmse, error, estimated_y_values]  = ransac_line_model(line_data, 2, 75, 0.2, 75);

    fprintf('num_points: %d, rmse: %.6f, orth_err: %.6f \n',...
        max_inliers, rmse, error);
    
    figure;
    scatter(line_data(1,:),line_data(2,:))
    hold on;
    p2 = plot(final_inliers(1,:)', estimated_y_values, 'r');

    %% Increment the question count
    num_questions = num_questions + 1;
    
end

%% Function to refit the line using the Homogeneous Least Squares Method
function [theta_value, root_mean_squared_error, orthogonal_error, estimatedMatrix2] = total_least_squares(inliers_final)
	
	x = inliers_final(1,:)';
    y = inliers_final(2,:)';
   
	meanX = mean(x);
    meanY = mean(y);
    
    U = [x-meanX y-meanY];
    
    Y = U'*U;
    
    % Finding the eigenvalues and eigenvectors of Y
    [V, D] = eig(Y);
    
    % Get the eigenvalue matrix from the diagonal matrix D and sort it
    [d, ind] = sort(diag(D));
    
    % Sort eigenvector matrix V with the same indices as used for D
    sortedV = V(:,ind);
    
    % Our solution matrix is the eigenvector corresponding to the smallest
    % eigenvalue
    solnMatrix2 = sortedV(:,1);
    a = solnMatrix2(1);
    b = solnMatrix2(2);
    
    % Calculate value of d
    d = a*meanX + b*meanY;
    
    % Estimated y-value matrix
    estimatedMatrix2 = (d - (a*x))/b;
    
    % Calculate the value of theta
    theta_value = [d/b; -a/b];
    
    % Calculating the RMSE
    m = size(inliers_final,2);
    [y_size,~] = size(y);
    predicted_y = [ones([m,1]) x] * theta_value;
    root_mean_squared_error = sqrt(sum((y - predicted_y).^2)/y_size);
    orthogonal_error = abs(U * V(:,1));
	orthogonal_error = mean(orthogonal_error.^2);
    
end

%% Start for Part B
if questionFlag('2')
    avg_rmse = zeros(3, 1);
    for i = 1:10
        % print iteration
        for num_pts = 4:6
            % print cur # pts
            
            points_initial = [rand(num_pts, 2) ones(num_pts, 1)];
            points_final = [rand(num_pts, 2) ones(num_pts, 1)];
            homography = calc_homography(points_initial, points_final);
            points_transformed = (homography * points_initial')';
            points_transformed = points_transformed ./ points_transformed(:, 3);
            rmse = calc_rmse(points_transformed, points_final);
            avg_rmse(num_pts - 3) = avg_rmse(num_pts - 3) + rmse;
        end
    end
    avg_rmse = avg_rmse ./ 10
    
    part_b_b(1:8);
    part_b_b(1:2:8);
end

%%
function[] = part_b_b(point_indices)
    slide1_pts = importdata('slide1_manual_matches.txt');
    slide1_pts = [slide1_pts ones(8, 1)];
    slide1_selected_pts = slide1_pts(point_indices, :);
    frame1_pts = importdata('frame1_manual_matches.txt');
    frame1_pts = [frame1_pts ones(8, 1)];
    frame1_selected_pts = frame1_pts(point_indices, :);
    slide2_pts = importdata('slide2_manual_matches.txt');
    slide2_pts = [slide2_pts ones(8, 1)];
    slide2_selected_pts = slide2_pts(point_indices, :);
    frame2_pts = importdata('frame2_manual_matches.txt');
    frame2_pts = [frame2_pts ones(8, 1)];
    frame2_selected_pts = frame2_pts(point_indices, :);
    slide3_pts = importdata('slide3_manual_matches.txt');
    slide3_pts = [slide3_pts ones(8, 1)];
    slide3_selected_pts = slide3_pts(point_indices, :);
    frame3_pts = importdata('frame3_manual_matches.txt');
    frame3_pts = [frame3_pts ones(8, 1)];
    frame3_selected_pts = frame3_pts(point_indices, :);
    
    homography1 = calc_homography(slide1_selected_pts, frame1_selected_pts);
    homography2 = calc_homography(slide2_selected_pts, frame2_selected_pts);
    homography3 = calc_homography(slide3_selected_pts, frame3_selected_pts);
    
    frame1_calc_pts = (homography1 * slide1_pts')';
    frame1_calc_pts = frame1_calc_pts ./ frame1_calc_pts(:, 3);
    
    frame2_calc_pts = (homography2 * slide2_pts')';
    frame2_calc_pts = frame2_calc_pts ./ frame2_calc_pts(:, 3);
    
    frame3_calc_pts = (homography3 * slide3_pts')';
    frame3_calc_pts = frame3_calc_pts ./ frame3_calc_pts(:, 3);
    
    
    % Read in the frame and slide images over which to draw the results
    frame1_kp = imread('frame1.jpg');
    slide1_kp = imread('slide1.tiff');
    slide1_kp = slide1_kp(:, :, 1:3);
    
	frame2_kp = imread('frame2.jpg');
    slide2_kp = imread('slide2.tiff');
    slide2_kp = slide2_kp(:, :, 1:3);
    
	frame3_kp = imread('frame3.jpg');
    slide3_kp = imread('slide3.tiff');
    slide3_kp = cat(3, slide3_kp, slide3_kp, slide3_kp);
    
    % Stitch the frame and slide images
    comparison_img_1 = zeros(270, 720, 3, 'uint8');
    comparison_img_1(:, 1:360, :) = slide1_kp;
    comparison_img_1(16:255, 361:720, :) = frame1_kp;
    
    comparison_img_2 = zeros(264, 704, 3, 'uint8');
    comparison_img_2(:, 1:352, :) = slide2_kp;
    comparison_img_2(16:255, 353:704, :) = frame2_kp;
    
	comparison_img_3 = zeros(264, 704, 3, 'uint8');
    comparison_img_3(:, 1:352, :) = slide3_kp;
    comparison_img_3(16:255, 353:704, :) = frame3_kp;
    
    comparison_img_1_new = comparison_img_1;
    for i = 1:8
        fs1p1 = [(15 + frame1_calc_pts(i, 1)) (360 + frame1_calc_pts(i, 2))];
        fs1p2 = [slide1_pts(i, 1) slide1_pts(i, 2)];
        fs1p3 = [(15 + frame1_pts(i, 1)) (360 + frame1_pts(i, 2))];
        comparison_img_1_new = draw_line(comparison_img_1_new, fs1p1, fs1p2, 0,   0, 255,   0);
        comparison_img_1_new = draw_line(comparison_img_1_new, fs1p3, fs1p3, 1, 255, 255,   0);
        comparison_img_1_new = draw_line(comparison_img_1_new, fs1p1, fs1p1, 1, 255,   0,   0);
        comparison_img_1_new = draw_line(comparison_img_1_new, fs1p2, fs1p2, 1, 255,   0,   0);
    end
    figure();
    imshow(comparison_img_1_new);
    
    comparison_img_2_new = comparison_img_2;
    for i = 1:8
        fs1p1 = [(12 + frame2_calc_pts(i, 1)) (352 + frame2_calc_pts(i, 2))];
        fs1p2 = [slide2_pts(i, 1) slide2_pts(i, 2)];
        fs1p3 = [(12 + frame2_pts(i, 1)) (352 + frame2_pts(i, 2))];
        comparison_img_2_new = draw_line(comparison_img_2_new, fs1p1, fs1p2, 0, 0, 255, 0);
        comparison_img_2_new = draw_line(comparison_img_2_new, fs1p3, fs1p3, 1, 255, 255,   0);
        comparison_img_2_new = draw_line(comparison_img_2_new, fs1p1, fs1p1, 1, 255, 0, 0);
        comparison_img_2_new = draw_line(comparison_img_2_new, fs1p2, fs1p2, 1, 255, 0, 0);
    end
    figure();
    imshow(comparison_img_2_new);
    
    comparison_img_3_new = comparison_img_3;
    for i = 1:8
        fs1p1 = [(12 + frame3_calc_pts(i, 1)) (352 + frame3_calc_pts(i, 2))];
        fs1p2 = [slide3_pts(i, 1) slide3_pts(i, 2)];
        comparison_img_3_new = draw_line(comparison_img_3_new, fs1p1, fs1p2, 0, 0, 255, 0);
        comparison_img_3_new = draw_line(comparison_img_3_new, fs1p1, fs1p1, 1, 255, 0, 0);
        comparison_img_3_new = draw_line(comparison_img_3_new, fs1p2, fs1p2, 1, 255, 0, 0);
    end
    figure();
    imshow(comparison_img_3_new);
end

%%
if questionFlag('3')
    % Best method
    match_parameters.pruning_method = 'first_second_ratio';
    match_parameters.first_second_ratio = 0.85;
    match_parameters.distance_metric = 'euclidean';
    part_c_single_pair('slide1.tiff', 'frame1.jpg', 'slide1.sift', 'frame1.sift', match_parameters, 14, 5, 30);
    part_c_single_pair('slide2.tiff', 'frame2.jpg', 'slide2.sift', 'frame2.sift', match_parameters, 100, 10, 10);
    part_c_single_pair('slide3.tiff', 'frame3.jpg', 'slide3.sift', 'frame3.sift', match_parameters, 20, 2, 50);
   
    % Worst(?) method
    match_parameters.pruning_method = 'top_percentile';
    match_parameters.top_percentile = 0.3;
    match_parameters.distance_metric = 'chi_sq';
    part_c_single_pair('slide1.tiff', 'frame1.jpg', 'slide1.sift', 'frame1.sift', match_parameters, 50, 5, 40);
    part_c_single_pair('slide2.tiff', 'frame2.jpg', 'slide2.sift', 'frame2.sift', match_parameters, 100, 10, 10);
    part_c_single_pair('slide3.tiff', 'frame3.jpg', 'slide3.sift', 'frame3.sift', match_parameters, 40, 2, 50);
end

%%
function [matching_indices] = find_matches(slide_features, frame_features, match_parameters)
    slide_size = size(slide_features);
    frame_size = size(frame_features);
    
    frame_3d_mat = permute(frame_features(:, 5:132), [1 3 2]);
    frame_3d_mat = repmat(frame_3d_mat, 1, slide_size(1), 1);
    
    slide_3d_mat = permute(slide_features(:, 5:132), [3 1 2]);
    slide_3d_mat = repmat(slide_3d_mat, frame_size(1), 1, 1);
    
    frame_distances = [];
    if strcmp(match_parameters.distance_metric, 'euclidean')
        frame_distances = sqrt(sum((frame_3d_mat- slide_3d_mat) .^ 2, 3));
    elseif strcmp(match_parameters.distance_metric, 'chi_sq')
        frame_3d_mat = frame_3d_mat + 0.0001;
        slide_3d_mat = slide_3d_mat + 0.0001;
        frame_distances = 0.5 .* sum((frame_3d_mat - slide_3d_mat) .^ 2 ./ (frame_3d_mat + slide_3d_mat), 3);
    else
        % Here's where I'd throw an exception
    end
    [distance_mins, distance_slide_entries] = min(frame_distances, [], 2);
    
    pruned_indices = [];
    if strcmp(match_parameters.pruning_method, 'first_second_ratio')
        frame_second_distances = frame_distances;
        for i = 1:frame_size(1)
            frame_second_distances(i, distance_slide_entries(i)) = Inf;
        end

        second_distance_mins = min(frame_second_distances, [], 2);

        ratios = distance_mins ./ second_distance_mins;
        pruned_indices = find(ratios <= match_parameters.first_second_ratio);
    elseif strcmp(match_parameters.pruning_method, 'top_percentile')
        [sorted_distances, sorted_distances_index_map] = sort(distance_mins);
        distance_mins_size = size(distance_mins);
        pruned_indices = sorted_distances_index_map(1:round(distance_mins_size * match_parameters.top_percentile));
    else
        % Here's where I'd throw an exception
    end
    pruned_indices_size = size(pruned_indices);
    
    matching_indices = [];
    for i = 1:pruned_indices_size(1)
        cur_slide_coord = slide_features(distance_slide_entries(pruned_indices(i)), [2 1]);
        cur_frame_coord = frame_features(pruned_indices(i), [2 1]);
        cur_entry = [pruned_indices(i) distance_slide_entries(pruned_indices(i)) cur_frame_coord cur_slide_coord];
        if isempty(matching_indices)
            matching_indices = cur_entry
        else
            found_match = find( ...
                max([matching_indices(:, 2) == distance_slide_entries(pruned_indices(i))...
                     min(matching_indices(:, 3:4) == cur_frame_coord, 2)...
                     min(matching_indices(:, 5:6) == cur_slide_coord, 2)]), 2);
            if(~isempty(found_match) && (distance_mins(found_match(1)) > distance_mins(pruned_indices(i))))
                matching_indices(found_match(1), :) = cur_entry;
            else
                matching_indices = [matching_indices; cur_entry];
            end
        end
    end
end

%%
function [] = part_c_single_pair(slide_filename, frame_filename, slide_keypoint_filename, frame_keypoint_filename, match_parameters, ransac_num_iters, ransac_threshold, ransac_num_inliers)
    slide_im = imread(slide_filename);
    frame_im = imread(frame_filename);
    
    slide_im_size = size(slide_im);
    slide_im_size_size = size(slide_im_size);
    frame_im_size = size(frame_im);
    if(slide_im_size_size(2) == 2)
        slide_im = cat(3, slide_im, slide_im, slide_im);
    elseif(slide_im_size(3) > 3)
        slide_im = slide_im(:, :, 1:3);
    end
    
    offset_x = slide_im_size(2);
    offset_y = round((slide_im_size(1) - frame_im_size(1)) / 2);
    comparison_img_1 = zeros(slide_im_size(1), slide_im_size(2) + frame_im_size(2), 3, 'uint8');
    
    comparison_img_1(:, 1:slide_im_size(2), :) = slide_im;
    comparison_img_1(offset_y:(frame_im_size(1) + offset_y - 1), offset_x:(frame_im_size(2) + offset_x - 1), :) = frame_im;
    
        % Read in the SIFT files containing the keypoint data
    slide_data = importdata(slide_keypoint_filename);
    frame_data = importdata(frame_keypoint_filename);
    
    % Get the size of each keypoint descriptor file
	slide_data_size = size(slide_data);
    frame_data_size = size(frame_data);
    
    pair1_matches = find_matches(slide_data, frame_data, match_parameters)
    pair1_matches_size = size(pair1_matches);
    
    slide1Matches = [ slide_data(pair1_matches(:, 2), [2 1]) ones(pair1_matches_size(1), 1)];
    frame1Matches = [ frame_data(pair1_matches(:, 1), [2 1]) ones(pair1_matches_size(1), 1)];
    
    comparison_img_1_new = comparison_img_1;
    for i = 1:pair1_matches_size(1)
        fs1p1 = [(offset_y + frame1Matches(i, 1)) (offset_x + frame1Matches(i, 2))];
        fs1p2 = [slide1Matches(i, 1) slide1Matches(i, 2)];
        comparison_img_1_new = draw_line(comparison_img_1_new, fs1p1, fs1p2, 0,   0, 255,   0);
        comparison_img_1_new = draw_line(comparison_img_1_new, fs1p1, fs1p1, 1, 255,   0,   0);
        comparison_img_1_new = draw_line(comparison_img_1_new, fs1p2, fs1p2, 1, 255,   0,   0);
    end
    figure();
    imshow(comparison_img_1_new);
    
    [slide1_inliers, frame1_inliers] = ransac_homography_model(slide1Matches, frame1Matches, 4, ransac_num_iters, ransac_threshold, ransac_num_inliers);
    slide1_inliers_size = size(slide1_inliers);
    
    % Also display all found inliers
    comparison_img_1_new = comparison_img_1;
    for i = 1:slide1_inliers_size(1)
        fs1p1 = [(offset_y + frame1_inliers(i, 1)) (offset_x + frame1_inliers(i, 2))];
        fs1p2 = [slide1_inliers(i, 1) slide1_inliers(i, 2)];
        comparison_img_1_new = draw_line(comparison_img_1_new, fs1p1, fs1p2, 0,   0, 255,   0);
        comparison_img_1_new = draw_line(comparison_img_1_new, fs1p1, fs1p1, 1, 255,   0,   0);
        comparison_img_1_new = draw_line(comparison_img_1_new, fs1p2, fs1p2, 1, 255,   0,   0);
    end
    figure();
    imshow(comparison_img_1_new);
    
    homography1 = calc_homography(slide1_inliers, frame1_inliers);
    
    slide1_predictions = (homography1 * [slide_data(:, [2 1]) ones(slide_data_size(1), 1)]')';
    slide1_predictions = slide1_predictions ./ slide1_predictions(:, 3);
    
    comparison_img_1_new = comparison_img_1;
    for i = 1:20:slide_data_size(1)
        fs1p1 = [(offset_y + slide1_predictions(i, 1)) (offset_x + slide1_predictions(i, 2))];
        fs1p2 = [slide_data(i, 2) slide_data(i, 1)];
        comparison_img_1_new = draw_line(comparison_img_1_new, fs1p1, fs1p2, 0,   0, 255,   0);
        comparison_img_1_new = draw_line(comparison_img_1_new, fs1p1, fs1p1, 1, 255,   0,   0);
        comparison_img_1_new = draw_line(comparison_img_1_new, fs1p2, fs1p2, 1, 255,   0,   0);
    end
    figure();
    imshow(comparison_img_1_new);
end

%%
if questionFlag('4')
    % Many points with little noise
    do_question_4(100, 5, [0.01 0.01 0.01 0.01 0.01], [250 250 250 250 250], [4 5 6]);
    % Fewer points with some more noise
    do_question_4(50, 7, [0.02 0.02 0.02 0.02 0.02 0.02 0.02], [50 50 50 50 50 50 50], [4 7 10]);
    % Even fewer points
    do_question_4(50, 3, [0.02 0.02 0.02], [10 10 10], [2 3 4]);
    % More noisy lines
    do_question_4(50, 5, [0.05 0.05 0.05 0.05 0.05], [50 50 50 50 50], [3 5 7]);
    % Lines with different noise values
    do_question_4(50, 5, [0.02 0.01 0.05 0.1 0.001], [50 50 50 50 50], [4 5 6]);
    % Different sample numbers and much more noise
    do_question_4(250, 5, [0.02 0.02 0.02 0.02 0.02], [10 50 25 100 50], [3 5 7]);
end

end

%%
function [] = do_question_4(num_noise_points, num_lines, std_devs_for_lines, samples_per_line, ks_to_try)
    [labeled_points, lines] = generate_all_line_points(num_noise_points, num_lines, std_devs_for_lines, samples_per_line)
    unlabeled_points = labeled_points(:, 1:2);
    figure();
    hold on
    scatter(labeled_points(:, 1), labeled_points(:, 2), [], labeled_points(:, 3));
    for i = 1:num_lines
        scatter(labeled_points(:, 1), labeled_points(:, 2), [], labeled_points(:, 3));
        plot(0.0:0.05:1.0, lines(i, 1) * (0.0:0.05:1.0) + lines(i, 2));
    end
    % Ensure the view is only for the window (0, 1), (0, 1) on both axes.
    ax = gca;
    ax.XLim = [0.0 1.0];
    ax.YLim = [0.0 1.0];
    
    % Plot the unlabeled data, as the k-means alg would receive it.
    figure();
    hold on
    scatter(unlabeled_points(:, 1), unlabeled_points(:, 2));
    
    ks_to_try_size = size(ks_to_try);
    for i = 1:ks_to_try_size(2)
        [line_parameters, assigned_points] = k_means_lines(ks_to_try(i), unlabeled_points);
        figure();
        hold on
        for j = 1:ks_to_try(i)
            scatter(assigned_points(:, 1), assigned_points(:, 2), [], assigned_points(:, 3));
            plot(0.0:0.05:1.0, line_parameters(j, 1) * (0.0:0.05:1.0) + line_parameters(j, 2));
        end
        % Ensure the view is only for the window (0, 1), (0, 1) on both axes.
        ax = gca;
        ax.XLim = [0.0 1.0];
        ax.YLim = [0.0 1.0];
    end
end

%%
% Generates many labeled points in the window (0, 1) in both dimensions (x, y).
% A label of 0 indicates random noise. Otherwise, the number matches the index 
% into line_parameters, the line that generated that point.
% num_noise_points - scalar natural number
% num_lines - scalar natural number
% std_devs_for_lines - a vector of length equal to num_lines, a std. dev. for each.
% samples_per_line - a vector of length equal to num_lines, the number of samples 
%   to generate per line.
function [labeled_points, line_parameters] = generate_all_line_points(num_noise_points, num_lines, std_devs_for_lines, samples_per_line)
    labeled_points = rand(num_noise_points, 2);
    % A label of 0 indicates 
    labeled_points = [labeled_points zeros(num_noise_points, 1)];
    for i = 1:num_lines
        [new_points, new_line_parameters] = generate_line_points(std_devs_for_lines(i), samples_per_line(i));
        if i == 1
            line_parameters = new_line_parameters;
        else
            line_parameters = [line_parameters; new_line_parameters];
        end
        new_points = [new_points repmat([i], samples_per_line(i), 1)];
        labeled_points = [labeled_points; new_points];
    end
end

%%
function [points, line_parameters] = generate_line_points(std_dev, num_samples)
    line_points = rand(2, 2);
    % rise / run
    m = (line_points(2, 2) - line_points(1, 2)) / (line_points(2, 1) - line_points(1, 1));
    % y = mx + b -> b = y - mx
    b = line_points(1, 2) - m * line_points(1, 1);
    line_parameters = [m b];
    points = [];
    for i = 1:num_samples
        if i == 1
            points = generate_line_point(m, b, std_dev);
        else
            points = [points; generate_line_point(m, b, std_dev)];
        end
    end
end

%%
% Generates a new point from the line and given noise parameter (std_dev) within 
% the x range of (0, 1).
function [new_point] = generate_line_point(m, b, std_dev)
    % a parallel vector can be computed using [1 m]
    % a perpendicular vector should be [-m 1]
    perpendicular_unit_vector = [-m 1];
    % Make it into a unit
    perpendicular_unit_vector = perpendicular_unit_vector ./ sqrt(sum(perpendicular_unit_vector .^ 2));

    % Unless the slope is really large (almost vertical line), we can just 
    % randomly sample from x between (0, 1), check if it's in the bounds in both 
    % x and y directions, then if it's good, return. If not, try again.
    new_point = [-1.0 -1.0]
    % Loop until both values are within the range
    % Recall max == 1 is like logical or
    while max([max(new_point < 0.0) max(new_point > 1.0)]) == 1
        x = rand(1, 1);
        new_point = [x (m * x + b)];
        % Add the noise
        new_point = new_point + perpendicular_unit_vector * normrnd(0, std_dev);
    end
end

%%
function [line_parameters, assigned_points] = k_means_lines(k, unlabeled_points)
    % Either start with random line parameters or random point assignments.
    % We start with random line parameters
    % First dimension - line index, second dimension - point index (two
    % points generated per each line), third dimension - x or y for each
    % point (x = 1, y = 2).
    line_points = rand(k, 2, 2);
    line_ms = (line_points(:, 2, 2) - line_points(:, 1, 2)) ./ (line_points(:, 2, 1) - line_points(:, 1, 1))
    line_bs = line_points(:, 1, 2) - line_ms .* line_points(:, 1, 1);
    line_parameters = [line_ms line_bs];
    prev_err = Inf;
    cur_err = Inf;
    while 1
        assigned_points = chicken(line_parameters, unlabeled_points);
        line_parameters = egg(assigned_points, k);
        prev_err = cur_err;
        cur_err = calc_line_err(assigned_points, line_parameters)
        
        if prev_err == cur_err
            break;
        end
    end
end

%%
function [assigned_points] = chicken(line_parameters, unassigned_points)
    % Technique for this one is to make a 3-d matrix of distances between
    % each point and each line (one point per row, one line per column).
    % Then collect the maxes for each row, and use those indices as the
    % assignments.
    line_parameters_size = size(line_parameters);
    unassigned_points_size = size(unassigned_points);
    unassigned_points_mat = repmat(permute(unassigned_points, [1 3 2]), 1, line_parameters_size(1), 1);
    projected_points_mat = zeros(unassigned_points_size(1), line_parameters_size(1), 2);
    for i = 1:line_parameters_size(1)
        cur_projected_pts = project_points_onto_line(unassigned_points, line_parameters(i, 1), line_parameters(i, 2));
        cur_projected_pts = permute(cur_projected_pts, [1 3 2]);
        projected_points_mat(:, i, :) = cur_projected_pts;
    end
    distances = sqrt(sum((unassigned_points_mat - projected_points_mat) .^ 2, 3));
    [best_distances, best_distances_indices] = min(distances, [], 2);
    assigned_points = [unassigned_points best_distances_indices];
end

%%
function [line_parameters] = egg(assigned_points, k)
    line_parameters = zeros(k, 2);
    for i = 1:k
        cur_pts = assigned_points(find(assigned_points(:, 3) == i), 1:2);
        % Since our error metric is perpendicular distance, we need to use
        % non-homogeneous least squared error method.
        mean(cur_pts, 1)
        a_b_parameters = calc_hlse_sln(cur_pts - mean(cur_pts, 1));
        line_parameters(i, :) = homogeneous_to_non_homogeneous_model(a_b_parameters, mean(cur_pts(:, 1)), mean(cur_pts(:, 2)));
    end
end

%%
function [line_err] = calc_line_err(assigned_points, line_parameters)
    line_err = 0.0;
    line_parameters_size = size(line_parameters);
    for i = 1:line_parameters_size(1)
        cur_pts = assigned_points(find(assigned_points(:, 3) == i), 1:2);
        if isempty(cur_pts)
            continue;
        end
        cur_pts_projections = project_points_onto_line(cur_pts, line_parameters(1), line_parameters(2));
        line_err = line_err + calc_rmse(cur_pts, cur_pts_projections);
    end
end

%%
function [rmse] = calc_rmse(first_points, second_points)
    rmse = sqrt(mean((first_points - second_points) .^ 2, 'all'));
end

%%
% Sadly, since lines are affine, this projection will need to be in
% homogeneous space. This means the projection will be 3x3.
function [projected_points] = project_points_onto_line(points, m, b)
    % The strategy is to compute the perpendicular line that goes through a
    % given point in points. Then compute the intersection between that
    % line and the original line.
    points_size = size(points);
    m_perpendicular = 1 / -m;
    projected_points = zeros(points_size(1), 2);
    for i = 1:points_size(1)
        b_perpendicular = points(i, 2) - m_perpendicular * points(i, 1);
        x_line_intercept = (b - b_perpendicular) / (m_perpendicular - m);
        projected_points(i, :) = [x_line_intercept (m * x_line_intercept + b)];
    end
end

%%
function [solnMatrix2] = calc_hlse_sln(U)
    
    Y = U'*U;
    
    % Finding the eigenvalues and eigenvectors of Y
    [V, D] = eig(Y);
    
    % Get the eigenvalue matrix from the diagonal matrix D and sort it
    [d, ind] = sort(diag(D));
    
    % Sort eigenvector matrix V with the same indices as used for D
    sortedV = V(:,ind);
    
    % Our solution matrix is the eigenvector corresponding to the smallest
    % eigenvalue
    solnMatrix2 = sortedV(:,1);
end

%%
function [slope_int] = homogeneous_to_non_homogeneous_model(a_b, X_mean, y_mean)
    m = -a_b(1)/a_b(2);
    b = y_mean - m * X_mean;
    slope_int = [m b];
end