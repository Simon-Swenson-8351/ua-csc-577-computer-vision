function [] = hw12()

close all;
format long g;
num_questions = 0;
questionSet = {'1', '2'};
flagSet = [1, 0];
questionFlag = containers.Map(questionSet,flagSet);

%% Function to calculate disparity
function[disparity_map] = calculate_disparity(left_image, right_image)

    [row1, column1] = size(left_image);
    
    disparity_map = zeros(row1, column1);
    highest_dot = 0;
    disparity_offset = 0;
    for i = 1:row1
        for j = 26:(column1-25)
           left_array = left_image(i,j-10:j+10);
           for m = j-15:j+15
               right_array = right_image(i,m-10:m+10);
               result = left_array*right_array';
               if result > highest_dot %|| (result == highest_dot && j-m < disparity_offset)
                   highest_dot = result;
                   disparity_offset = j-m;
               end
           end
           disparity_map(i,j) = abs(disparity_offset);
           highest_dot = 0;
        end
    end
    
end


if questionFlag('1')
    
    %% Read in image files
    left1 = imread('left-1.tiff');
    left1 = double(left1)/255.0;
    
    right1 = imread('right-1.tiff');
    right1 = double(right1)/255.0;
    
    left2 = imread('left-2.tiff');
    left2 = double(left2)/255.0;
    
    right2 = imread('right-2.tiff');
    right2 = double(right2)/255.0;
    
    %% Computer Disparity between the files
    disparity1 = calculate_disparity(left1, right1);
    disparity2 = calculate_disparity(left2, right2);
    
    %% Calculations for the first image
    max_disparity1 = max(max(disparity1));
    
    disparity_singular1 = reshape(disparity1, [size(disparity1,1)*size(disparity1,2),1]);
    disparity_singular1_sorted = sort(disparity_singular1, 'descend');
    disparity_top10_1 = disparity_singular1_sorted(round(1:0.1*size(disparity_singular1,1)),1);
    disparity_average_1 = sum(disparity_top10_1)/size(disparity_top10_1,1)
        
    %% Calculations for the second image
    max_disparity2 = max(max(disparity2));
    
    disparity_singular2 = reshape(disparity2, [size(disparity2,1)*size(disparity2,2),1]);
    disparity_singular2_sorted = sort(disparity_singular2, 'descend');
    disparity_top10_2 = disparity_singular2_sorted(round(1:0.1*size(disparity_singular2,1)),1);
    disparity_average_2 = sum(disparity_top10_2)/size(disparity_top10_2,1)
    
    %% Calculate distance from eye
    f = 2;
    d = 10;
    D1 = 0.025*disparity_average_1;
    D2 = 0.025*disparity_average_2;
    
    z1 = (f*d)/D1
    z2 = (f*d)/D2
    
    
    %% Print out the images
	
    disparity1 = disparity1./max_disparity1;
	disparity2 = disparity2./max_disparity2;
    figure;
    imshow(disparity1);
    figure;
    imshow(disparity2);
    
    %%
    num_questions = num_questions + 1;
end

if questionFlag('2')
    im1 = imread('FM-inside-1.jpg');
    pts1 = importdata('FM-inside-1-matches.txt');
    im2 = imread('FM-inside-2.jpg');
    pts2 = importdata('FM-inside-2-matches.txt');
    print_matching_points(im1, im2, pts1, pts2);
    
    picked_pts = randperm(size(pts1, 1), 12);
    picked_pts1 = pts1(picked_pts, :);
    picked_pts2 = pts2(picked_pts, :);
    
    fundamental_matrix = calc_fundamental_matrix(picked_pts1, picked_pts2)
    
    
    % Need homogeneous points for calculating the error
    picked_pts1 = [picked_pts1 ones(size(picked_pts1, 1), 1)];
    picked_pts2 = [picked_pts2 ones(size(picked_pts2, 1), 1)];
    picked_errors = sum(picked_pts2 .* (fundamental_matrix * picked_pts1')', 2)
    picked_rmse = calc_rmse(picked_errors, zeros(size(picked_errors)))
    
    unpicked_pts1 = [];
    unpicked_pts2 = [];
    for i = 1:size(pts1, 1)
        if isempty(find(picked_pts == i))
            unpicked_pts1 = [unpicked_pts1; pts1(i, :) 1];
            unpicked_pts2 = [unpicked_pts2; pts2(i, :) 1];
        end
    end
    unpicked_errors = sum(unpicked_pts2 .* (fundamental_matrix * unpicked_pts1')', 2)
    unpicked_rmse = calc_rmse(unpicked_errors, zeros(size(unpicked_errors)))
    
    fundamental_matrix = calc_fundamental_matrix(pts1, pts2)
    
    %line_eqns = [pts2 ones(size(pts2, 1), 1)] * fundamental_matrix;
    line_eqns = (fundamental_matrix * [pts1 ones(size(pts1, 1), 1)]')'
    for i = 1:size(line_eqns, 1)
        line_eqn = line_eqns(i, :);
        im2 = draw_line(im2, line_eqn', 0, 0, 255, 0);
    end
    print_matching_points(im1, im2, pts1, pts2);
end

end

function [] = print_matching_points(im1, im2, pts1, pts2)
    
    im1_out = im1;
    im2_out = im2;
    for i = 1:size(pts1, 1)
        hsv = [(i * 1.0 / size(pts1, 1)) 1.0 0.8];
        rgb = uint8(round(hsv2rgb(hsv) .* 255));
        rgb = permute(rgb, [1 3 2]);
        rgb = repmat(rgb, 11, 11, 1);
        coord1 = pts1(i, :);
        coord2 = pts2(i, :);
        im1_out((coord1(1, 1) - 5):(coord1(1, 1) + 5), (coord1(1, 2) - 5):(coord1(1, 2) + 5), :) = rgb;
        im2_out((coord2(1, 1) - 5):(coord2(1, 1) + 5), (coord2(1, 2) - 5):(coord2(1, 2) + 5), :) = rgb;
    end
    
    figure;
    imshow(im1_out);
    figure;
    imshow(im2_out);
end

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

function [rmse] = calc_rmse(first_points, second_points)
    rmse = sqrt(mean((first_points - second_points) .^ 2, 'all'));
end

function [U] = calc_fundamental_matrix_U(pts1, pts2)
    U = zeros(size(pts1, 1), 9);
    U(:, 1) = pts1(:, 1) .* pts2(:, 1);
    U(:, 2) = pts1(:, 2) .* pts2(:, 1);
    U(:, 3) = pts2(:, 1);
    U(:, 4) = pts1(:, 1) .* pts2(:, 2);
    U(:, 5) = pts1(:, 2) .* pts2(:, 2);
    U(:, 6) = pts2(:, 2);
    U(:, 7) = pts1(:, 1);
    U(:, 8) = pts1(:, 2);
    U(:, 9) = 1.0;
end

function [fundamental_matrix] = calc_fundamental_matrix(pts1, pts2)
    U = calc_fundamental_matrix_U(pts1, pts2);
    fundamental_matrix_vector = permute(calc_hlse_sln(U), [2 1]);
    fundamental_matrix = [fundamental_matrix_vector(:, 1:3); fundamental_matrix_vector(:, 4:6); fundamental_matrix_vector(:, 7:9)];
end