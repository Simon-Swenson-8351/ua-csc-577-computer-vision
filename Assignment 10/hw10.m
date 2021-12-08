function [] = hw10()

close all;
format long g;
num_questions = 0;
questionSet = {'1', '2', '3', '4', '5', '6', '7'};
flagSet = [1, 1, 1, 1, 0, 0, 0];
questionFlag = containers.Map(questionSet,flagSet);

if questionFlag('1')
	
    % Read in the SIFT files containing the keypoint data
    frame1Data = importdata("frame1.sift");
    frame2Data = importdata("frame2.sift");
    frame3Data = importdata("frame3.sift");
    slide1Data = importdata("slide1.sift");
    slide2Data = importdata("slide2.sift");
    slide3Data = importdata("slide3.sift");
    
    % Get the size of each keypoint descriptor file
	f1_size = size(frame1Data);
    s1_size = size(slide1Data);
    f2_size = size(frame2Data);
    s2_size = size(slide2Data);
    f3_size = size(frame3Data);
    s3_size = size(slide3Data);
    
    % Convert the keypoint matrix into a 3D matrix to help in finding the
    % minimum Euclidean distance
    % First frame-slide pair
    f1_3d_mat = permute(frame1Data(:, 5:132), [1 3 2]);
    f1_3d_mat = repmat(f1_3d_mat, 1, s1_size(1), 1);
    
    s1_3d_mat = permute(slide1Data(:, 5:132), [3 1 2]);
    s1_3d_mat = repmat(s1_3d_mat, f1_size(1), 1, 1);
    
    f1_distances = sqrt(sum((f1_3d_mat - s1_3d_mat) .^ 2, 3));
    [f1_mins, f1_entries] = min(f1_distances, [], 2);
    
    % Second frame-slide pair
    f2_3d_mat = permute(frame2Data(:, 5:132), [1 3 2]);
    f2_3d_mat = repmat(f2_3d_mat, 1, s2_size(1), 1);
    
    s2_3d_mat = permute(slide2Data(:, 5:132), [3 1 2]);
    s2_3d_mat = repmat(s2_3d_mat, f2_size(1), 1, 1);
    
    f2_distances = sqrt(sum((f2_3d_mat - s2_3d_mat) .^ 2, 3));
    [f2_mins, f2_entries] = min(f2_distances, [], 2);
    
    % Third frame-slide pair
    f3_3d_mat = permute(frame3Data(:, 5:132), [1 3 2]);
    f3_3d_mat = repmat(f3_3d_mat, 1, s3_size(1), 1);
    
    s3_3d_mat = permute(slide3Data(:, 5:132), [3 1 2]);
    s3_3d_mat = repmat(s3_3d_mat, f3_size(1), 1, 1);
    
    f3_distances = sqrt(sum((f3_3d_mat - s3_3d_mat) .^ 2, 3));
    [f3_mins, f3_entries] = min(f3_distances, [], 2);
    
    % Read in the frame and slide images over which to draw the results
    frame1_kp = imread('frame1.jpg');
    frame1_kp_fresh = frame1_kp;
    slide1_kp = imread('slide1.tiff');
    slide1_kp = slide1_kp(:, :, 1:3);
    slide1_kp_fresh = slide1_kp;
    
	frame2_kp = imread('frame2.jpg');
    frame2_kp_fresh = frame2_kp;
    slide2_kp = imread('slide2.tiff');
    slide2_kp = slide2_kp(:, :, 1:3);
    slide2_kp_fresh = slide2_kp;
    
	frame3_kp = imread('frame3.jpg');
    frame3_kp_fresh = frame3_kp;
    slide3_kp = imread('slide3.tiff');
    slide3_kp = cat(3, slide3_kp, slide3_kp, slide3_kp);
    slide3_kp_fresh = slide3_kp;
    
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
    
    % Start computation for first frame-slide pair
    for i = 1:5:f1_size(1)
        f1p1 = [frame1Data(i, 2) frame1Data(i, 1)];
        f1p2 = [(frame1Data(i, 2) + 5 * frame1Data(i, 3) * sin(frame1Data(i, 4))) (frame1Data(i, 1) + 5 * frame1Data(i, 3) * cos(frame1Data(i, 4)))];
        frame1_kp = draw_line(frame1_kp, f1p1, f1p2, 0, 0, 255, 0);
        frame1_kp = draw_line(frame1_kp, f1p1, f1p1, 1, 255, 0, 0);
    end
    figure();
    imshow(frame1_kp);
    
    for i = 1:5:s1_size(1)
        s1p1 = [slide1Data(i, 2) slide1Data(i, 1)];
        s1p2 = [(slide1Data(i, 2) + 5 * slide1Data(i, 3) * sin(slide1Data(i, 4))) (slide1Data(i, 1) + 5 * slide1Data(i, 3) * cos(slide1Data(i, 4)))];
        slide1_kp = draw_line(slide1_kp, s1p1, s1p2, 0, 0, 255, 0);
        slide1_kp = draw_line(slide1_kp, s1p1, s1p1, 1, 255, 0, 0);
    end
    figure();
    imshow(slide1_kp);
    
    comparison_img_1_new = comparison_img_1;
    for i = 1:20:f1_size(1)
        fs1p1 = [(15 + frame1Data(i, 2)) (360 + frame1Data(i, 1))];
        fs1p2 = [slide1Data(f1_entries(i), 2) slide1Data(f1_entries(i), 1)];
        comparison_img_1_new = draw_line(comparison_img_1_new, fs1p1, fs1p2, 0, 0, 255, 0);
        comparison_img_1_new = draw_line(comparison_img_1_new, fs1p1, fs1p1, 1, 255, 0, 0);
        comparison_img_1_new = draw_line(comparison_img_1_new, fs1p2, fs1p2, 1, 255, 0, 0);
    end
    figure();
    imshow(comparison_img_1_new);
    
	% Start computation for second frame-slide pair
    for i = 1:5:f2_size(1)
        f2p1 = [frame2Data(i, 2) frame2Data(i, 1)];
        f2p2 = [(frame2Data(i, 2) + 5 * frame2Data(i, 3) * sin(frame2Data(i, 4))) (frame2Data(i, 1) + 5 * frame2Data(i, 3) * cos(frame2Data(i, 4)))];
        frame2_kp = draw_line(frame2_kp, f2p1, f2p2, 0, 0, 255, 0);
        frame2_kp = draw_line(frame2_kp, f2p1, f2p1, 1, 255, 0, 0);
    end
    figure();
    imshow(frame2_kp);
    
    for i = 1:5:s2_size(1)
        s2p1 = [slide2Data(i, 2) slide2Data(i, 1)];
        s2p2 = [(slide2Data(i, 2) + 5 * slide2Data(i, 3) * sin(slide2Data(i, 4))) (slide2Data(i, 1) + 5 * slide2Data(i, 3) * cos(slide2Data(i, 4)))];
        slide2_kp = draw_line(slide2_kp, s2p1, s2p2, 0, 0, 255, 0);
        slide2_kp = draw_line(slide2_kp, s2p1, s2p1, 1, 255, 0, 0);
    end
    figure();
    imshow(slide2_kp);
    
    comparison_img_2_new = comparison_img_2;
    for i = 1:20:f2_size(1)
        fs2p1 = [(12 + frame2Data(i, 2)) (352 + frame2Data(i, 1))];
        fs2p2 = [slide2Data(f2_entries(i), 2) slide2Data(f2_entries(i), 1)];
        comparison_img_2_new = draw_line(comparison_img_2_new, fs2p1, fs2p2, 0, 0, 255, 0);
        comparison_img_2_new = draw_line(comparison_img_2_new, fs2p1, fs2p1, 1, 255, 0, 0);
        comparison_img_2_new = draw_line(comparison_img_2_new, fs2p2, fs2p2, 1, 255, 0, 0);
    end
    figure();
    imshow(comparison_img_2_new);
    
    % Start computation for third frame-slide pair
    for i = 1:5:f3_size(1)
        f3p1 = [frame3Data(i, 2) frame3Data(i, 1)];
        f3p2 = [(frame3Data(i, 2) + 5 * frame3Data(i, 3) * sin(frame3Data(i, 4))) (frame3Data(i, 1) + 5 * frame3Data(i, 3) * cos(frame3Data(i, 4)))];
        frame3_kp = draw_line(frame3_kp, f3p1, f3p2, 0, 0, 255, 0);
        frame3_kp = draw_line(frame3_kp, f3p1, f3p1, 1, 255, 0, 0);
    end
    figure();
    imshow(frame3_kp);
    
    for i = 1:5:s3_size(1)
        s3p1 = [slide3Data(i, 2) slide3Data(i, 1)];
        s3p2 = [(slide3Data(i, 2) + 5 * slide3Data(i, 3) * sin(slide3Data(i, 4))) (slide3Data(i, 1) + 5 * slide3Data(i, 3) * cos(slide3Data(i, 4)))];
        slide3_kp = draw_line(slide3_kp, s3p1, s3p2, 0, 0, 255, 0);
        slide3_kp = draw_line(slide3_kp, s3p1, s3p1, 1, 255, 0, 0);
    end
    figure();
    imshow(slide3_kp);
    
    comparison_img_3_new = comparison_img_3;
    for i = 1:20:f3_size(1)
        fs3p1 = [(12 + frame3Data(i, 2)) (352 + frame3Data(i, 1))];
        fs3p2 = [slide3Data(f3_entries(i), 2) slide3Data(f3_entries(i), 1)];
        comparison_img_3_new = draw_line(comparison_img_3_new, fs3p1, fs3p2, 0, 0, 255, 0);
        comparison_img_3_new = draw_line(comparison_img_3_new, fs3p1, fs3p1, 1, 255, 0, 0);
        comparison_img_3_new = draw_line(comparison_img_3_new, fs3p2, fs3p2, 1, 255, 0, 0);
    end
    figure();
    imshow(comparison_img_3_new);
    
    num_questions = num_questions + 1;
    
end

if questionFlag('2')
    
    % First frame-slide pair
    [min_sorted_1, min_old_indices_1] = sort(f1_mins);
    min_old_indices_size_1 = size(min_old_indices_1);
    min_old_indices_pruned_1 = min_old_indices_1(1:round(min_old_indices_size_1/20));
    min_old_indices_pruned_size_1 = size(min_old_indices_pruned_1);
    
    comparison_img_1_new = comparison_img_1;
    frame1_kp = frame1_kp_fresh;
    slide1_kp = slide1_kp_fresh;
    for i = 1:min_old_indices_pruned_size_1
        cur1_1st = min_old_indices_pruned_1(i);
        cur1_2nd = f1_entries(min_old_indices_pruned_1(i));
        f1p1 = [(15 + frame1Data(cur1_1st, 2)) (360 + frame1Data(cur1_1st, 1))];
        f1p2 = [slide1Data(cur1_2nd, 2) slide1Data(cur1_2nd, 1)];
        comparison_img_1_new = draw_line(comparison_img_1_new, f1p1, f1p2, 0, 0, 255, 0);
        comparison_img_1_new = draw_line(comparison_img_1_new, f1p1, f1p1, 1, 255, 0, 0);
        comparison_img_1_new = draw_line(comparison_img_1_new, f1p2, f1p2, 1, 255, 0, 0);
        
        f1p1 = [frame1Data(cur1_1st, 2) frame1Data(cur1_1st, 1)];
        f1p2 = [(frame1Data(cur1_1st, 2) + 5 * frame1Data(cur1_1st, 3) * sin(frame1Data(cur1_1st, 4))) (frame1Data(cur1_1st, 1) + 5 * frame1Data(cur1_1st, 3) * cos(frame1Data(cur1_1st, 4)))];
        frame1_kp = draw_line(frame1_kp, f1p1, f1p2, 0, 0, 255, 0);
        frame1_kp = draw_line(frame1_kp, f1p1, f1p1, 1, 255, 0, 0);
        
        s1p1 = [slide1Data(cur1_2nd, 2) slide1Data(cur1_2nd, 1)];
        s1p2 = [(slide1Data(cur1_2nd, 2) + 5 * slide1Data(cur1_2nd, 3) * sin(slide1Data(cur1_2nd, 4))) (slide1Data(cur1_2nd, 1) + 5 * slide1Data(cur1_2nd, 3) * cos(slide1Data(cur1_2nd, 4)))];
        slide1_kp = draw_line(slide1_kp, s1p1, s1p2, 0, 0, 255, 0);
        slide1_kp = draw_line(slide1_kp, s1p1, s1p1, 1, 255, 0, 0);
    end
    figure();
    imshow(frame1_kp);
    
    figure();
    imshow(slide1_kp);
    
    figure();
    imshow(comparison_img_1_new);
    
    % Second frame-slide pair
    [min_sorted_2, min_old_indices_2] = sort(f2_mins);
    min_old_indices_size_2 = size(min_old_indices_2);
    min_old_indices_pruned_2 = min_old_indices_2(1:round(min_old_indices_size_2/20));
    min_old_indices_pruned_size_2 = size(min_old_indices_pruned_2);
    
    comparison_img_2_new = comparison_img_2;
	frame2_kp = frame2_kp_fresh;
    slide2_kp = slide2_kp_fresh;
    for i = 1:min_old_indices_pruned_size_2
        cur2_1st = min_old_indices_pruned_2(i);
        cur2_2nd = f2_entries(min_old_indices_pruned_2(i));
        f2p1 = [(12 + frame2Data(cur2_1st, 2)) (352 + frame2Data(cur2_1st, 1))];
        f2p2 = [slide2Data(cur2_2nd, 2) slide2Data(cur2_2nd, 1)];
        comparison_img_2_new = draw_line(comparison_img_2_new, f2p1, f2p2, 0, 0, 255, 0);
        comparison_img_2_new = draw_line(comparison_img_2_new, f2p1, f2p1, 1, 255, 0, 0);
        comparison_img_2_new = draw_line(comparison_img_2_new, f2p2, f2p2, 1, 255, 0, 0);
        
        f2p1 = [frame2Data(cur2_1st, 2) frame2Data(cur2_1st, 1)];
        f2p2 = [(frame2Data(cur2_1st, 2) + 5 * frame2Data(cur2_1st, 3) * sin(frame2Data(cur2_1st, 4))) (frame2Data(cur2_1st, 1) + 5 * frame2Data(cur2_1st, 3) * cos(frame2Data(cur2_1st, 4)))];
        frame2_kp = draw_line(frame2_kp, f2p1, f2p2, 0, 0, 255, 0);
        frame2_kp = draw_line(frame2_kp, f2p1, f2p1, 1, 255, 0, 0);
        
        s2p1 = [slide2Data(cur2_2nd, 2) slide2Data(cur2_2nd, 1)];
        s2p2 = [(slide2Data(cur2_2nd, 2) + 5 * slide2Data(cur2_2nd, 3) * sin(slide2Data(cur2_2nd, 4))) (slide2Data(cur2_2nd, 1) + 5 * slide2Data(cur2_2nd, 3) * cos(slide2Data(cur2_2nd, 4)))];
        slide2_kp = draw_line(slide2_kp, s2p1, s2p2, 0, 0, 255, 0);
        slide2_kp = draw_line(slide2_kp, s2p1, s2p1, 1, 255, 0, 0);
    end
    figure();
    imshow(frame2_kp);
    
    figure();
    imshow(slide2_kp);
    
    figure();
    imshow(comparison_img_2_new);
    
    % Third frame-slide pair
    [min_sorted_3, min_old_indices_3] = sort(f3_mins);
    min_old_indices_size_3 = size(min_old_indices_3);
    min_old_indices_pruned_3 = min_old_indices_3(1:round(min_old_indices_size_3/20));
    min_old_indices_pruned_size_3 = size(min_old_indices_pruned_3);
    
    comparison_img_3_new = comparison_img_3;
	frame3_kp = frame3_kp_fresh;
    slide3_kp = slide3_kp_fresh;
    for i = 1:min_old_indices_pruned_size_3
        cur3_1st = min_old_indices_pruned_3(i);
        cur3_2nd = f3_entries(min_old_indices_pruned_3(i));
        f3p1 = [(12 + frame3Data(cur3_1st, 2)) (352 + frame3Data(cur3_1st, 1))];
        f3p2 = [slide3Data(cur3_2nd, 2) slide3Data(cur3_2nd, 1)];
        comparison_img_3_new = draw_line(comparison_img_3_new, f3p1, f3p2, 0, 0, 255, 0);
        comparison_img_3_new = draw_line(comparison_img_3_new, f3p1, f3p1, 1, 255, 0, 0);
        comparison_img_3_new = draw_line(comparison_img_3_new, f3p2, f3p2, 1, 255, 0, 0);
        
        f3p1 = [frame3Data(cur3_1st, 2) frame3Data(cur3_1st, 1)];
        f3p2 = [(frame3Data(cur3_1st, 2) + 5 * frame3Data(cur3_1st, 3) * sin(frame3Data(cur3_1st, 4))) (frame3Data(cur3_1st, 1) + 5 * frame3Data(cur3_1st, 3) * cos(frame3Data(cur3_1st, 4)))];
        frame3_kp = draw_line(frame3_kp, f3p1, f3p2, 0, 0, 255, 0);
        frame3_kp = draw_line(frame3_kp, f3p1, f3p1, 1, 255, 0, 0);
        
        s3p1 = [slide3Data(cur3_2nd, 2) slide3Data(cur3_2nd, 1)];
        s3p2 = [(slide3Data(cur3_2nd, 2) + 5 * slide3Data(cur3_2nd, 3) * sin(slide3Data(cur3_2nd, 4))) (slide3Data(cur3_2nd, 1) + 5 * slide3Data(cur3_2nd, 3) * cos(slide3Data(cur3_2nd, 4)))];
        slide3_kp = draw_line(slide3_kp, s3p1, s3p2, 0, 0, 255, 0);
        slide3_kp = draw_line(slide3_kp, s3p1, s3p1, 1, 255, 0, 0);
    end
    figure();
    imshow(frame3_kp);
    
    figure();
    imshow(slide3_kp);
    
    figure();
    imshow(comparison_img_3_new);
    
	num_questions = num_questions + 1;
    
end

if questionFlag('3')
    
    % a dot b = mag(a) * mag(b) * cos(theta)
    % cos(theta) = (mag(a) * mag(b)) / (a dot b)
    % theta = arccos((mag(a) * mag(b)) / (a dot b))
    
    % First frame-slide pair
    mag_f1 = sqrt(sum(frame1Data(:, 5:128) .^ 2, 2));
    mag_s1 = sqrt(sum(slide1Data(:, 5:128) .^ 2, 2));
    angles_numerator_1 = mag_f1 * mag_s1';
    angles_denominator_1 = sum(f1_3d_mat .* s1_3d_mat, 3);
    angles_1 = acos(angles_denominator_1 ./ angles_numerator_1);
    [min_angles_1, argmin_angles_1] = min(angles_1, [], 2);
    [min_angles_sorted_1, min_angles_old_indices_1] = sort(min_angles_1);
    min_angles_old_indices_pruned_1 = min_angles_old_indices_1(1:round(min_old_indices_size_1/20));
    min_angles_old_indices_pruned_size_1 = size(min_angles_old_indices_pruned_1);
    
    comparison_img_1_new = comparison_img_1;
	frame1_kp = frame1_kp_fresh;
    slide1_kp = slide1_kp_fresh;
    for i = 1:min_angles_old_indices_pruned_size_1
        cur1_1st = min_angles_old_indices_pruned_1(i);
        cur1_2nd = argmin_angles_1(min_angles_old_indices_pruned_1(i));
        f1p1 = [(15 + frame1Data(cur1_1st, 2)) (360 + frame1Data(cur1_1st, 1))];
        f1p2 = [slide1Data(cur1_2nd, 2) slide1Data(cur1_2nd, 1)];
        comparison_img_1_new = draw_line(comparison_img_1_new, f1p1, f1p2, 0, 0, 255, 0);
        comparison_img_1_new = draw_line(comparison_img_1_new, f1p1, f1p1, 1, 255, 0, 0);
        comparison_img_1_new = draw_line(comparison_img_1_new, f1p2, f1p2, 1, 255, 0, 0);
        
        f1p1 = [frame1Data(cur1_1st, 2) frame1Data(cur1_1st, 1)];
        f1p2 = [(frame1Data(cur1_1st, 2) + 5 * frame1Data(cur1_1st, 3) * sin(frame1Data(cur1_1st, 4))) (frame1Data(cur1_1st, 1) + 5 * frame1Data(cur1_1st, 3) * cos(frame1Data(cur1_1st, 4)))];
        frame1_kp = draw_line(frame1_kp, f1p1, f1p2, 0, 0, 255, 0);
        frame1_kp = draw_line(frame1_kp, f1p1, f1p1, 1, 255, 0, 0);
        
        s1p1 = [slide1Data(cur1_2nd, 2) slide1Data(cur1_2nd, 1)];
        s1p2 = [(slide1Data(cur1_2nd, 2) + 5 * slide1Data(cur1_2nd, 3) * sin(slide1Data(cur1_2nd, 4))) (slide1Data(cur1_2nd, 1) + 5 * slide1Data(cur1_2nd, 3) * cos(slide1Data(cur1_2nd, 4)))];
        slide1_kp = draw_line(slide1_kp, s1p1, s1p2, 0, 0, 255, 0);
        slide1_kp = draw_line(slide1_kp, s1p1, s1p1, 1, 255, 0, 0);
    end
    figure();
    imshow(frame1_kp);
    
    figure();
    imshow(slide1_kp);
    
    figure();
    imshow(comparison_img_1_new);
    
	% Second frame-slide pair
    mag_f2 = sqrt(sum(frame2Data(:, 5:128) .^ 2, 2));
    mag_s2 = sqrt(sum(slide2Data(:, 5:128) .^ 2, 2));
    angles_numerator_2 = mag_f2 * mag_s2';
    angles_denominator_2 = sum(f2_3d_mat .* s2_3d_mat, 3);
    angles_2 = acos(angles_denominator_2 ./ angles_numerator_2);
    [min_angles_2, argmin_angles_2] = min(angles_2, [], 2);
    [min_angles_sorted_2, min_angles_old_indices_2] = sort(min_angles_2);
    min_angles_old_indices_pruned_2 = min_angles_old_indices_2(1:round(min_old_indices_size_2/20));
    min_angles_old_indices_pruned_size_2 = size(min_angles_old_indices_pruned_2);
    
    comparison_img_2_new = comparison_img_2;
	frame2_kp = frame2_kp_fresh;
    slide2_kp = slide2_kp_fresh;
    for i = 1:min_angles_old_indices_pruned_size_2
        cur2_1st = min_angles_old_indices_pruned_2(i);
        cur2_2nd = argmin_angles_2(min_angles_old_indices_pruned_2(i));
        f2p1 = [(12 + frame2Data(cur2_1st, 2)) (352 + frame2Data(cur2_1st, 1))];
        f2p2 = [slide2Data(cur2_2nd, 2) slide2Data(cur2_2nd, 1)];
        comparison_img_2_new = draw_line(comparison_img_2_new, f2p1, f2p2, 0, 0, 255, 0);
        comparison_img_2_new = draw_line(comparison_img_2_new, f2p1, f2p1, 1, 255, 0, 0);
        comparison_img_2_new = draw_line(comparison_img_2_new, f2p2, f2p2, 1, 255, 0, 0);
        
        f2p1 = [frame2Data(cur2_1st, 2) frame2Data(cur2_1st, 1)];
        f2p2 = [(frame2Data(cur2_1st, 2) + 5 * frame2Data(cur2_1st, 3) * sin(frame2Data(cur2_1st, 4))) (frame2Data(cur2_1st, 1) + 5 * frame2Data(cur2_1st, 3) * cos(frame2Data(cur2_1st, 4)))];
        frame2_kp = draw_line(frame2_kp, f2p1, f2p2, 0, 0, 255, 0);
        frame2_kp = draw_line(frame2_kp, f2p1, f2p1, 1, 255, 0, 0);
        
        s2p1 = [slide2Data(cur2_2nd, 2) slide2Data(cur2_2nd, 1)];
        s2p2 = [(slide2Data(cur2_2nd, 2) + 5 * slide2Data(cur2_2nd, 3) * sin(slide2Data(cur2_2nd, 4))) (slide2Data(cur2_2nd, 1) + 5 * slide2Data(cur2_2nd, 3) * cos(slide2Data(cur2_2nd, 4)))];
        slide2_kp = draw_line(slide2_kp, s2p1, s2p2, 0, 0, 255, 0);
        slide2_kp = draw_line(slide2_kp, s2p1, s2p1, 1, 255, 0, 0);
    end
    figure();
    imshow(frame2_kp);
    
    figure();
    imshow(slide2_kp);
    
    figure();
    imshow(comparison_img_2_new);
    
	% Third frame-slide pair
    mag_f3 = sqrt(sum(frame3Data(:, 5:128) .^ 2, 2));
    mag_s3 = sqrt(sum(slide3Data(:, 5:128) .^ 2, 2));
    angles_numerator_3 = mag_f3 * mag_s3';
    angles_denominator_3 = sum(f3_3d_mat .* s3_3d_mat, 3);
    angles_3 = acos(angles_denominator_3 ./ angles_numerator_3);
    [min_angles_3, argmin_angles_3] = min(angles_3, [], 2);
    [min_angles_sorted_3, min_angles_old_indices_3] = sort(min_angles_3);
    min_angles_old_indices_pruned_3 = min_angles_old_indices_3(1:round(min_old_indices_size_3/20));
    min_angles_old_indices_pruned_size_3 = size(min_angles_old_indices_pruned_3);
    
    comparison_img_3_new = comparison_img_3;
	frame3_kp = frame3_kp_fresh;
    slide3_kp = slide3_kp_fresh;
    for i = 1:min_angles_old_indices_pruned_size_3
        cur3_1st = min_angles_old_indices_pruned_3(i);
        cur3_2nd = argmin_angles_3(min_angles_old_indices_pruned_3(i));
        f3p1 = [(12 + frame3Data(cur3_1st, 2)) (352 + frame3Data(cur3_1st, 1))];
        f3p2 = [slide3Data(cur3_2nd, 2) slide3Data(cur3_2nd, 1)];
        comparison_img_3_new = draw_line(comparison_img_3_new, f3p1, f3p2, 0, 0, 255, 0);
        comparison_img_3_new = draw_line(comparison_img_3_new, f3p1, f3p1, 1, 255, 0, 0);
        comparison_img_3_new = draw_line(comparison_img_3_new, f3p2, f3p2, 1, 255, 0, 0);
        
        f3p1 = [frame3Data(cur3_1st, 2) frame3Data(cur3_1st, 1)];
        f3p2 = [(frame3Data(cur3_1st, 2) + 5 * frame3Data(cur3_1st, 3) * sin(frame3Data(cur3_1st, 4))) (frame3Data(cur3_1st, 1) + 5 * frame3Data(cur3_1st, 3) * cos(frame3Data(cur3_1st, 4)))];
        frame3_kp = draw_line(frame3_kp, f3p1, f3p2, 0, 0, 255, 0);
        frame3_kp = draw_line(frame3_kp, f3p1, f3p1, 1, 255, 0, 0);
        
        s3p1 = [slide3Data(cur3_2nd, 2) slide3Data(cur3_2nd, 1)];
        s3p2 = [(slide3Data(cur3_2nd, 2) + 5 * slide3Data(cur3_2nd, 3) * sin(slide3Data(cur3_2nd, 4))) (slide3Data(cur3_2nd, 1) + 5 * slide3Data(cur3_2nd, 3) * cos(slide3Data(cur3_2nd, 4)))];
        slide3_kp = draw_line(slide3_kp, s3p1, s3p2, 0, 0, 255, 0);
        slide3_kp = draw_line(slide3_kp, s3p1, s3p1, 1, 255, 0, 0);
    end
    figure();
    imshow(frame3_kp);
    
    figure();
    imshow(slide3_kp);
    
    figure();
    imshow(comparison_img_3_new);
    
	num_questions = num_questions + 1;
    
end

if questionFlag('4')
    
    % First frame-slide pair
    f1_3d_mat_smooth = f1_3d_mat + 0.0001;
    s1_3d_mat_smooth = s1_3d_mat + 0.0001;
    chi_sq_mat_1 = 0.5 .* sum((f1_3d_mat_smooth - s1_3d_mat_smooth) .^ 2 ./ (f1_3d_mat_smooth + s1_3d_mat_smooth), 3);
    
    [min_chi_sq_1, argmin_chi_sq_1] = min(chi_sq_mat_1, [], 2);
    [min_chi_sq_sorted_1, min_chi_sq_old_indices_1] = sort(min_chi_sq_1);
    min_chi_sq_old_indices_pruned_1 = min_chi_sq_old_indices_1(1:round(min_old_indices_size_1/40));
    min_chi_sq_old_indices_pruned_size_1 = size(min_chi_sq_old_indices_pruned_1);
    
    comparison_img_1_new = comparison_img_1;
	frame1_kp = frame1_kp_fresh;
    slide1_kp = slide1_kp_fresh;
    for i = 1:min_chi_sq_old_indices_pruned_size_1
        cur1_1st = min_chi_sq_old_indices_pruned_1(i);
        cur1_2nd = argmin_chi_sq_1(min_chi_sq_old_indices_pruned_1(i));
        f1p1 = [(15 + frame1Data(cur1_1st, 2)) (360 + frame1Data(cur1_1st, 1))];
        f1p2 = [slide1Data(cur1_2nd, 2) slide1Data(cur1_2nd, 1)];
        comparison_img_1_new = draw_line(comparison_img_1_new, f1p1, f1p2, 0, 0, 255, 0);
        comparison_img_1_new = draw_line(comparison_img_1_new, f1p1, f1p1, 1, 255, 0, 0);
        comparison_img_1_new = draw_line(comparison_img_1_new, f1p2, f1p2, 1, 255, 0, 0);
        
        f1p1 = [frame1Data(cur1_1st, 2) frame1Data(cur1_1st, 1)];
        f1p2 = [(frame1Data(cur1_1st, 2) + 5 * frame1Data(cur1_1st, 3) * sin(frame1Data(cur1_1st, 4))) (frame1Data(cur1_1st, 1) + 5 * frame1Data(cur1_1st, 3) * cos(frame1Data(cur1_1st, 4)))];
        frame1_kp = draw_line(frame1_kp, f1p1, f1p2, 0, 0, 255, 0);
        frame1_kp = draw_line(frame1_kp, f1p1, f1p1, 1, 255, 0, 0);
        
        s1p1 = [slide1Data(cur1_2nd, 2) slide1Data(cur1_2nd, 1)];
        s1p2 = [(slide1Data(cur1_2nd, 2) + 5 * slide1Data(cur1_2nd, 3) * sin(slide1Data(cur1_2nd, 4))) (slide1Data(cur1_2nd, 1) + 5 * slide1Data(cur1_2nd, 3) * cos(slide1Data(cur1_2nd, 4)))];
        slide1_kp = draw_line(slide1_kp, s1p1, s1p2, 0, 0, 255, 0);
        slide1_kp = draw_line(slide1_kp, s1p1, s1p1, 1, 255, 0, 0);
    end
    figure();
    imshow(frame1_kp);
    
    figure();
    imshow(slide1_kp);
    
    figure();
    imshow(comparison_img_1_new);
        
    % Second frame-slide pair
    f2_3d_mat_smooth = f2_3d_mat + 0.0001;
    s2_3d_mat_smooth = s2_3d_mat + 0.0001;
    chi_sq_mat_2 = 0.5 .* sum((f2_3d_mat_smooth - s2_3d_mat_smooth) .^ 2 ./ (f2_3d_mat_smooth + s2_3d_mat_smooth), 3);
    
    [min_chi_sq_2, argmin_chi_sq_2] = min(chi_sq_mat_2, [], 2);
    [min_chi_sq_sorted_2, min_chi_sq_old_indices_2] = sort(min_chi_sq_2);
    min_chi_sq_old_indices_pruned_2 = min_chi_sq_old_indices_2(1:round(min_old_indices_size_2/40));
    min_chi_sq_old_indices_pruned_size_2 = size(min_chi_sq_old_indices_pruned_2);
    
    comparison_img_2_new = comparison_img_2;
	frame2_kp = frame2_kp_fresh;
    slide2_kp = slide2_kp_fresh;
	for i = 1:min_chi_sq_old_indices_pruned_size_2
        cur2_1st = min_chi_sq_old_indices_pruned_2(i);
        cur2_2nd = argmin_chi_sq_2(min_chi_sq_old_indices_pruned_2(i));
        f2p1 = [(12 + frame2Data(cur2_1st, 2)) (352 + frame2Data(cur2_1st, 1))];
        f2p2 = [slide2Data(cur2_2nd, 2) slide2Data(cur2_2nd, 1)];
        comparison_img_2_new = draw_line(comparison_img_2_new, f2p2, f2p1, 0, 0, 255, 0);
        comparison_img_2_new = draw_line(comparison_img_2_new, f2p1, f2p1, 1, 255, 0, 0);
        comparison_img_2_new = draw_line(comparison_img_2_new, f2p2, f2p2, 1, 255, 0, 0);
        
        f2p1 = [frame2Data(cur2_1st, 2) frame2Data(cur2_1st, 1)];
        f2p2 = [(frame2Data(cur2_1st, 2) + 5 * frame2Data(cur2_1st, 3) * sin(frame2Data(cur2_1st, 4))) (frame2Data(cur2_1st, 1) + 5 * frame2Data(cur2_1st, 3) * cos(frame2Data(cur2_1st, 4)))];
        frame2_kp = draw_line(frame2_kp, f2p1, f2p2, 0, 0, 255, 0);
        frame2_kp = draw_line(frame2_kp, f2p1, f2p1, 1, 255, 0, 0);
        
        s2p1 = [slide2Data(cur2_2nd, 2) slide2Data(cur2_2nd, 1)];
        s2p2 = [(slide2Data(cur2_2nd, 2) + 5 * slide2Data(cur2_2nd, 3) * sin(slide2Data(cur2_2nd, 4))) (slide2Data(cur2_2nd, 1) + 5 * slide2Data(cur2_2nd, 3) * cos(slide2Data(cur2_2nd, 4)))];
        slide2_kp = draw_line(slide2_kp, s2p1, s2p2, 0, 0, 255, 0);
        slide2_kp = draw_line(slide2_kp, s2p1, s2p1, 1, 255, 0, 0);
	end
    figure();
    imshow(frame2_kp);
    
    figure();
    imshow(slide2_kp);
    
    figure();
    imshow(comparison_img_2_new);
    
    % Third frame-slide pair
	f3_3d_mat_smooth = f3_3d_mat + 0.0001;
    s3_3d_mat_smooth = s3_3d_mat + 0.0001;
    chi_sq_mat_3 = 0.5 .* sum((f3_3d_mat_smooth - s3_3d_mat_smooth) .^ 2 ./ (f3_3d_mat_smooth + s3_3d_mat_smooth), 3);
    
    [min_chi_sq_3, argmin_chi_sq_3] = min(chi_sq_mat_3, [], 2);
    [min_chi_sq_sorted_3, min_chi_sq_old_indices_3] = sort(min_chi_sq_3);
    min_chi_sq_old_indices_pruned_3 = min_chi_sq_old_indices_3(1:round(min_old_indices_size_3/40));
    min_chi_sq_old_indices_pruned_size_3 = size(min_chi_sq_old_indices_pruned_3);
    
    comparison_img_3_new = comparison_img_3;
	frame3_kp = frame3_kp_fresh;
    slide3_kp = slide3_kp_fresh;
    for i = 1:min_chi_sq_old_indices_pruned_size_3
        cur3_1st = min_chi_sq_old_indices_pruned_3(i);
        cur3_2nd = argmin_chi_sq_3(min_chi_sq_old_indices_pruned_3(i));
        f3p1 = [(12 + frame3Data(cur3_1st, 2)) (352 + frame3Data(cur3_1st, 1))];
        f3p2 = [slide3Data(cur3_2nd, 2) slide3Data(cur3_2nd, 1)];
        comparison_img_3_new = draw_line(comparison_img_3_new, f3p1, f3p2, 0, 0, 255, 0);
        comparison_img_3_new = draw_line(comparison_img_3_new, f3p1, f3p1, 1, 255, 0, 0);
        comparison_img_3_new = draw_line(comparison_img_3_new, f3p2, f3p2, 1, 255, 0, 0);
        
        f3p1 = [frame3Data(cur3_1st, 2) frame3Data(cur3_1st, 1)];
        f3p2 = [(frame3Data(cur3_1st, 2) + 5 * frame3Data(cur3_1st, 3) * sin(frame3Data(cur3_1st, 4))) (frame3Data(cur3_1st, 1) + 5 * frame3Data(cur3_1st, 3) * cos(frame3Data(cur3_1st, 4)))];
        frame3_kp = draw_line(frame3_kp, f3p1, f3p2, 0, 0, 255, 0);
        frame3_kp = draw_line(frame3_kp, f3p1, f3p1, 1, 255, 0, 0);
        
        s3p1 = [slide3Data(cur3_2nd, 2) slide3Data(cur3_2nd, 1)];
        s3p2 = [(slide3Data(cur3_2nd, 2) + 5 * slide3Data(cur3_2nd, 3) * sin(slide3Data(cur3_2nd, 4))) (slide3Data(cur3_2nd, 1) + 5 * slide3Data(cur3_2nd, 3) * cos(slide3Data(cur3_2nd, 4)))];
        slide3_kp = draw_line(slide3_kp, s3p1, s3p2, 0, 0, 255, 0);
        slide3_kp = draw_line(slide3_kp, s3p1, s3p1, 1, 255, 0, 0);
    end
    figure();
    imshow(frame3_kp);
    
    figure();
    imshow(slide3_kp);
    
    figure();
    imshow(comparison_img_3_new);   
    
	num_questions = num_questions + 1;
        
end

if questionFlag('5')
    
	% Both methods (top 20% or ratios) have a parameter to tune (how many
    % to keep vs the ratio), but top 20% assumes that 20% of the features
    % will have valid matches. What if they're pictures of two very
    % different things? Then the ratio approach will be much better.
    
    % First frame-slide pair
    second_distances_1 = f1_distances;
    for i = 1:f1_size(1)
        second_distances_1(i, f1_entries(i)) = Inf;
    end
    [second_mins_1, second_entries_1] = min(second_distances_1, [], 2);
    ratios_1 = f1_mins ./ second_mins_1;
    good_ratio_indices_1 = find(ratios_1 <= 0.7);
    good_ratio_indices_size_1 = size(good_ratio_indices_1);
    
    comparison_img_1_new = comparison_img_1;
	frame1_kp = frame1_kp_fresh;
    slide1_kp = slide1_kp_fresh;
    for i = 1:good_ratio_indices_size_1
        cur1_1st = good_ratio_indices_1(i);
        cur1_2nd = f1_entries(good_ratio_indices_1(i));
        f1p1 = [(15 + frame1Data(cur1_1st, 2)) (360 + frame1Data(cur1_1st, 1))];
        f1p2 = [slide1Data(cur1_2nd, 2) slide1Data(cur1_2nd, 1)];
        comparison_img_1_new = draw_line(comparison_img_1_new, f1p1, f1p2, 0, 0, 255, 0);
        comparison_img_1_new = draw_line(comparison_img_1_new, f1p1, f1p1, 1, 255, 0, 0);
        comparison_img_1_new = draw_line(comparison_img_1_new, f1p2, f1p2, 1, 255, 0, 0);
        
        f1p1 = [frame1Data(cur1_1st, 2) frame1Data(cur1_1st, 1)];
        f1p2 = [(frame1Data(cur1_1st, 2) + 5 * frame1Data(cur1_1st, 3) * sin(frame1Data(cur1_1st, 4))) (frame1Data(cur1_1st, 1) + 5 * frame1Data(cur1_1st, 3) * cos(frame1Data(cur1_1st, 4)))];
        frame1_kp = draw_line(frame1_kp, f1p1, f1p2, 0, 0, 255, 0);
        frame1_kp = draw_line(frame1_kp, f1p1, f1p1, 1, 255, 0, 0);
        
        s1p1 = [slide1Data(cur1_2nd, 2) slide1Data(cur1_2nd, 1)];
        s1p2 = [(slide1Data(cur1_2nd, 2) + 5 * slide1Data(cur1_2nd, 3) * sin(slide1Data(cur1_2nd, 4))) (slide1Data(cur1_2nd, 1) + 5 * slide1Data(cur1_2nd, 3) * cos(slide1Data(cur1_2nd, 4)))];
        slide1_kp = draw_line(slide1_kp, s1p1, s1p2, 0, 0, 255, 0);
        slide1_kp = draw_line(slide1_kp, s1p1, s1p1, 1, 255, 0, 0);
    end
    figure();
    imshow(frame1_kp);
    
    figure();
    imshow(slide1_kp);
    
    figure();
    imshow(comparison_img_1_new);
    
	% Second frame-slide pair
    second_distances_2 = f2_distances;
    for i = 1:f2_size(1)
        second_distances_2(i, f2_entries(i)) = Inf;
    end
    [second_mins_2, second_entries_2] = min(second_distances_2, [], 2);
    ratios_2 = f2_mins ./ second_mins_2;
    good_ratio_indices_2 = find(ratios_2 <= 0.7);
    good_ratio_indices_size_2 = size(good_ratio_indices_2);
    
    comparison_img_2_new = comparison_img_2;
	frame2_kp = frame2_kp_fresh;
    slide2_kp = slide2_kp_fresh;
    for i = 1:good_ratio_indices_size_2
        cur2_1st = good_ratio_indices_2(i);
        cur2_2nd = f2_entries(good_ratio_indices_2(i));
        f2p1 = [(12 + frame2Data(cur2_1st, 2)) (352 + frame2Data(cur2_1st, 1))];
        f2p2 = [slide2Data(cur2_2nd, 2) slide2Data(cur2_2nd, 1)];
        comparison_img_2_new = draw_line(comparison_img_2_new, f2p1, f2p2, 0, 0, 255, 0);
        comparison_img_2_new = draw_line(comparison_img_2_new, f2p1, f2p1, 1, 255, 0, 0);
        comparison_img_2_new = draw_line(comparison_img_2_new, f2p2, f2p2, 1, 255, 0, 0);
        
        f2p1 = [frame2Data(cur2_1st, 2) frame2Data(cur2_1st, 1)];
        f2p2 = [(frame2Data(cur2_1st, 2) + 5 * frame2Data(cur2_1st, 3) * sin(frame2Data(cur2_1st, 4))) (frame2Data(cur2_1st, 1) + 5 * frame2Data(cur2_1st, 3) * cos(frame2Data(cur2_1st, 4)))];
        frame2_kp = draw_line(frame2_kp, f2p1, f2p2, 0, 0, 255, 0);
        frame2_kp = draw_line(frame2_kp, f2p1, f2p1, 1, 255, 0, 0);
        
        s2p1 = [slide2Data(cur2_2nd, 2) slide2Data(cur2_2nd, 1)];
        s2p2 = [(slide2Data(cur2_2nd, 2) + 5 * slide2Data(cur2_2nd, 3) * sin(slide2Data(cur2_2nd, 4))) (slide2Data(cur2_2nd, 1) + 5 * slide2Data(cur2_2nd, 3) * cos(slide2Data(cur2_2nd, 4)))];
        slide2_kp = draw_line(slide2_kp, s2p1, s2p2, 0, 0, 255, 0);
        slide2_kp = draw_line(slide2_kp, s2p1, s2p1, 1, 255, 0, 0);
    end
    figure();
    imshow(frame2_kp);
    
    figure();
    imshow(slide2_kp);
    
    figure();
    imshow(comparison_img_2_new);
   
	% Third frame-slide pair
    second_distances_3 = f3_distances;
    for i = 1:f3_size(1)
        second_distances_3(i, f3_entries(i)) = Inf;
    end
    [second_mins_3 second_entries_3] = min(second_distances_3, [], 2);
    ratios_3 = f3_mins ./ second_mins_3;
    good_ratio_indices_3 = find(ratios_3 <= 0.7);
    good_ratio_indices_size_3 = size(good_ratio_indices_3);
    
    comparison_img_3_new = comparison_img_3;
	frame3_kp = frame3_kp_fresh;
    slide3_kp = slide3_kp_fresh;
    for i = 1:good_ratio_indices_size_3
        cur3_1st = good_ratio_indices_3(i);
        cur3_2nd = f3_entries(good_ratio_indices_3(i));
        f3p1 = [(12 + frame3Data(cur3_1st, 2)) (352 + frame3Data(cur3_1st, 1))];
        f3p2 = [slide3Data(cur3_2nd, 2) slide3Data(cur3_2nd, 1)];
        comparison_img_3_new = draw_line(comparison_img_3_new, f3p1, f3p2, 0, 0, 255, 0);
        comparison_img_3_new = draw_line(comparison_img_3_new, f3p1, f3p1, 1, 255, 0, 0);
        comparison_img_3_new = draw_line(comparison_img_3_new, f3p2, f3p2, 1, 255, 0, 0);
        
        f3p1 = [frame3Data(cur3_1st, 2) frame3Data(cur3_1st, 1)];
        f3p2 = [(frame3Data(cur3_1st, 2) + 5 * frame3Data(cur3_1st, 3) * sin(frame3Data(cur3_1st, 4))) (frame3Data(cur3_1st, 1) + 5 * frame3Data(cur3_1st, 3) * cos(frame3Data(cur3_1st, 4)))];
        frame3_kp = draw_line(frame3_kp, f3p1, f3p2, 0, 0, 255, 0);
        frame3_kp = draw_line(frame3_kp, f3p1, f3p1, 1, 255, 0, 0);
        
        s3p1 = [slide3Data(cur3_2nd, 2) slide3Data(cur3_2nd, 1)];
        s3p2 = [(slide3Data(cur3_2nd, 2) + 5 * slide3Data(cur3_2nd, 3) * sin(slide3Data(cur3_2nd, 4))) (slide3Data(cur3_2nd, 1) + 5 * slide3Data(cur3_2nd, 3) * cos(slide3Data(cur3_2nd, 4)))];
        slide3_kp = draw_line(slide3_kp, s3p1, s3p2, 0, 0, 255, 0);
        slide3_kp = draw_line(slide3_kp, s3p1, s3p1, 1, 255, 0, 0);
    end
    figure();
    imshow(frame3_kp);
    
    figure();
    imshow(slide3_kp);
    
    figure();
    imshow(comparison_img_3_new);
    
	num_questions = num_questions + 1;
    
end

if questionFlag('6')
    
    combined_frame_matrix = [permute(frame1Data(:, 5:132), [1 3 2]);  permute(frame2Data(:, 5:132), [1 3 2]); permute(frame3Data(:, 5:132), [1 3 2])];
    combined_frame_matrix = repmat(combined_frame_matrix, 1, s1_size(1) + s2_size(1) + s3_size(1), 1);
    f1_indices = 1:f1_size(1);
    f2_indices = (f1_size(1) + 1):(f1_size(1) + f2_size(1));
    f3_indices = (f1_size(1) + f2_size(1) + 1):(f1_size(1) + f2_size(1) + f3_size(1));
    
    combined_slide_matrix = [permute(slide1Data(:, 5:132), [3 1 2]) permute(slide2Data(:, 5:132), [3 1 2]) permute(slide3Data(:, 5:132), [3 1 2])];
    combined_slide_matrix = repmat(combined_slide_matrix, f1_size(1) + f2_size(1) + f3_size(1), 1, 1);
    s1_indices = 1:s1_size(1);
    s2_indices = (s1_size(1) + 1):(s1_size(1) + s2_size(1));
    s3_indices = (s1_size(1) + s2_size(1) + 1):(s1_size(1) + s2_size(1) + s3_size(1));
    
    combined_distance_matrix = sqrt(sum((combined_frame_matrix - combined_slide_matrix).^2, 3));
    
    [min_combined_distances, argmin_combined_distances] = min(combined_distance_matrix, [], 2);
    [sorted_combined_distance_values, old_combined_distance_indices] = sort(min_combined_distances);
    
    combined_indices_to_consider = old_combined_distance_indices(1:round(((f1_size(1) + f2_size(1) + f3_size(1)) * 0.05)));
    combined_indices_to_consider_size = size(combined_indices_to_consider);
    
    confusion_matrix = zeros(3);
    for i = 1:combined_indices_to_consider_size(1)
        f_cur_idx = combined_indices_to_consider(i);
        s_cur_idx = argmin_combined_distances(f_cur_idx);
        cur_row = -1;
        cur_col = -1;
        if size(find(f1_indices == f_cur_idx)) > 0
            cur_col = 1;
        elseif size(find(f2_indices == f_cur_idx)) > 0
            cur_col = 2;
        else
            cur_col = 3;
        end
        if size(find(s1_indices == s_cur_idx)) > 0
            cur_row = 1;
        elseif size(find(s2_indices == s_cur_idx)) > 0
            cur_row = 2;
        else
            cur_row = 3;
        end
        confusion_matrix(cur_row, cur_col) = confusion_matrix(cur_row, cur_col) + 1;
    end
    confusion_matrix
end

if questionFlag('7')
    % Parameters (fiddle around with these)
    try_part_7('building.jpeg', 1.0, 13, 4, 0.5, 0.00001);
    try_part_7('leaf.jpg', 1.0, 13, 4, 0.5, 0.001);
    try_part_7('chandelier.tiff', 1.0, 13, 4, 0.5, 0.001);
    try_part_7('climber.tiff', 3.0, 25, 2, 0.5, 0.00000001);
    try_part_7('tent.jpg', 2.0, 13, 4, 0.5, 0.0001);
end
    
end

function [] = try_part_7(building_im_name, gaussian_std_dev, gaussian_window_size, harris_window_size, k, threshold)

    building_im = imread(building_im_name);
    building_im_gray = int_to_dbl_img(rgb2gray(building_im));
    building_im_size = size(building_im_gray);
    
    gaussian_filter = gen_gaussian_filter(gaussian_std_dev, gaussian_window_size);

    finite_differences_x_filter = [1.0 -1.0;
                                   1.0 -1.0];
    finite_differences_y_filter = [1.0  1.0;
                                  -1.0 -1.0];
    smoothed_x_filter = conv2(finite_differences_x_filter, gaussian_filter);
    smoothed_y_filter = conv2(finite_differences_y_filter, gaussian_filter);
    
    building_x_derivative = conv2(building_im_gray, smoothed_x_filter);
    building_y_derivative = conv2(building_im_gray, smoothed_y_filter);
    building_derivative_size = size(building_x_derivative);
    
    building_xx = building_x_derivative .* building_x_derivative;
    building_xy = building_x_derivative .* building_y_derivative;
    building_yy = building_y_derivative .* building_y_derivative;
    clear building_x_derivative building_y_derivative;
    
    building_outer_products = zeros(building_derivative_size(1), building_derivative_size(2), 2, 2);
    building_outer_products(:, :, 1, 1) = building_xx;
    building_outer_products(:, :, 1, 2) = building_xy;
    building_outer_products(:, :, 2, 1) = building_xy;
    building_outer_products(:, :, 2, 2) = building_yy;
    clear building_xx building_xy building_yy;
    
    building_edge_criterion = zeros(building_derivative_size(1) + harris_window_size - 2, building_derivative_size(2) + harris_window_size - 2);
    building_edge_criterion_size = size(building_edge_criterion);
    for i = 1:building_edge_criterion_size(1)
        for j = 1:building_edge_criterion_size(2)
            height_range = max([(i - harris_window_size + 1) 1]):min([i building_derivative_size(1)]);
            width_range =  max([(j - harris_window_size + 1) 1]):min([j building_derivative_size(2)]);
            
            % Compute H matrix from gradient
            outer_product_sum = sum(sum(building_outer_products(height_range, width_range, :, :), 1), 2);
            
            outer_product_sum = permute(outer_product_sum, [3 4 1 2]);
            outer_product_sum = reshape(outer_product_sum, 2, 2);
            
            % Compute edge criterion
            building_edge_criterion(i, j) = det(outer_product_sum) - k * (trace(outer_product_sum) / 2.0) ^ 2;
        end
    end
    building_edge_criterion;
    
    % Find local maxima within threshold
    building_maxima = [];
    for i = 1:building_edge_criterion_size(1)
        for j = 1:building_edge_criterion_size(2)
            height_range = max([(i - 1) 1]):min([(i + 1) building_edge_criterion_size(1)]);
            width_range =  max([(j - 1) 1]):min([(j + 1) building_edge_criterion_size(2)]);
            if building_edge_criterion(i, j) == max(building_edge_criterion(height_range, width_range), [], 'all') && building_edge_criterion(i, j) > threshold
                building_edge_criterion(i, j);
                building_maxima = [building_maxima; i j building_edge_criterion(i, j)];
            end
        end
    end
    building_maxima_size = size(building_maxima);
    
    building_im_new = zeros(building_edge_criterion_size(1), building_edge_criterion_size(2), 3, 'uint8');
    building_offset_y = round((building_edge_criterion_size(1) - building_im_size(1)) / 2);
    building_offset_x = round((building_edge_criterion_size(2) - building_im_size(2)) / 2);
    building_im_new(building_offset_y:(building_offset_y + building_im_size(1) - 1), building_offset_x:(building_offset_x + building_im_size(2) - 1), :) = building_im;
    for i = 1:building_maxima_size(1)
        building_im_new = draw_line(building_im_new, building_maxima(i, 1:2), building_maxima(i, 1:2), 1, 255, 0, 0);
    end
    figure();
    imshow(building_im_new);
    imwrite(building_im_new, strcat(building_im_name, '_new.png'));
end


function [dbl_img] = int_to_dbl_img(int_img)
    dbl_img = double(int_img) / 255.0;
end


function [gaussian_filter] = gen_gaussian_filter(sigma, window)
    size_candidate = window;
    if mod(size_candidate, 2) == 0
        size_candidate = size_candidate + 1;
    end
    gaussian_filter = zeros(size_candidate);
    gaussian_filter_size = size(gaussian_filter);
    samples = (-(gaussian_filter_size(1) - 1) / 2):((gaussian_filter_size(1) - 1) / 2);
    [X1, X2] = meshgrid(samples, samples);
    mvn_z = mvnpdf([X1(:) X2(:)], [0 0], [(sigma .^ 2) 0.0; 0.0 (sigma .^ 2)]);

    gaussian_filter = reshape(mvn_z, size_candidate, size_candidate);
    gaussian_filter = gaussian_filter ./ sum(sum(gaussian_filter));
end