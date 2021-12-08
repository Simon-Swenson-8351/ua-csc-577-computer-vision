% #1

rng(477);
sensor_sensitivities = importdata('rgb_sensors.txt');

w = 40;
h = 40;
num_samples = w * h;
sensor_sensitivities_size = size(sensor_sensitivities);
num_sensor_bins = sensor_responses_size(1);

random_spectra = rand(num_samples, num_sensor_bins);
random_responses = random_spectra * sensor_sensitivities;

max_rgb = max(max(random_responses));
k = 255 / max_rgb;
random_spectra = random_spectra .* k;
random_responses = random_spectra * sensor_sensitivities;
max(max(random_responses))

im_small = zeros(h, w, 3, 'uint8');
im_scalar = 10;
im_big = zeros(h * im_scalar, w * im_scalar, 3, 'uint8');
for i = 1:h
    for j = 1:w
        im_small(i, j, :) = round(random_responses((i - 1) * 40 + j, :));
        im_big(((i - 1) * im_scalar + 1):(i * im_scalar), ((j - 1) * im_scalar + 1):(j * im_scalar), :) = repelem(im_small(i, j, :), im_scalar, im_scalar);
    end
end

figure();
imshow(im_big);

% #2
% U * x = y, where U has a lot of rows.
% Best inverse we can do is the pseudo invserse, U^cross = (U^T * U)^-1 * U^T
% so x \approx U^cross * y
% Note we're pretending sensor responses are our unknown, so x U represents
% the data, x is the sensor responses, and y is the output.
rs_pseudoinv = pseudoinv(random_spectra);
predicted_sensors = rs_pseudoinv * random_responses;

plot_for_all_channel_diffs(sensor_sensitivities, predicted_sensors);
no_noise_rmse = rmse(sensor_sensitivities, predicted_sensors)

fake_responses = random_spectra * predicted_sensors;
data_no_noise_rmse = rmse(random_responses, fake_responses)

% #3
noise = (randn(num_samples, 3) .* 10);
noisy_output = random_responses + noise;

predicted_sensors = rs_pseudoinv * noisy_output;

plot_for_all_channel_diffs(sensor_sensitivities, predicted_sensors);
noisy_rmse = rmse(sensor_sensitivities, predicted_sensors)

random_noisy_responses = random_spectra * predicted_sensors;
noisy_data_rmse = rmse(random_responses, random_noisy_responses)


noisy_output(find(noisy_output < 0)) = 0;
noisy_output(find(noisy_output > 255)) = 255;

predicted_sensors = rs_pseudoinv * noisy_output;

plot_for_all_channel_diffs(sensor_sensitivities, predicted_sensors);
noisy_clipped_rmse = rmse(sensor_sensitivities, predicted_sensors)

random_noisy_clipped_responses = random_spectra * predicted_sensors;
noisy_data_clipped_rmse = rmse(random_responses, random_noisy_clipped_responses)

% #4
for i = 0:10
    do_4(i, num_samples, random_spectra, sensor_sensitivities);
end

% #5
% Gamma function: f(x) = 255 * (x / 255) ^ (1 / g)
% We want to use the gamma function so that a value of 80 will get pushed 
% to a value of 127. Thus we want to solve for g in the following:
% 127 = 255 * (80 / 255) ^ (1 / g)
% log(127 / 255) = (1 / g) log(80 / 255)
% g = log(80 / 255) / log(127 / 255)
g = log(80 / 255) / log(127 / 255)

% #6
% Wavelength across column, samples down row
real_light_spectra = importdata('light_spectra.txt');
real_light_spectra_size = size(real_light_spectra)
real_rgb = importdata('responses.txt');
real_rgb_size = size(real_rgb)
% using original sensors (sensor_sensitivities)
sensor_sensitivities_pred = pseudoinv(real_light_spectra) * real_rgb;
plot_for_all_channel_diffs(sensor_sensitivities, sensor_sensitivities_pred);
ss_real_data_rmse = sqrt(sum(rmse(sensor_sensitivities, sensor_sensitivities_pred) .^ 2) / 3)

real_rgb_pred = real_light_spectra * sensor_sensitivities_pred;
rrgb_rmse = sqrt(sum(rmse(real_rgb, real_rgb_pred) .^ 2) / 3)

% #7
% x'(U'U)x - 2y'Ux + y'y              subject to x     >  0
% 
% Let     H = U'U
%         f' = -y'U
%         A = -I
%         b = 0
% 
% 
% proportional to:
%           x' * H * x + 2 * f' * x       subject to A * x <= 0
% 
% min 0.5 * x' * H * x +     f' * x       subject to A * x <= b

% Have to do this one one channel at a time, otherwise quadprog was whining
% at me.
constrained_sensor_sensitivities_pred = zeros(real_light_spectra_size(2), 3);
for channel = 1:3
    H = real_light_spectra' * real_light_spectra;
    f = -real_light_spectra' * real_rgb(:, channel);
    A = -1 * eye(real_light_spectra_size(2));
    b = zeros(real_light_spectra_size(2), 1);
    constrained_sensor_sensitivities_pred(:, channel) = quadprog(H, f, A, b);
end
plot_for_all_channel_diffs(sensor_sensitivities, constrained_sensor_sensitivities_pred);

% #8
real_rgb_aug = [real_rgb; zeros(real_light_spectra_size(2) - 1, 3)];
M = zeros(real_light_spectra_size(2) - 1, real_light_spectra_size(2));
for i = 1:(real_light_spectra_size(2) - 1)
    M(i, i) = 1;
    M(i, i+1) = -1;
end

% Extra to see if the real rgbs actually make a picture or anything
% Turns out it's 2 * 13 * 23. Let's try 23 x 26.
proposed_img = zeros(26, 23, 3, 'uint8');
for i = 1:598
    proposed_img(floor((i - 1) / 23) + 1, mod(i - 1, 23) + 1, :) = round(real_rgb(i, :));
end
size(proposed_img)
figure();
imshow(proposed_img);

for lambda = [0.0001, 0.001, 0.01, 0.1, 1.0]
    M_cur = lambda * M;
    real_light_spectra_aug = [real_light_spectra; M_cur];
    constrained_smooth_sensor_sensitivities_pred = zeros(real_light_spectra_size(2), 3);
    for channel = 1:3
        H = real_light_spectra_aug' * real_light_spectra_aug;
        f = -real_light_spectra_aug' * real_rgb_aug(:, channel);
        A = -1 * eye(real_light_spectra_size(2));
        b = zeros(real_light_spectra_size(2), 1);
        constrained_smooth_sensor_sensitivities_pred(:, channel) = quadprog(H, f, A, b);
    end
    plot_for_all_channel_diffs(sensor_sensitivities, constrained_smooth_sensor_sensitivities_pred);
end

% Assumes parameters are two-dimensional with |second dimension| = 3.
function [] = plot_for_all_channel_diffs(actual, guessed)
    plot_diffs(actual(:, 1), guessed(:, 1));
    plot_diffs(actual(:, 2), guessed(:, 2));
    plot_diffs(actual(:, 3), guessed(:, 3));
end

function [] = plot_diffs(actual, guessed)
    figure();
    plot(actual, 'b');
    hold on;
    plot(guessed, 'r');
end

% Assumes each row is a data point
function [out] = rmse(actual, guesses)
    actual_size = size(actual);
    actual_rows = actual_size(1);
    % Sum each row together.
    out = sqrt(sum((actual - guesses) .^ 2) ./ actual_rows);
end

function [out] = pseudoinv(in)
    out = (in' * in)^(-1) * in';
end

% all values in random_spectra * sensor_sensitivities are assumed to be
% in interval [0, 255].
function [] = do_4(i, num_samples, random_spectra, sensor_sensitivities)
    i = i
    colors = random_spectra * sensor_sensitivities;
    rs_pseudoinv = pseudoinv(random_spectra);
    
    noise = randn(num_samples, 3) .* (10 * i);
    noisy_output = colors + noise;
    
    % sensor error, without clipping
    predicted_sensors = rs_pseudoinv * noisy_output;
    sewoc_rmse = sqrt(sum(rmse(sensor_sensitivities, predicted_sensors) .^ 2) / 3)
    
    % response error, without clipping
    predicted_responses = random_spectra * predicted_sensors;
    rewoc_rmse = sqrt(sum(rmse(colors, predicted_responses) .^ 2) / 3)
    
    % sensor error, with clipping
    noisy_output(find(noisy_output < 0)) = 0;
    noisy_output(find(noisy_output > 255)) = 255;
    predicted_sensors_clipped = rs_pseudoinv * noisy_output;
    rewc_rmse = sqrt(sum(rmse(sensor_sensitivities, predicted_sensors_clipped) .^ 2) / 3)
    
    % response error, with clipping
    predicted_responses_clipped = random_spectra * predicted_sensors_clipped;
    rewc_rmse = sqrt(sum(rmse(colors, predicted_responses_clipped) .^ 2) / 3)
    
    if i == 5 || i == 10
        plot_for_all_channel_diffs(sensor_sensitivities, predicted_sensors);
        plot_for_all_channel_diffs(sensor_sensitivities, predicted_sensors_clipped);
    end
end