clear
filename = "3.mp4";
v = VideoReader(filename);

% Number of frames
nfr = v.NumFrames; % Adjusted to handle corrupted frames

% Read frames
A = read(v, [1 nfr]);
B = squeeze(A(:,:,1,:));

% Initialize variables
offsetv = zeros(2, nfr);
oldimage = zeros(4600, 4600, 'uint8'); % Initialize large image buffer

outputVideo = VideoWriter('large.mp4','MPEG-4'); % uncomment to create movie
outputVideo.FrameRate = v.FrameRate; % Set the frame rate to 30 frames per second
open(outputVideo);

% Loop through frames for motion compensation
for nr = 2:nfr
    % Display progress
    disp(['Processing frame: ', num2str(nr), '/', num2str(nfr)]);

    % Estimate translation
    [row, col] = findTranslation(B(:, :, nr - 1), B(:, :, nr));
    offsetv(:, nr) = offsetv(:, nr - 1) + [row; col];

    % Compensate for motion and update image
    [oldimage, newimage] = compensateMotion(B(:, :, nr), offsetv(:, nr), oldimage);
    
    % Display compensated frame
    imshow(newimage);
    pause(0.1); % Adjust as needed
    
    % Resize frame to match expected size
    resizedFrame = imresize(newimage, [3200, 3200]);

    % Write compensated frame to video
    writeVideo(outputVideo, resizedFrame);
end

close(outputVideo);

function [row, col] = findTranslation(img1, img2)
    % Compute the 2D FFT of both images
    F1 = fft2(img1);
    F2 = fft2(img2);

    % Compute the cross-power spectrum
    C = (F1 .* conj(F2)) ./ abs(F1 .* conj(F2));

    % Compute the inverse 2D FFT of the cross-power spectrum
    inverseC = ifft2(C);

    % Find the peak location
    [~, index] = max(abs(inverseC(:)));
    [dy, dx] = ind2sub(size(inverseC), index);

    % Adjust for translations that are represented as large positive values
    if dx > size(img1, 2) / 2
        dx = dx - size(img1, 2);
    end
    if dy > size(img1, 1) / 2
        dy = dy - size(img1, 1);
    end
    
    % Adjust for translations that are represented as large negative values
    if dx < -size(img1, 2) / 2
        dx = dx + size(img1, 2);
    end
    if dy < -size(img1, 1) / 2
        dy = dy + size(img1, 1);
    end
    
    dx = dx - 1; % MATLAB uses 1-indexing
    dy = dy - 1;
    col = dx; % (treat image as matrix with rows and columns)
    row = dy;
end

function [oldimage, newimage] = compensateMotion(frame, offset, oldimage)
    % Apply translation to the frame and update old image buffer
    [rows, cols] = size(oldimage);
    
    drow = floor((rows - size(frame, 1)) / 2) + 1; % Position frame at the center vertically
    dcol = floor((cols - size(frame, 2)) / 2) + 1; % Position frame at the center horizontally
    
    % Adjust translation based on offset
    drow = drow + offset(1);
    dcol = dcol + offset(2) + 150; % Add an additional horizontal offset
    
    % Ensure that the frame fits within the bounds of the oldimage
    drow = max(1, min(drow, rows - size(frame, 1) + 1));
    dcol = max(1, min(dcol, cols - size(frame, 2) + 1));
    
    % Update the region of interest in the old image with the frame
    newimage = oldimage;
    newimage(drow:drow+size(frame, 1)-1, dcol:dcol+size(frame, 2)-1) = frame;
    oldimage = newimage;
end


