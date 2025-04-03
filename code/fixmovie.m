clear
filename = "A_ENTRY_A_005.mp4";
v = VideoReader(filename);
%D = v.Duration
%v.NumFrames

nfr = 422   % skip end of the movie after frame 422, it is corrupted

A = read(v,[1 nfr]);
B = squeeze(A(:,:,1,:));
clear A
offsetv = zeros(2,nfr);

%load offsetv.mat

for nr = 2:nfr
    figure(2)
    imshow(B(:,:,nr));
    [row, col] = findTranslation(B(:,:,nr-1), B(:,:,nr));
    offsetv(:,nr) = offsetv(:,nr-1) + [row; col];
    %pause(0.1)
end

minrow = min(offsetv(1,:))
maxrow = max(offsetv(1,:))
mincol = min(offsetv(2,:))
maxcol = max(offsetv(2,:))

%%
currAxes = axes;
%outputVideo = VideoWriter('large.mp4','MPEG-4'); % uncomment to create movie
%outputVideo.FrameRate = 10; % Set the frame rate to 30 frames per second
%open(outputVideo);

oldimage = zeros(4600,4600,'uint8');  % choose a sufficiently large area to store results

for nr = 1:nfr
    nr
    drow = offsetv(1,nr)+1;     % make sure we stay inside output area
    dcol = offsetv(2,nr)+150;
    newimage = oldimage;
    newimage(drow:drow+1607, dcol:dcol+1607) = B(:,:,nr);
    image(newimage,"Parent",currAxes)
    currAxes.Visible = "off";
    colormap(gray(256));
    oldimage = newimage;
    %pause(0.1)

    %writeVideo(outputVideo, newimage);  % uncomment to create movie
end

%close(outputVideo);


