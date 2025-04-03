function [row,col] = findTranslation(img1, img2)
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
    dx = dx - 1;   % matlab uses 1-indexing
    dy = dy - 1;
    col = dx;    % (treat image as matrix with rows and columns)
    row = dy;
end
