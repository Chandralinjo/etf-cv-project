function mask = detectHands(rgbImage)
    % Ova funkcija nalazi sve regije koje sadr?e ?aku. ?ake se tra?e kao
    % delovi osobe koji su boje ko?e, a nisu glava. 
    
    
    % Najpre filtriramo slike radi uklanjanja ?uma.
    % Filtriranje vr?imo u svakoj boji
    rgbImage(:,:,1) = medfilt2(rgbImage(:,:,1), [3 3]);
    rgbImage(:,:,2) = medfilt2(rgbImage(:,:,2), [3 3]);
    rgbImage(:,:,3) = medfilt2(rgbImage(:,:,3), [3 3]);
    
    % Get the dimensions of the image.  numberOfColorBands should be = 3.
    [rows, columns, numberOfColorBands] = size(rgbImage);
    
    % Prelazimo u hsv prostor boja radi lak?e segmentacije
    hsv = rgb2hsv(rgbImage);
    h = hsv(:, :, 1);
    s = hsv(:, :, 2);
    v = hsv(:, :, 3);
    
    % Jedan pristup obele?avanju ko?e.
    hBinary = h < 0.07;
    sBinary = s > 0.25;
    vBinary = v > 0.57;
    skinPixels = hBinary & sBinary & vBinary;
    
    % Display them all.
    figure;
    subplot(2, 3, 1);
    imshow(skinPixels, []);
    title('Skin Pixels - ANDing of all binary images');
    subplot(2, 3, 2);
    imshow(hBinary, []);
    title('Hue Image Binarized');
    subplot(2, 3, 3);
    imshow(sBinary, []);
    title('Saturation Image Binarized');
    subplot(2, 3, 4);
    imshow(vBinary, []);
    title('Value Image Binarized');
    % Put up status bar so user can mouse around images and see pixel values.
    % Status bar will be in the lower left corner of the figure.
    hv = impixelinfo();
    % Mask the image.
    maskedRgbImage = bsxfun(@times, rgbImage, cast(skinPixels,class(rgbImage)));
    subplot(2,3, 5);
    imshow(maskedRgbImage);
    title('Skin Pixels in Color');
    maskedRgbImage = bsxfun(@times, rgbImage, cast(~skinPixels,class(rgbImage)));
    subplot(2,3, 6);
    imshow(maskedRgbImage);
    title('Non-Skin Pixels in Color');

    
    mask = skinPixels;
    
end