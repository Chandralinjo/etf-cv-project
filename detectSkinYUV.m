function res = detectSkinYUV(rgb)    

    r = rgb(:, :, 1);
    g = rgb(:, :, 2);
    b = rgb(:, :, 3);

    Y = r + 2*g + b/4;
    U = r - g;
    V = b - g;
    
    yuv = cat(3, Y, U, V);
    
    res = double(yuv(:,:,2)>20 & yuv(:,:,2)<=74);
end