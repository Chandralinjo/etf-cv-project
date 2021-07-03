function mask = detectSkinYCbCr(rgbImage)
    ycbcr = rgb2ycbcr(rgbImage);
    mask = double(ycbcr(:,:,2)>=77 & ycbcr(:,:,2)<=127 & ycbcr(:,:,3)>=139 & ycbcr(:,:,3)<=210); 
end