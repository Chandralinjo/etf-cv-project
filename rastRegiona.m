% ideje za rast regiona

% Pra?enje regiona pomo?u rasta regiona
% Sejanje vr?imo nala?enjem piksela ruke koji je najbli?i prose?noj boji
% regiona

hsvFrame = rgb2hsv(videoFrame);

% Prvo racunamo prosecnu boju leve_ruke
hsv_sum = double(zeros(1,1,3));
pixel_sum = double(zeros(1,1,3));
for k = 1:length(blobMeasurements(leva_ruka).PixelList)
    curr_pixel = blobMeasurements(leva_ruka).PixelList(k,:);
    pixel_sum = pixel_sum + double(videoFrame(curr_pixel(1), curr_pixel(2), :));
    hsv_sum = hsv_sum + double(hsvFrame(curr_pixel(1), curr_pixel(2), :));
end

mean_lhand_pixel = pixel_sum / length(blobMeasurements(leva_ruka).PixelList);
mean_lhand_hsv = hsv_sum / length(blobMeasurements(leva_ruka).PixelList);

pixel_sum = double(zeros(1,1,3));
hsv_sum = double(zeros(1,1,3));
for k = 1:length(blobMeasurements(desna_ruka).PixelList)
    curr_pixel = blobMeasurements(desna_ruka).PixelList(k,:);
    pixel_sum = pixel_sum + double(videoFrame(curr_pixel(1), curr_pixel(2), :));
    hsv_sum = hsv_sum + double(hsvFrame(curr_pixel(1), curr_pixel(2), :));
end
mean_dhand_pixel = pixel_sum / length(blobMeasurements(desna_ruka).PixelList);
mean_dhand_hsv = hsv_sum / length(blobMeasurements(leva_ruka).PixelList);

% Nalazimo u obe ruke piksele koji su najbli?i
