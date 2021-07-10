function [ handImage , bounds ] = HandDetector( grayImage )

    binaryImage = im2bw(grayImage,graythresh(grayImage));
    %binaryImage = grayImage < 150;
    binaryImage = bwmorph(binaryImage,'majority',inf) ;


    labeledImage = bwlabel(binaryImage);
    measurements = regionprops(labeledImage, 'BoundingBox', 'Area');


    % Let's extract the first biggest blob - that will be the hand.
    allAreas = [measurements.Area];
    [sortedAreas, sortingIndexes] = sort(allAreas, 'descend');
    handIndex = sortingIndexes(1); 
    % Use ismember() to extact the hand from the labeled image.
    handImage = ismember(labeledImage, handIndex);
    % Now binarize
    handImage = handImage > 0;

    handImage = imfill(handImage,'holes');

    bounds = measurements(handIndex).BoundingBox;
    handImage = imcrop(handImage, bounds);
    %grayImage = imcrop(grayImage, bounds);
end