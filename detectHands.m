function [mask, face, person]= detectHands(rgbImage, oldmask, oldface, oldperson)
    % Ova funkcija nalazi sve regije koje sadrze ruke. Ruke se traze kao
    % delovi osobe koji su boje koze, a nisu glava. 
    
    if nargin > 4
        error('detectHands:TooManyInputs requires at most 3 optional inputs');
    end
    if nargin > 1
        has_old = 1;
    else
        has_old = 0;
    end
    
    % Detekcija lica i detekcija tela koje ce biti kasnije korisceni
    faceDetector = vision.CascadeObjectDetector();
    face   = step(faceDetector, rgbImage);
    
    face_size = size(face);
    
    if face_size(1) > 1
        if has_old
            face = oldface;
        else
            face = face(1,:);
        end
    else
        if face_size(1) < 1
            if has_old
                face = oldface;
            end
        end
    end
    
    [person, person_scores] = detectPerson(rgbImage);
    num_of_persons = size(person);
    num_of_persons = num_of_persons(1);
    
    if num_of_persons > 1
        if has_old
            person = oldperson;
        else
            person = person(1,:);
        end
    else
        if num_of_persons < 1
            if has_old
                person = oldperson;
            end
        end
    end
    
    % Zamucivanje slike radi otklanjanja suma
    w = fspecial('average', 3);
    blurred_frame = imfilter(rgbImage, w, 'replicate');
    mask_custom = createMask(blurred_frame);
    
    % Erozija i dilatacija radi otklanjanja kontura
    SE_erode = strel('square', 3);
    SE_dilate = strel('square', 5);
    mask_eroded = imerode(mask_custom, SE_erode);
    mask_dilated = imdilate(mask_eroded, SE_dilate);
    
    % Otklanjanje segmenta koji nisu ruke
    [labeledImage, numberOfBlobs] = bwlabel(mask_dilated);
    blobMeasurements = regionprops(labeledImage, 'area', 'Centroid');
    
    blobsToDropFace = zeros(1,numberOfBlobs);
    blobsToDropBody = zeros(1,numberOfBlobs);
    blobsToDropLegs = zeros(1,numberOfBlobs);
    
    for label = 1:numberOfBlobs
        currBlob = labeledImage == label;
        currBlobSize = size(currBlob);
        if sum(sum(sum(currBlob(face(2):face(2)+face(4),face(1):face(1)+face(3))))) > 0
            blobsToDropFace(label) = 1;
        end
        if sum(sum(sum(currBlob(person(2):min(person(2)+person(4), currBlobSize(1)),person(1):min(person(1)+person(3),currBlobSize(2)))))) == 0
            blobsToDropBody(label) = 1;
        end
        if abs(blobMeasurements(label).Centroid(2) - person(2) - person(4)) < abs(blobMeasurements(label).Centroid(2) - person(2) - person(4)*0.5)
            blobsToDropLegs(label) = 1;
        end
    end
    
    blobsToDrop = blobsToDropBody | blobsToDropFace | blobsToDropLegs;
    for label = 1:numberOfBlobs
        if blobsToDrop(label) == 1
            labeledImage(labeledImage == label) = 0;
        end
    end
    
    [labeledImage, numberOfBlobs] = bwlabel(labeledImage);
    mask = ExtractNLargestBlobs(labeledImage, 2);
    
end