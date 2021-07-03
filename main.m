% Ova skripta treba da demonstrira ceo flow, od u?itavanja fajla, preko
% detekcije osobe do prepoznavanja pokreta

%% U?itavanje fajla
file_path = '/Users/boris_majic/Downloads/trainning2/Sample00152/Sample00152_color.mp4';
file2 = '/Users/boris_majic/Downloads/trainning2/Sample00101/Sample00101_color.mp4';

% Pravimo VideoReader objekat koji ?e iterirati kroz frejmove
v = VideoReader(file2);

%% Detekcija coveka
% Detekciju coveka radimo na prvom frejmu. Pretpostavka je da je osoba
% prisutna u prvom frejmu.

frame = readFrame(v);
imshow(frame)

% figure()
% frame = imresize(frame, 0.5);
% imshow(frame)

% Za detekciju koristimo napisanu funkciju koja povecava verovatnocu
% detekcije osobe. Ova funkcija ce isprobati tri razlicita pristupa
% detekciji osobe sekvnecijalno dok ne pronadje osobu
[person_bboxes, person_scores] = detectPerson(frame);

I = insertObjectAnnotation(frame,'rectangle',person_bboxes, person_scores);
figure, imshow(I)
title('Detektovane osobe');

%% Detekcija ko?e na osnovu detekcije lica

% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();

% Read a video frame and run the face detector.
% videoReader = VideoReader('/Users/boris_majic/Downloads/trainning2/Sample00152/Sample00152_color.mp4');
videoReader = VideoReader('/Users/boris_majic/Downloads/trainning2/Sample00101/Sample00101_color.mp4');
% videoReader = VideoReader('/Users/boris_majic/Downloads/trainning2/Sample00198/Sample00198_color.mp4');
videoFrame      = readFrame(videoReader);
face_bbox            = step(faceDetector, videoFrame);

% Draw the returned bounding box around the detected face.
videoFrame = insertShape(videoFrame, 'Rectangle', face_bbox);
figure; imshow(videoFrame); title('Detected face');


% Zamu?ivanje slike radi oktlanjanja ?uma
w = fspecial('average', 5);
videoFrame = imfilter(videoFrame, w, 'replicate');

face = videoFrame(face_bbox(2):face_bbox(2)+face_bbox(4),face_bbox(1):face_bbox(1)+face_bbox(3), :);
faceMeanColor = mean(mean(face));
faceMedianColor = double(median(median(face)));


figure()
meanMask = colorseg('euclidean', videoFrame, 40, faceMeanColor);
subplot(421), imshow(meanMask), title('EUC mean');
medianMask = colorseg('euclidean', videoFrame, 40, faceMedianColor);
subplot(422), imshow(medianMask), title('EUC median');

meanMask = colorseg('MAHALANOBIS', videoFrame, 40, faceMeanColor);
subplot(423), imshow(meanMask), title('EMAH mean');
medianMask = colorseg('MAHALANOBIS', videoFrame, 40, faceMedianColor);
subplot(424), imshow(medianMask), title('MAH median');

mask_YCbCr = detectSkinYCbCr(videoFrame);
subplot(425), imshow(mask_YCbCr), title('YCbCr detekcija');
mask_HSV = detectSkinHSV(videoFrame);
subplot(426), imshow(mask_HSV), title('HSV detekcija');
mask_custom = createMask(videoFrame);
subplot(427), imshow(mask_custom), title('Ru?no definisana maska');
mask_custom2 = createMask2(videoFrame);
subplot(428), imshow(mask_custom2), title('Ru?no definisana maska 2');


%% erozija i dilatacija maski


mask_custom = createMask(videoFrame);
SE_erode = strel('square', 5);
SE_dilate = strel('square', 7);
mask_eroded = imerode(mask_custom, SE_erode);
mask_dilated = imdilate(mask_eroded, SE_dilate);

figure()
subplot(311), imshow(mask_custom), title('Segmentacija');
subplot(312), imshow(mask_eroded), title('Erozija');
subplot(313), imshow(mask_dilated), title('Dilatacija');

%% Filtriranje segmenata

% Prelazimo iz slike u segmente
[labeledImage, numberOfBlobs] = bwlabel(mask_dilated);

% Za svaki segment racunamo centar mase i povrsinu
blobMeasurements = regionprops(labeledImage, 'area', 'Centroid');

% Odbacujemo one segmente koji se preklapaju sa licem i one koji su van
% tela
blobsToDropFace = zeros(1,numberOfBlobs);
blobsToDropBody = zeros(1,numberOfBlobs);
for label = 1:numberOfBlobs
    currBlob = labeledImage == label;
    if sum(sum(sum(currBlob(face_bbox(2):face_bbox(2)+face_bbox(4),face_bbox(1):face_bbox(1)+face_bbox(3))))) > 0
        blobsToDropFace(label) = 1;
    end
    if sum(sum(sum(currBlob(person_bboxes(2):person_bboxes(2)+person_bboxes(4),person_bboxes(1):person_bboxes(1)+person_bboxes(3))))) == 0
        blobsToDropBody(label) = 1;
    end
end

blobsToDrop = blobsToDropBody | blobsToDropFace;
for label = 1:numberOfBlobs
    if blobsToDrop(label) == 1
        labeledImage(labeledImage == label) = 0;
    end
end

imshow(labeledImage)

% Ostaju - ruke, ponekad vrat, noge, delovi ode?e
% Noge elimini?emo na osnovu njihove pozicije. Ako je centar mase bloba
% bli?i dnu osobe nego njenoj sredini, segment odbacujemo kao "noga"


[labeledImage, numberOfBlobs] = bwlabel(labeledImage);
blobMeasurements = regionprops(labeledImage, 'area', 'Centroid');

blobsToDropLegs = zeros(1,numberOfBlobs);
for label = 1:numberOfBlobs
    currBlob = labeledImage == label;
    if abs(blobMeasurements(label).Centroid(2) - person_bboxes(2) - person_bboxes(4)) < abs(blobMeasurements(label).Centroid(2) - person_bboxes(2) - person_bboxes(4)*0.5)
        blobsToDropLegs(label) = 1;
    end
end

for label = 1:numberOfBlobs
    if blobsToDropLegs(label) == 1
        labeledImage(labeledImage == label) = 0;
    end
end

[labeledImage, numberOfBlobs] = bwlabel(labeledImage);
blobMeasurements = regionprops(labeledImage, 'area', 'Centroid');

% Trebalo bi da su preostali segmenti ruke, eventualno vrat i delovi ode?e.
% Pretpostavljamo da su ruke najve?i preostali segmenti

hands_only = ExtractNLargestBlobs(labeledImage, 2);

figure()
subplot(121), imshow(videoFrame), title('Originalna slika');
subplot(122), imshow(hands_only), title('Izdvojene ruke');

%% Use PeopleDetector with video
peopleDetector = vision.PeopleDetector;
peopleDetector.WindowStride = [4 4];
peopleDetector.MinSize = [256 128];
video = vision.VideoFileReader('/Users/boris_majic/Downloads/trainning2/Sample00101/Sample00101_color.mp4');
viewer = vision.VideoPlayer;
while ~isDone(video)
    image = step(video);
    [bboxes, scores] = detectPerson(image);
    I_people = insertObjectAnnotation(image,'rectangle',bboxes,scores);
    step(viewer,I_people);
end

%% Izdvajanje osobe od pozadine
% https://www.mathworks.com/help/vision/ref/vision.foregrounddetector-system-object.html

detector = vision.ForegroundDetector('NumTrainingFrames', 25);

for i = 1:30
    frame = readFrame(v);  % ucitavanje sledeceg frejma
  
% Segmentacija osobe u prednji plan (foreground)
    foreground = step(detector, frame);
    figure(3),
    subplot(1,3,1), imshow(frame);
    title('Frejm');
    subplot(1,3,2), imshow(foreground); 
    title('Foreground');

% Koristimo morfoloske operacije za uklanjanje suma i popunjavanje praznina u otkrivenim objektima.
    se = strel('square', 1);
    
% Filtriranje prednjeg plana i prikaz slike sa cistim prednjim planom (clean foreground)
    filteredForeground = imopen(foreground, se);
    subplot(1,3,3), imshow(filteredForeground); 
    title('Clean Foreground');
end


