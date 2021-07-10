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
% videoReader = VideoReader('/Users/boris_majic/Downloads/trainning2/Sample00101/Sample00101_color.mp4');
videoReader = vision.VideoFileReader('/Users/boris_majic/Downloads/trainning2/Sample00198/Sample00198_color.mp4', 'VideoOutputDataType', 'uint8');
videoFrame  = step(videoReader);
face_bbox   = step(faceDetector, videoFrame);

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

%% Izdvajanje obele?ja za pra?enje
videoReader = vision.VideoFileReader('/Users/boris_majic/Downloads/trainning2/Sample00198/Sample00198_color.mp4', 'VideoOutputDataType', 'uint8');
videoFrame = step(videoReader);

% Razlikovanje leve i desne ruke
[labeledImage, numberOfBlobs] = bwlabel(hands_only);
blobMeasurements = regionprops(labeledImage, 'area', 'Centroid', 'BoundingBox', 'PixelList');

if blobMeasurements(1).Centroid(1) > blobMeasurements(2).Centroid(1)
    leva_ruka = 2;
    desna_ruka = 1;
else
    leva_ruka = 1;
    desna_ruka = 2;
end

lpoints = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', blobMeasurements(leva_ruka).BoundingBox);
rpoints = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', blobMeasurements(desna_ruka).BoundingBox);

% pointImage = insertMarker(videoFrame, lpoints.Location, '+', 'Color', 'blue');
% pointImage = insertMarker(pointImage, rpoints.Location, '+', 'Color', 'red');
% figure, imshow(pointImage), title('Detektovane ta?ke od interesa');

%% Pracenje ruku
videoPlayer = vision.VideoPlayer('Position', [100,100,680,520]);

rtracker = vision.PointTracker('MaxBidirectionalError', 5);
ltracker = vision.PointTracker('MaxBidirectionalError', 5);

initialize(rtracker, rpoints.Location, videoFrame);
initialize(ltracker, lpoints.Location, videoFrame);

i = 0;
while ~isDone(videoReader) && i < 15
    frame = step(videoReader);
    [lpoints, lvalidity] = step(ltracker, frame);
    [rpoints, rvalidity] = step(rtracker, frame);
    
    out = insertMarker(frame, rpoints(rvalidity, :), '+', 'Color', 'red');
    out = insertMarker(out, lpoints(lvalidity, :), '+', 'Color', 'blue');
    
    step(videoPlayer, out);
    imwrite(out, ['KLT_' num2str(i) '.png'])
    i = i + 1;
end

%% Resetovanje player-a i trackera
release(videoPlayer);
release(videoReader);

% videoReader = vision.VideoFileReader('/Users/boris_majic/Downloads/trainning2/Sample00152/Sample00152_color.mp4', 'VideoOutputDataType', 'uint8');
 videoReader = vision.VideoFileReader('/Users/boris_majic/Downloads/trainning2/Sample00101/Sample00101_color.mp4', 'VideoOutputDataType', 'uint8');

%videoReader = vision.VideoFileReader('/Users/boris_majic/Downloads/trainning2/Sample00198/Sample00198_color.mp4', 'VideoOutputDataType', 'uint8');
videoPlayer = vision.VideoPlayer('Position', [100,100,680,520]);

%% Pracenje ruku onlajn detektorom

i = 0;
while ~isDone(videoReader) && i < 45
    
    frame = step(videoReader);
    hsv = rgb2hsv(frame);
    snapshot = hsv(:,:,3);
    
    [bw, bnds] = HandDetector(snapshot);
    
    % Prostor za izvla?enje obele?ja
    [labeledImage, numberOfBlobs] = bwlabel(hands);
    blobMeasurements = regionprops(labeledImage, 'area', 'Centroid', 'BoundingBox', 'PixelList');
    
    
    step(videoPlayer, hands);
    % imwrite(out, ['KLT_' num2str(i) '.png'])
    i = i + 1;
end

%% Pokusaj rada sa custom detectorom

lhand_positions = [];
rhand_positions = [];
face_positions = [];
lhand_areas = [];
rhand_areas = [];

i = 0;
while ~isDone(videoReader) && i < 60
    
    frame = step(videoReader);

    if i == 0
        [hands, face, person] = detectHands(frame);
        [labeledImage, numberOfBlobs] = bwlabel(hands);
        blobMeasurements = regionprops(labeledImage, 'area', 'Centroid', 'BoundingBox', 'PixelList');
        if blobMeasurements(1).Centroid(1) > blobMeasurements(2).Centroid(1)
            leva_ruka = 2;
            desna_ruka = 1;
        else
            leva_ruka = 1;
            desna_ruka = 2;
        end
        leftLabeled = labeledImage == leva_ruka;
        rightLabeled = labeledImage == desna_ruka;
        
    else
        [new_hands, new_face, new_person] = detectHands(frame, hands, face, person);
        [newlabeledImage, newnumberOfBlobs] = bwlabel(hands);
        newblobMeasurements = regionprops(labeledImage, 'area', 'Centroid', 'BoundingBox', 'PixelList');
        if newblobMeasurements(1).Centroid(1) > newblobMeasurements(2).Centroid(1)
            new_leva_ruka = 2;
            new_desna_ruka = 1;
        else
            new_leva_ruka = 1;
            new_desna_ruka = 2;
        end
        
        rAreaChange = abs(newblobMeasurements(new_desna_ruka).Area - blobMeasurements(desna_ruka).Area)/blobMeasurements(desna_ruka).Area;
        lAreaChange = abs(newblobMeasurements(new_leva_ruka).Area - blobMeasurements(leva_ruka).Area)/blobMeasurements(desna_ruka).Area;
        rCentroidChange = sqrt(sum((newblobMeasurements(new_desna_ruka).Centroid - blobMeasurements(desna_ruka).Centroid).^2));
        lCentroidChange = sqrt(sum((newblobMeasurements(new_leva_ruka).Centroid - blobMeasurements(leva_ruka).Centroid).^2));
        
        new_right = rAreaChange < 0.5 && rCentroidChange < 50;
        new_left = lAreaChange < 0.5 && lCentroidChange < 50;
        
        face = new_face;
        person = new_person;
        
        if new_right && new_left
            hands = new_hands;
            rightLabeled = newlabeledImage == new_desna_ruka;
            leftLabeled = newlabeledImage == new_leva_ruka;
        else
            if new_right
                rightLabeled == newlabeledImage == new_desna_ruka;
                hands = rightLabeled | leftLabeled;
            else
                if new_left
                    leftLabeled == newlabeledImage == new_leva_ruka;
                    hands = rightLabeled | leftLabeled;
                end
            end
        end
    end
    
    % Prostor za izvla?enje obele?ja
    [labeledImage, numberOfBlobs] = bwlabel(hands);
    blobMeasurements = regionprops(labeledImage, 'area', 'Centroid', 'BoundingBox', 'PixelList');
    
    if blobMeasurements(1).Centroid(1) > blobMeasurements(2).Centroid(1)
        leva_ruka = 2;
        desna_ruka = 1;
    else
        leva_ruka = 1;
        desna_ruka = 2;
    end
    
    face_positions = [face_positions; face];
    lhand_positions = [lhand_positions; blobMeasurements(leva_ruka).Centroid];
    rhand_positions = [rhand_positions; blobMeasurements(desna_ruka).Centroid];
    lhand_areas = [lhand_areas, blobMeasurements(leva_ruka).Area];
    rhand_areas = [rhand_areas, blobMeasurements(desna_ruka).Area];
    
    
    step(videoPlayer, hands);
    if mod(i,5) == 0
        imwrite(hands, ['segment_track_' num2str(i) '.png'])
    end
    
    i = i + 1;
end

%% Pracenje slike rastom regiona

% Na prvom frejmu izvlacimo ruke:
frame = step(videoReader);
[hands, face, person] = detectHands(frame);

[labeledImage, numberOfBlobs] = bwlabel(hands);
blobMeasurements = regionprops(labeledImage, 'area', 'Centroid', 'BoundingBox', 'PixelList');




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

%% Ucitavanje podataka
folderi = folderList('/Users/boris_majic/Downloads/trainning2');
num_of_samples = length(folderi);

for data_counter=1:num_of_samples
    curr_path = [folderi(data_counter).folder '/' folderi(data_counter).name '/' folderi(data_counter).name];
    data_files(data_counter) = load([curr_path '_data.mat']);
end

%% Pravljenje liste segmenata. 
% Jedan segment sadr?i informaciju o klasi, o kom video se radi i o kojim
% frjmovima

labels = {'cheduepalle';'fame';'chevuoi';'basta'};

segment_counter = 1;
for data_counter=1:num_of_samples
    currVideo = data_files(data_counter).Video;
    for label_counter = 1: length(currVideo.Labels)
        if ismember(currVideo.Labels(label_counter).Name, labels)
            % Treba dodati isecak u listu struktura
            s = struct('videoID', data_counter, 'start',currVideo.Labels(label_counter).Begin, 'end', currVideo.Labels(label_counter).End, 'label', currVideo.Labels(label_counter).Name);
            segment_list(segment_counter) = s;
            segment_counter = segment_counter + 1;
        end
    end
end

%% Dohvatanje pozicija delova tela od interesa

for idx = 1:length(segment_list)
    curr_face = [];
    curr_lhand = [];
    curr_rhand = [];
    for Frameidx = segment_list(idx).start:segment_list(idx).end
        curr_face = [curr_face; data_files(segment_list(idx).videoID).Video.Frames(Frameidx).Skeleton.PixelPosition(4,:)];
        curr_lhand = [curr_lhand; data_files(segment_list(idx).videoID).Video.Frames(Frameidx).Skeleton.PixelPosition(8,:)];
        curr_rhand = [curr_rhand; data_files(segment_list(idx).videoID).Video.Frames(Frameidx).Skeleton.PixelPosition(12,:)];
    end
    segment_list(idx).FaceArray = curr_face;
    segment_list(idx).lHandArray = curr_lhand;
    segment_list(idx).rHandArray = curr_rhand;
end

%% relativne pozicije ruku u odnosu na lice

for idx = 1:length(segment_list)
    segment_list(idx).lHandPos = segment_list(idx).FaceArray - segment_list(idx).lHandArray;
    segment_list(idx).rHandPos = segment_list(idx).FaceArray - segment_list(idx).rHandArray;
end

%% Kreiranje obelezja
for idx = 1:length(segment_list)
    % min LHand
    minL = min(segment_list(idx).lHandPos);
    segment_list(idx).minLX = minL(1);
    segment_list(idx).minLY = minL(2);
    
    % max LHand
    maxL = max(segment_list(idx).lHandPos);
    segment_list(idx).maxLX = maxL(1);
    segment_list(idx).maxLY = maxL(2);
    
    % min RHand
    minR = min(segment_list(idx).rHandPos);
    segment_list(idx).minRX = minR(1);
    segment_list(idx).minRY = minR(2);
    
    % max RHand
    maxR = max(segment_list(idx).rHandPos);
    segment_list(idx).maxRX = maxR(1);
    segment_list(idx).maxRY = maxR(2);
    
    % mean LHand
    meanL = mean(segment_list(idx).lHandPos);
    segment_list(idx).meanLX = meanL(1);
    segment_list(idx).meanLY = meanL(2);
    
    % mean RHand
    meanR = mean(segment_list(idx).rHandPos);
    segment_list(idx).meanRX = meanR(1);
    segment_list(idx).meanRY = meanR(2);
    
    % brojPromeneSmera kretanja leve ruke - X i Y
    N_promena_X = 0;
    N_promena_Y = 0;
    
    curr_direction_X = sign(segment_list(idx).lHandPos(2,1) - segment_list(idx).lHandPos(1,1));
    curr_direction_Y = sign(segment_list(idx).lHandPos(2,2) - segment_list(idx).lHandPos(1,2));
    curr_segment_size = size(segment_list(idx).lHandPos);
    for posidx = 2:curr_segment_size(1)-1
        new_direction_X = sign(segment_list(idx).lHandPos(posidx + 1,1) - segment_list(idx).lHandPos(posidx,1));
        new_direction_Y = sign(segment_list(idx).lHandPos(posidx + 1,2) - segment_list(idx).lHandPos(posidx,2));
        if new_direction_X * curr_direction_X < 1
            curr_direction_X = new_direction_X;
            N_promena_X = N_promena_X + 1;
        end
        if new_direction_Y * curr_direction_Y < 1
            curr_direction_Y = new_direction_Y;
            N_promena_Y = N_promena_Y + 1;
        end
    end
    
    segment_list(idx).numLX = N_promena_X;
    segment_list(idx).numLY = N_promena_Y;
    
    % brojPromeneSmera kretanja desne ruke - X i Y
    N_promena_X = 0;
    N_promena_Y = 0;
    
    curr_direction_X = sign(segment_list(idx).rHandPos(2,1) - segment_list(idx).rHandPos(1,1));
    curr_direction_Y = sign(segment_list(idx).rHandPos(2,2) - segment_list(idx).rHandPos(1,2));
    curr_segment_size = size(segment_list(idx).rHandPos);
    for posidx = 2:curr_segment_size(1)-1
        new_direction_X = sign(segment_list(idx).rHandPos(posidx + 1,1) - segment_list(idx).rHandPos(posidx,1));
        new_direction_Y = sign(segment_list(idx).rHandPos(posidx + 1,2) - segment_list(idx).rHandPos(posidx,2));
        if new_direction_X * curr_direction_X < 1
            curr_direction_X = new_direction_X;
            N_promena_X = N_promena_X + 1;
        end
        if new_direction_Y * curr_direction_Y < 1
            curr_direction_Y = new_direction_Y;
            N_promena_Y = N_promena_Y + 1;
        end
    end
    
    segment_list(idx).numRX = N_promena_X;
    segment_list(idx).numRY = N_promena_Y;
    
    % Rad sa brzinama 
    speedLX = [];
    speedLY = [];
    speedRX = [];
    speedRY = [];
    for posidx = 1:curr_segment_size(1)-1
        speedLX = [speedLX; segment_list(idx).lHandPos(posidx + 1,1) - segment_list(idx).lHandPos(posidx,1)];
        speedLY = [speedLY; segment_list(idx).lHandPos(posidx + 1,2) - segment_list(idx).lHandPos(posidx,2)];
        speedRX = [speedLY; segment_list(idx).rHandPos(posidx + 1,1) - segment_list(idx).rHandPos(posidx,1)];
        speedRY = [speedRY; segment_list(idx).rHandPos(posidx + 1,2) - segment_list(idx).rHandPos(posidx,2)];
    end
    
    segment_list(idx).speedLX = mean(speedLX);
    segment_list(idx).speedLY = mean(speedLY);
    segment_list(idx).speedRX = mean(speedRX);
    segment_list(idx).speedRY = mean(speedRY);
    
    segment_list(idx).speedsdLX = std(speedLX);
    segment_list(idx).speedsdLY = std(speedLY);
    segment_list(idx).speedsdRX = std(speedRX);
    segment_list(idx).speedsdRY = std(speedRY);
    
end

%% Transform labels to a categorical

y = cell(length(segment_list), 1);
for idx = 1:length(segment_list)
    y{idx} = segment_list(idx).label;
end
y = categorical(y);

%% Tranfosrm features to a matrix

X = zeros(length(segment_list), 24);
for idx = 1:length(segment_list)
    X(idx, 1) = segment_list(idx).minLX;
    X(idx, 2) = segment_list(idx).minLY;
    X(idx, 3) = segment_list(idx).minRX;
    X(idx, 4) = segment_list(idx).minRY;
    X(idx, 5) = segment_list(idx).maxLX;
    X(idx, 6) = segment_list(idx).maxLY;
    X(idx, 7) = segment_list(idx).maxRX;
    X(idx, 8) = segment_list(idx).maxRY;
    X(idx, 9) = segment_list(idx).meanLX;
    X(idx, 10) = segment_list(idx).meanLY;
    X(idx, 11) = segment_list(idx).meanRX;
    X(idx, 12) = segment_list(idx).meanRY;
    X(idx, 13) = segment_list(idx).numLX;
    X(idx, 14) = segment_list(idx).numLY;
    X(idx, 15) = segment_list(idx).numRX;
    X(idx, 16) = segment_list(idx).numRY;
    X(idx, 17) = segment_list(idx).speedLX;
    X(idx, 18) = segment_list(idx).speedLY;
    X(idx, 19) = segment_list(idx).speedRX;
    X(idx, 20) = segment_list(idx).speedRY;
    X(idx, 21) = segment_list(idx).speedsdLX;
    X(idx, 22) = segment_list(idx).speedsdLY;
    X(idx, 23) = segment_list(idx).speedsdRX;
    X(idx, 24) = segment_list(idx).speedsdRY;
end

%% feature standardization

X = (X - mean(X))./std(X);

%% Split data into training and testing sets
PD = 0.2;

cv = cvpartition(size(X,1),'HoldOut',PD);
Xtrain = X(cv.training,:);
Xtest = X(cv.test,:);
ytrain = y(cv.training);
ytest = y(cv.test);

%% train linear classiffier (logReg)

t = templateLinear('Lambda', 0.01);
linearMdl = fitcecoc(Xtrain, ytrain, 'Learners', t);
ypredlg = predict(linearMdl, Xtest);
confusionmat(ytest, ypredlg)

%% train linear classiffier (logReg)

t = templateLinear('Lambda', 0.1);
linearMdl = fitcecoc(Xtrain, ytrain, 'Learners', t);
ypredlg = predict(linearMdl, Xtest);
confusionmat(ytest, ypredlg)

%% train linear classiffier (logReg)

t = templateLinear('Lambda', 1);
linearMdl = fitcecoc(Xtrain, ytrain, 'Learners', t);
ypredlg = predict(linearMdl, Xtest);
confusionmat(ytest, ypredlg)

%% train linear classiffier (logReg)

t = templateLinear('Lambda', 10);
linearMdl = fitcecoc(Xtrain, ytrain, 'Learners', t);
ypredlg = predict(linearMdl, Xtest);
confusionmat(ytest, ypredlg)

%% Train SVM
t = templateSVM('KernelFunction','gaussian');
Mdl = fitcecoc(Xtrain, ytrain, 'Learners', t);
ypredsvm = predict(Mdl, Xtest);
confusionmat(ytest, ypredsvm)

%% Train SVM
t = templateSVM('KernelFunction','polynomial','PolynomialOrder',2);
Mdl = fitcecoc(Xtrain, ytrain, 'Learners', t);
ypredsvm = predict(Mdl, Xtest);
confusionmat(ytest, ypredsvm)

%% Train SVM
t = templateSVM('KernelFunction','polynomial','PolynomialOrder',4);
Mdl = fitcecoc(Xtrain, ytrain, 'Learners', t);
ypredsvm = predict(Mdl, Xtest);
confusionmat(ytest, ypredsvm)

%% 
cv = cvpartition(size(X,1),'HoldOut',PD);
Xtrain = X(cv.training,:);
Xtest = X(cv.test,:);
ytrain = y(cv.training);
ytest = y(cv.test);

t = templateSVM('KernelFunction','polynomial','PolynomialOrder',2);
Mdl = fitcecoc(Xtrain, ytrain, 'Learners', t);
ypredsvm = predict(Mdl, Xtest);
confusionmat(ytest, ypredsvm)

