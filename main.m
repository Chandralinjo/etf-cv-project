% Ova skripta treba da demonstrira ceo flow, od u?itavanja fajla, preko
% detekcije osobe do prepoznavanja pokreta

%% U?itavanje fajla
file_path = '/Users/boris_majic/Downloads/trainning2/Sample00152/Sample00152_color.mp4';
file2 = '/Users/boris_majic/Downloads/trainning2/Sample00101/Sample00101_color.mp4';

% Pravimo VideoReader objekat koji ?e iterirati kroz frejmove
v = VideoReader(file2);

%% Detekcija ?oveka
% Detekciju ?oveka radimo na prvom frejmu. Pretpostavka je da je osoba
% prisutna u prvom frejmu.

frame = readFrame(v);
imshow(frame)

% figure()
% frame = imresize(frame, 0.5);
% imshow(frame)

% Za detekciju koristimo detectPeopleACF funkciju
[bboxes,scores] = detectPerson(frame);

I = insertObjectAnnotation(frame,'rectangle',bboxes,scores);
figure, imshow(I)
title('Detektovane osobe');

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


