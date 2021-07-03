function [bboxes, scores] = detectPerson(rgbImage)
    % Detekcija osobe na slici. Kako postoji vi?e metoda i kako ne daje
    % svaki metod rezultate, ideja iza ove funkcije je isprobavanje par
    % razli?itih funkcija sve dok se ne detektuja osoba. Ukoliko nijedan
    % metod ne na?e osobu, vratiti prazne liste
    
    % ACF inria
    [bboxes,scores] = detectPeopleACF(rgbImage);
    bboxes_size = size(bboxes);
    
    if(bboxes_size(2) ~= 4)
        % ACF caltech
        [bboxes,scores] = detectPeopleACF(rgbImage, 'Model', 'caltech-50x21');
        bboxes_size = size(bboxes);
        
        if(bboxes_size(2) ~= 4)
            % HoG detector
            peopleDetector = vision.PeopleDetector;
            peopleDetector.WindowStride = [4 4];
            [bboxes, scores] = step(peopleDetector, rgbImage);
        end
    end
    
end