function dfolders = folderList(folder_name)
    % get the folder contents
    d = dir(folder_name);
    % remove all files (isdir property is 0)
    dfolders = d([d(:).isdir]); 
    % remove '.' and '..' 
    dfolders = dfolders(~ismember({dfolders(:).name},{'.','..'}));
end