% Load the wav file
[file, path] = uigetfile('*.wav', 'Select a wav file'); % Open file dialog to select wav file
if isequal(file, 0)
   disp('User selected Cancel');
else
   disp(['User selected ', fullfile(path, file)]);
end

% Read the wav file
[audioData, sampleRate] = audioread(fullfile(path, file));

% Define total audio duration and segment duration in seconds
totalDuration = 30;  % Total 30 seconds
segmentDuration = 3; % 3 seconds per clip

% Calculate the number of samples per segment
samplesPerSegment = segmentDuration * sampleRate;

% Get the total number of samples needed for 30 seconds
totalSamples = totalDuration * sampleRate;

% If the audio is shorter than 30 seconds, pad with zeros
if length(audioData) < totalSamples
    audioData = [audioData; zeros(totalSamples - length(audioData), size(audioData, 2))];
else
    audioData = audioData(1:totalSamples, :); % Trim if longer than 30 seconds
end

% Define the segment file names in the required order
fileNames = {'flower.wav', 'frog.wav', 'bike.wav', 'car.wav', 'fork.wav', 'cinnamon.wav', 'pizza.wav', 'broccoli.wav', 'chair.wav', 'bear.wav'};

% Loop through each segment and save with the specific names
for i = 1:length(fileNames)
    % Calculate the start and end sample indices for the current segment
    startSample = (i - 1) * samplesPerSegment + 1;
    endSample = i * samplesPerSegment;
    
    % Extract the current segment
    audioSegment = audioData(startSample:endSample, :);
    
    % Save the segment with the specific name
    segmentFileName = fullfile(path, fileNames{i});
    audiowrite(segmentFileName, audioSegment, sampleRate);
    
    % Display a message
    disp(['Saved segment ', num2str(i), ' as ', fileNames{i}]);
end

disp('All segments saved with specified names.');

