
% Load the wav file
[file, path] = uigetfile('*.wav', 'Select a wav file'); % Open file dialog to select wav file
if isequal(file,0)
   disp('User selected Cancel');
else
   disp(['User selected ', fullfile(path, file)]);
end

% Read the wav file
[audioData, sampleRate] = audioread(fullfile(path, file));

% Define segment duration in seconds
segmentDuration = 3; % 3 seconds

% Calculate the number of samples per segment
samplesPerSegment = segmentDuration * sampleRate;

% Get the total number of samples in the audio file
totalSamples = length(audioData);

% Define the segment file names in the required order
fileNames = {'flower.wav', 'frog.wav', 'bike.wav', 'car.wav', 'fork.wav', 'cinnamon.wav', 'pizza.wav'};

% Limit the number of segments to 7 (as per the specific requirement)
numSegments = min(7, ceil(totalSamples / samplesPerSegment));

% Loop through each segment and save with the specific names
for i = 1:numSegments
    % Calculate the start and end sample indices for the current segment
    startSample = (i - 1) * samplesPerSegment + 1;
    endSample = min(i * samplesPerSegment, totalSamples); % Make sure it doesn't exceed total samples
    
    % Extract the current segment
    audioSegment = audioData(startSample:endSample, :);
    
    % Save the segment with the specific name
    segmentFileName = fullfile(path, fileNames{i});
    audiowrite(segmentFileName, audioSegment, sampleRate);
    
    % Display a message
    disp(['Saved segment ', num2str(i), ' as ', fileNames{i}]);
end

disp('All segments saved with specified names.');
