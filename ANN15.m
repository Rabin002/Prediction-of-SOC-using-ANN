inputsTable = input_all_new5';
targetsTable = SOC_all_new3';
TraininputsTable=inputsTable(:,:);
TraintargetsTable=targetsTable(:,:);
TestinputsTable=inputsTable(:,2000000:end);
TesttargetsTable=targetsTable(:,2000000:end);
dataTrain = {TraininputsTable; TraintargetsTable};
dataTrain=dataTrain.';
dataTest={TestinputsTable; TesttargetsTable};
dataTest=dataTest.';
%X=input_all_new;
%Y=SOC_all_new;

numResponses = 1;
numFeatures = 4;
numHiddenUnits = 55;
Epochs = 5100;
LearnRateDropPeriod = 1000;
InitialLearnRate = 0.01;
LearnRateDropFactor = 0.1;
validationFrequency = 10;
repeat = 5;

layers = [
    sequenceInputLayer(numFeatures, 'Normalization', 'zerocenter')
    fullyConnectedLayer(numHiddenUnits)
    tanhLayer
    fullyConnectedLayer(numHiddenUnits)
    tanhLayer
    fullyConnectedLayer(numHiddenUnits)
    tanhLayer
    fullyConnectedLayer(numHiddenUnits)
    leakyReluLayer(0.3)
    fullyConnectedLayer(numResponses)
    clippedReluLayer(1)
    regressionLayer
 ];

options = trainingOptions('adam', ...
    'MaxEpochs', Epochs, ...
    'ExecutionEnvironment', 'gpu', ...  % Use GPU for training
    'InitialLearnRate', InitialLearnRate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', LearnRateDropPeriod, ...
    'LearnRateDropFactor', LearnRateDropFactor, ...
    'L2Regularization', 1, ...
    'ValidationData', {TestinputsTable,TesttargetsTable}, ... % Use testing data as validation data
    'Shuffle', 'never', ...
    'Verbose', 1, ...
    'ValidationFrequency', validationFrequency, ...
    'MiniBatchSize', 1, ...
    'Plots', 'training-progress');

%nets = cell(repeat, 1); % Preallocate cell array to store networks

%for i = 1:repeat
    nets10 = trainNetwork(TraininputsTable, TraintargetsTable, layers, options);

%end

