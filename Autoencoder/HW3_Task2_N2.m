%% Network 1 Autoencoder
addpath('D:\Masters Program Chalmers\Projects and Labs\ANN\')
clear vars
%%
exerciseNumber = 3;
[trainTrainImage, trainTrainLabel, trainValidImage, trainValidLabel,...
    testImage, testLabel] = LoadMNIST(exerciseNumber);

trainTrainImage = trainTrainImage/255;
trainValidImage = trainValidImage/255;
testImage = testImage/255;

trainTrainImageStacked = reshape(trainTrainImage, [28*28, length(trainTrainImage)]);
trainValidImageStacked = reshape(trainValidImage, [28*28, length(trainValidImage)]);
testImageStacked = reshape(testImage, [28*28, length(testImage)]);

lengthTrain = length(trainTrainImage);
%%
imageSize = 784;
numBottleneckNeuron = 4;
layers = [
    sequenceInputLayer(imageSize)

    fullyConnectedLayer(50,...
    'WeightsInitializer', 'glorot')
    reluLayer
    
    fullyConnectedLayer(numBottleneckNeuron,...
    'WeightsInitializer', 'glorot')
    reluLayer
    
    fullyConnectedLayer(784,...
    'WeightsInitializer', 'glorot')
    reluLayer
    
    regressionLayer];    

options = trainingOptions('adam',...
    'MaxEpochs',800, ...
    'MiniBatchSize', 8192,...
    'InitialLearnRate', 0.001,...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'verboseFrequency', 6,...
    'ValidationData',{trainValidImageStacked,trainValidImageStacked});

[net, tr] = trainNetwork(trainTrainImageStacked,trainTrainImageStacked,layers,options);

%% Analysis 1

predictImageStacked = predict(net, trainTrainImageStacked);
predictImageUnstacked = reshape(predictImageStacked, [28 28 1 lengthTrain]);
trainTrainImageUnstacked = reshape(trainTrainImage, [28 28 1 lengthTrain]);

montageIndices = [2 18 268 9 277 54 326 23 33 37];

trainTrainImageUnstacked = trainTrainImageUnstacked*255;
trainTrainImageUnstacked = im2single(trainTrainImageUnstacked);
figure
subplot(1,2,1)
montage(trainTrainImageUnstacked(:,:,:,montageIndices),'Size',[2, 5])

title('Original')

predictImageUnstacked = predictImageUnstacked/max(predictImageUnstacked(:));

subplot(1,2,2)
montage(predictImageUnstacked(:,:,:,montageIndices),'Size',[2, 5])
title('Predicted')

%% If required then do more epochs

optionsUpdated = trainingOptions('adam',...
    'MaxEpochs',200, ...
    'MiniBatchSize', 8192,...
    'InitialLearnRate', 0.001,...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'verboseFrequency', 6,...
    'ValidationData',{trainValidImageStacked,trainValidImageStacked});

[net, tr] = trainNetwork(trainTrainImageStacked,trainTrainImageStacked,net.Layers,optionsUpdated);

%% Analysis 2 - Create encode and decode

for iLayer = 1:5
    layers_encode(iLayer) = net.Layers(iLayer);
end
layers_encode(6) = regressionLayer;
net_encode = assembleNetwork(layers_encode);

layers_decode(1) = sequenceInputLayer(numBottleneckNeuron);
layers_decode(2) = net.Layers(6);
layers_decode(3) = net.Layers(7);
layers_decode(4) = regressionLayer;
net_decode = assembleNetwork(layers_decode);

%% Analysis 2 - Scatter plot

bottleneckNeuron = predict(net_encode, testImageStacked);
figure(1)
gscatter(bottleneckNeuron(1,1:1000), bottleneckNeuron(2,1:1000),testLabel(1:1000));
%% Network 2 well produced
wellLabel = [0,5,8,1,7,9];
indices = find(testLabel(1:1000) == num2str(wellLabel(1)));
for iIndex = 2:length(wellLabel)
    temp = find(testLabel(1:1000) == num2str(wellLabel(iIndex)));
    indices = [indices; temp];
end
numPoints = length(indices);
figure(2)
gscatter(bottleneckNeuron(1,indices), bottleneckNeuron(2,indices),testLabel(indices))

%% Rule testing Network 2
rules = [
    2.5 5 3 8.5;            % 0
    7 1.5 17.5 2;           % 1
    11.5 18 19 19;          % 8
    9 9.5 5 6];             % 9
rules = transpose(rules);
%%
numDecode = 1;
imageDecodeStacked = zeros(784, numDecode);
for iRule = 1:numDecode
    imageDecodeStacked(:,iRule) = predict(net_decode, rules(:,iRule));
end
figure(3)
imageDecodeUnstacked = reshape(imageDecodeStacked, [28 28 1 numDecode]);
imageDecodeUnstacked = imageDecodeUnstacked/max(imageDecodeUnstacked(:));
montage(imageDecodeUnstacked(:,:,:,1:numDecode),'Size',[2, 2])
title('Decoded')
