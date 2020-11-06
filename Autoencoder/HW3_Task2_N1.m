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
numBottleneckNeuron = 2;
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
xlabel('Neuron 1')
ylabel('Neuron 2')
title('Scatter plot for all digits')
%% Network 2 well produced
wellLabel = [0,1,9];
indices = find(testLabel(1:1000) == num2str(wellLabel(1)));
for iIndex = 2:length(wellLabel)
    temp = find(testLabel(1:1000) == num2str(wellLabel(iIndex)));
    indices = [indices; temp];
end
numPoints = length(indices);
figure(2)
gscatter(bottleneckNeuron(1,indices), bottleneckNeuron(2,indices),testLabel(indices))
xlabel('Neuron 1')
ylabel('Neuron 2')
title('Scatter plot for well produced digits')

%% Rule testing Network 2
numDecode = 3;
rules = [
    7 9;           % 0
    1 1;           % 1
    12 2];         % 9
rules = transpose(rules);
%%
imageDecodeStacked = zeros(784, numDecode);
for iRule = 1:numDecode
    imageDecodeStacked(:,iRule) = predict(net_decode, rules(:,iRule));
end
figure(3)
imageDecodeUnstacked = reshape(imageDecodeStacked, [28 28 1 numDecode]);

imageDecodeUnstacked = imageDecodeUnstacked/max(imageDecodeUnstacked(:));
montage(imageDecodeUnstacked(:,:,:,1:numDecode),'Size',[1, 3])
title('Decoded')

%% Montage plot for both B2 and B4. 
% This requires both net to run and load in a way that variable fit
% properly. Only done for the objective of report. Hence, commenting out.

figure(4)
s1 = subplot(1,3,1)
montage(trainTrainImageUnstacked(:,:,:,montageIndices),'Size',[2, 5])
title('Original')

s2 = subplot(1,3,2)
montage(predictImageUnstackedB2(:,:,:,montageIndices),'Size',[2, 5])
title('Predicted B2')

s3 = subplot(1,3,3)
montage(predictImageUnstackedB4(:,:,:,montageIndices),'Size',[2, 5])
title('Predicted B4')

s1Pos = get(s1,'position');
s2Pos = get(s2,'position');
s3Pos = get(s3,'position');

ha=get(gcf,'children');
set(ha(3),'position',[s1Pos(1) s1Pos(2) s1Pos(3) s1Pos(4)])
set(ha(2),'position',[s1Pos(1)+s1Pos(3)+0.01 s2Pos(2) s2Pos(3) s2Pos(4)])
set(ha(1),'position',[s1Pos(1)+s1Pos(3)+0.01+s2Pos(3)+0.01 s3Pos(2) s3Pos(3) s3Pos(4)])