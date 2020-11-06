%% Network 2
addpath('D:\Masters Program Chalmers\Projects and Labs\ANN')
clear vars
exerciseNumber = 4;
[trainTrainImage, trainTrainLabel, trainValidImage, trainValidLabel,...
    testImage, testLabel] = LoadMNIST(exerciseNumber);

trainTrainImage = trainTrainImage/255;
trainValidImage = trainValidImage/255;
testImage = testImage/255;

imageSize = [28 28 1];

layers = [
    imageInputLayer(imageSize)

    convolution2dLayer([3 3],...
    20,...
    'Stride', [1 1],...
    'Padding', [1 1 1 1],...
    'WeightsInitializer', 'narrow-normal')
    batchNormalizationLayer
    reluLayer   

    maxPooling2dLayer([2 2],...
    'Stride',[2 2],...
    'Padding',[0 0 0 0])

    convolution2dLayer([3 3],...
    30,...
    'Stride', [1 1],...
    'Padding', [1 1 1 1],...
    'WeightsInitializer', 'narrow-normal')
    batchNormalizationLayer
    reluLayer   

    maxPooling2dLayer([2 2],...
    'Stride',[2 2],...
    'Padding',[0 0 0 0])

    convolution2dLayer([3 3],...
    50,...
    'Stride', [1 1],...
    'Padding', [1 1 1 1],...
    'WeightsInitializer', 'narrow-normal')
    batchNormalizationLayer
    reluLayer   

    fullyConnectedLayer(10,...
    'WeightsInitializer', 'narrow-normal')
    softmaxLayer
    classificationLayer];    

options = trainingOptions('sgdm',...
    'MaxEpochs',30, ...
    'MiniBatchSize', 8192,...
    'InitialLearnRate', 0.01,...
    'Momentum', 0.9,...
    'Shuffle','every-epoch', ...
    'ValidationPatience', 5,...
    'ValidationFrequency',30,...
    'Plots','training-progress', ...
    'Verbose',true, ...
    'verboseFrequency', 6,...
    'ValidationData',{trainValidImage,trainValidLabel});

net = trainNetwork(trainTrainImage,trainTrainLabel,layers,options);

predictedLabels = classify(net,testImage);
accuracy = sum(predictedLabels == testLabel)/numel(testLabel);
display(accuracy);
error = sum(predictedLabels ~= testLabel);
display(error);

predictedLabels = classify(net,trainValidImage);
accuracy = sum(predictedLabels == trainValidLabel)/numel(trainValidLabel);
display(accuracy);
error = sum(predictedLabels ~= trainValidLabel);
display(error);

predictedLabels = classify(net,trainTrainImage);
accuracy = sum(predictedLabels == trainTrainLabel)/numel(trainTrainLabel);
display(accuracy);
error = sum(predictedLabels ~= trainTrainLabel);
display(error);