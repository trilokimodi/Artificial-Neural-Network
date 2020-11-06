addpath('D:\Masters Program Chalmers\Projects and Labs\ANN')
clear vars

exerciseNumber = 4;
[trainTrainImage, trainTrainLabel, trainValidImage, trainValidLabel,...
    testImage, testLabel] = LoadMNIST(exerciseNumber);

%%

trainTrainImage = trainTrainImage/255;
trainValidImage = trainValidImage/255;
testImage = testImage/255;

imageSize = [28 28 1];

learningRate = 0.01;
momentum = 0.9;

filterSizeLayer1 = [5 5];
numFiltersLayer1 = 20;
strideLayer1 = [1 1];
paddingSizeLayer1 = [1 1 1 1];

strideLayer2 = [2 2];
poolSizelayer2 = [2 2];
paddingSizeLayer2 = [0 0 0 0];

numNeuronDenseLayer1 = 100;
numNeuronDenseLayer2 = 10;

layers = [
    imageInputLayer(imageSize, 'Name', 'Input')
    
    convolution2dLayer(filterSizeLayer1,...
    numFiltersLayer1,...
    'Stride', strideLayer1,...
    'Padding', paddingSizeLayer1,...
    'WeightsInitializer', 'narrow-normal',...
    'Name', 'Conv_1')
    reluLayer('Name','Relu_1')   
    
    maxPooling2dLayer(poolSizelayer2,...
    'Stride',strideLayer2,...
    'Padding',paddingSizeLayer2,...
    'Name','MaxPool_1')
      
    fullyConnectedLayer(numNeuronDenseLayer1,...
    'WeightsInitializer', 'narrow-normal',...
    'Name','Dense_1')
    reluLayer('Name','Relu_2')
    
    fullyConnectedLayer(numNeuronDenseLayer2,...
    'WeightsInitializer', 'narrow-normal',...
    'Name','Dense_2')
    softmaxLayer('Name','Softmax')
    classificationLayer('Name','Classification')];    

options = trainingOptions('sgdm',...
    'MaxEpochs',60, ...
    'MiniBatchSize', 8192,...
    'InitialLearnRate', learningRate,...
    'Momentum', momentum,...
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
error = sum(predictedLabels ~= testLabel)/numel(testLabel);
display(error)

predictedLabels = classify(net,trainValidImage);
accuracy = sum(predictedLabels == trainValidLabel)/numel(trainValidLabel);
display(accuracy);
error = sum(predictedLabels ~= trainValidLabel);
display(error);
error = sum(predictedLabels ~= trainValidLabel)/numel(trainValidLabel);
display(error)

predictedLabels = classify(net,trainTrainImage);
accuracy = sum(predictedLabels == trainTrainLabel)/numel(trainTrainLabel);
display(accuracy);
error = sum(predictedLabels ~= trainTrainLabel);
display(error);
error = sum(predictedLabels ~= trainTrainLabel)/numel(trainTrainLabel);
display(error)

h= findall(groot,'Type','Figure');
h.MenuBar = 'figure';