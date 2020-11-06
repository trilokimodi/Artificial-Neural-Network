import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

(trainImages, trainLabels), (testImages, testLabels) = datasets.mnist.load_data()
trainImages, testImages = trainImages / 255.0, testImages / 255.0

meanTrainImage = np.zeros([28, 28])
meanTestImage = np.zeros([28, 28])
for iPixel in range(28):
    for jPixel in range(28):
        meanTrainImage[iPixel][jPixel] = np.mean(trainImages[:][iPixel][jPixel])
        meanTestImage[iPixel][jPixel] = np.mean(testImages[:][iPixel][jPixel])

trainImages, testImages = trainImages - meanTrainImage, testImages - meanTestImage

classNames = range(0, 10)
classNames = [str(x) for x in classNames]

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(trainImages[i], cmap=plt.cm.binary)
    plt.xlabel(classNames[trainLabels[i]])
plt.show()

trainImages = trainImages.reshape((60000, 28, 28, 1))
testImages = testImages.reshape((10000, 28, 28, 1))

# trainValidateImages = trainImages[50000:]
# trainValidateLabels = trainLabels[50000:]
# trainImages = trainImages[:50000]
# trainLabels = trainLabels[:50000]

trainLabelsNumeric = [int(x) for x in trainLabels]
index = 0
classIndices = np.zeros([10, 1000], dtype=int)
count = np.zeros(10, dtype=int)
while sum(count) < 10000:
    while count[trainLabelsNumeric[index]] < 1000:
        classIndices[trainLabelsNumeric[index]][count[trainLabelsNumeric[index]]] = index
        count[trainLabelsNumeric[index]] += 1
        index += 1
    index += 1

trainTrainImages = trainImages[[x for x in range(60000) if x not in classIndices.flatten()]]
trainTrainLabels = trainLabels[[x for x in range(60000) if x not in classIndices.flatten()]]
trainValidateImages = trainImages[classIndices.flatten()]
trainValidateLabels = trainLabels[classIndices.flatten()]


model = models.Sequential()
tf.keras.layers.ZeroPadding2D(padding=1)
model.add(layers.Conv2D(filters=20, kernel_size=(5,5), strides=(1,1), activation='relu', input_shape=(28,28,1),kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.01)))
model.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid"))
model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu',kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.01)))
model.add(layers.Dense(10,activation='softmax',kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=0.01)))
print(model.summary())

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=sgd, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
history = model.fit(trainTrainImages, trainTrainLabels, batch_size=8192, epochs=60, verbose=1, callbacks=callback,shuffle=True,validation_data=(trainValidateImages, trainValidateLabels), validation_freq=4)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

testLoss, testAcc = model.evaluate(testImages,  testLabels, verbose=1)
print(testAcc)
print(testLoss)
