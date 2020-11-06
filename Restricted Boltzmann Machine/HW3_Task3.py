import numpy as np
import itertools
import matplotlib.pyplot as plt

# Data distribution
# for element in itertools.product([-1, 1], repeat=3):
#     print(element)
patterns = [[-1, -1, -1, -1, -1, -1, -1, -1, -1],
            [1, -1, -1, 1, -1, -1, 1, -1, -1],
            [-1, 1, -1, -1, 1, -1, -1, 1, -1],
            [-1, -1, 1, -1, -1, 1, -1, -1, 1],
            [1, 1, -1, 1, 1, -1, 1, 1, -1],
            [-1, 1, 1, -1, 1, 1, -1, 1, 1],
            [1, -1, 1, 1, -1, 1, 1, -1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, 1, 1, 1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, -1, -1, -1],
            [-1, -1, -1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, -1, -1, -1, 1, 1, 1]]

# incompletePattern = np.array([1, -1, 1, 0, 0, 0, 0, 0, 0])
incompletePattern = [
            [1, -1, -1, 0, 0, 0, 0, 0, 0],
            [1, -1, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, -1, 0, 0]]

indicesOfComplete = [[0, 1, 2], [0, 1, 2], [0, 3, 6]]  # Corresponding to above incomplete pattern

patternSize = len(patterns[0])
numOfPatterns = 14
numHiddenNeurons = [2, 4, 8, 16]
numOfCDkIterations = 100  # num of CDK Iterations

learningRate = 0.001
numVisibleNeurons = patternSize
maxIterations = 20000
klDivergenceThreshold = 0.1
numMcCullohIterates = 20

klDivergenceVectorM = list()

for iProblem in range(len(numHiddenNeurons)):
    print("problem = ", iProblem)

    weights = np.random.randn(numHiddenNeurons[iProblem], numVisibleNeurons)
    visibleNeurons = np.zeros((1, numVisibleNeurons), int)
    visibleMcCullohStates = np.zeros((1, numVisibleNeurons), int)

    localFieldVisible = np.zeros((1, numVisibleNeurons), float)
    hiddenNeurons = np.zeros((1, numHiddenNeurons[iProblem]), int)
    hiddenMcCullohStates = np.zeros((1, numHiddenNeurons[iProblem]), int)

    localFieldHidden = np.zeros((1, numHiddenNeurons[iProblem]), float)
    thresholdVisible = np.random.randn(1, numVisibleNeurons)
    thresholdHidden = np.random.randn(1, numHiddenNeurons[iProblem])

    boltzmannFrequencyOfPattern = np.zeros((1, numOfPatterns), int)
    boltzmannTotalPatterns = 0
    boltzmannProbabilityOfPattern = np.zeros((1, numOfPatterns), float)

    dataFrequencyOfPattern = np.zeros((1, numOfPatterns), int)
    dataTotalPatterns = 0
    dataProbabilityOfPattern = np.zeros((1, numOfPatterns), float)

    klDivergenceOfPattern = 10 * np.ones((1, numOfPatterns), float)  # 10 is just some big number
    klDivergence = list()

    steadyStateIterations = 0
    iterationIndex = list()
    iterationFlag = 0

    while steadyStateIterations < maxIterations and iterationFlag == 0:
        if steadyStateIterations % 1000 == 0:
            print("Iterations = ", steadyStateIterations)
        weightIncrement = list()
        thresholdVisibleIncrement = list()
        thresholdHiddenIncrement = list()
        numSamplePatterns = 3
        samplePatterns = np.random.randint(0, numOfPatterns, numSamplePatterns)
        dataFrequencyOfPattern[0][samplePatterns] += 1
        dataTotalPatterns += len(samplePatterns)

        for iPattern in range(len(samplePatterns)):
            weightIncrement.append(np.zeros((numHiddenNeurons[iProblem], numVisibleNeurons)))
            # print("Begin")
            # print(weightIncrement)
            thresholdVisibleIncrement.append(np.zeros((1, numVisibleNeurons), float))
            thresholdHiddenIncrement.append(np.zeros((1, numHiddenNeurons[iProblem]), float))
            visibleNeurons = np.asarray(patterns[samplePatterns[iPattern]].copy())
            visibleNeurons = np.reshape(visibleNeurons, (1, numVisibleNeurons))
            localFieldHidden = np.matmul(visibleNeurons, np.transpose(weights)) - thresholdHidden
            initialHiddenField = localFieldHidden.copy()
            localFieldHidden = 1 / (1 + np.exp(-2 * localFieldHidden))
            for iHiddenNeuron in range(numHiddenNeurons[iProblem]):
                if localFieldHidden[0][iHiddenNeuron] >= np.random.rand():
                    hiddenNeurons[0][iHiddenNeuron] = 1
                else:
                    hiddenNeurons[0][iHiddenNeuron] = -1

            for iCDK in range(numOfCDkIterations):
                localFieldVisible = np.matmul(hiddenNeurons, weights) - thresholdVisible
                localFieldVisible = 1 / (1 + np.exp(-2 * localFieldVisible))
                for iVisibleNeuron in range(numVisibleNeurons):
                    if localFieldVisible[0][iVisibleNeuron] >= np.random.rand():
                        visibleNeurons[0][iVisibleNeuron] = 1
                    else:
                        visibleNeurons[0][iVisibleNeuron] = -1
                localFieldHidden = np.matmul(visibleNeurons, np.transpose(weights)) - thresholdHidden
                localFieldHidden = 1 / (1 + np.exp(-2 * localFieldHidden))
                for iHiddenNeuron in range(numHiddenNeurons[iProblem]):
                    if localFieldHidden[0][iHiddenNeuron] >= np.random.rand():
                        hiddenNeurons[0][iHiddenNeuron] = 1
                    else:
                        hiddenNeurons[0][iHiddenNeuron] = -1

            finalHiddenField = np.matmul(visibleNeurons, np.transpose(weights)) - thresholdHidden
            weightIncrement[iPattern] = learningRate * (np.matmul(np.transpose(np.tanh(initialHiddenField)),np.reshape(np.asarray(patterns[samplePatterns[iPattern]]),(1, numVisibleNeurons))) - np.matmul(np.transpose(np.tanh(finalHiddenField)),visibleNeurons))
            thresholdVisibleIncrement[iPattern] = -learningRate * (np.reshape(np.asarray(patterns[samplePatterns[iPattern]]),(1, numVisibleNeurons)) - visibleNeurons)
            thresholdHiddenIncrement[iPattern] = -learningRate * (np.tanh(initialHiddenField) - np.tanh(finalHiddenField))
            # print("End")
            # print(weightIncrement))

        for iPattern in range(len(samplePatterns)):
            weights = weights + weightIncrement[iPattern]
            thresholdVisible = thresholdVisible + thresholdVisibleIncrement[iPattern]
            thresholdHidden = thresholdHidden + thresholdHiddenIncrement[iPattern]

        # Kullback-Leibler Divergence
        if steadyStateIterations % 100 == 0:
            for iPattern in range(numOfPatterns):
                visibleMcCullohStates = np.asarray(patterns[iPattern].copy())
                visibleMcCullohStates = np.reshape(visibleMcCullohStates, (1, numVisibleNeurons))

                for iIterates in range(numMcCullohIterates):
                    localFieldHidden = np.matmul(visibleMcCullohStates, np.transpose(weights)) - thresholdHidden
                    localFieldHidden = 1/(1 + np.exp(-2 * localFieldHidden))
                    for iHiddenNeuron in range(numHiddenNeurons[iProblem]):
                        if localFieldHidden[0][iHiddenNeuron] >= np.random.rand():
                            hiddenMcCullohStates[0][iHiddenNeuron] = 1
                        else:
                            hiddenMcCullohStates[0][iHiddenNeuron] = -1
                    localFieldVisible = np.matmul(hiddenMcCullohStates, weights) - thresholdVisible
                    localFieldVisible = 1/(1 + np.exp(-2 * localFieldVisible))
                    for iVisibleNeuron in range(numVisibleNeurons):
                        if localFieldVisible[0][iVisibleNeuron] >= np.random.rand():
                            visibleMcCullohStates[0][iVisibleNeuron] = 1
                        else:
                            visibleMcCullohStates[0][iVisibleNeuron] = -1
                comparison = visibleMcCullohStates[0] == patterns[iPattern]
                if all(comparison) is True:
                    boltzmannFrequencyOfPattern[0][iPattern] += 1
                    boltzmannTotalPatterns += 1
                else:
                    boltzmannTotalPatterns += 1
            if all(boltzmannFrequencyOfPattern[0]) > 0 and all(dataFrequencyOfPattern[0]) > 0:
                dataProbabilityOfPattern = dataFrequencyOfPattern / dataTotalPatterns
                boltzmannProbabilityOfPattern = boltzmannFrequencyOfPattern / boltzmannTotalPatterns
                klDivergenceOfPattern = np.log(np.divide(dataProbabilityOfPattern, boltzmannProbabilityOfPattern))
                klDivergenceOfPattern = np.multiply(dataProbabilityOfPattern, klDivergenceOfPattern)

            elif any(boltzmannFrequencyOfPattern[0]) > 0:
                comparisonBoltzmann = boltzmannFrequencyOfPattern[0] > 0
                comparisonData = dataFrequencyOfPattern[0] > 0
                comparisonBoth = comparisonBoltzmann & comparisonData
                if any(comparisonBoth):
                    boltzmannProbabilityOfPattern[0][comparisonBoth] = boltzmannFrequencyOfPattern[0][comparisonBoth] / boltzmannTotalPatterns
                    dataProbabilityOfPattern[0][comparisonBoth] = dataFrequencyOfPattern[0][comparisonBoth] / dataTotalPatterns

                    klDivergenceOfPattern[0][comparisonBoth] = np.log(np.divide(dataProbabilityOfPattern[0][comparisonBoth], boltzmannProbabilityOfPattern[0][comparisonBoth]))
                    klDivergenceOfPattern[0][comparisonBoth] = np.multiply(dataProbabilityOfPattern[0][comparisonBoth], klDivergenceOfPattern[0][comparisonBoth])

            klDivergence.append(sum(klDivergenceOfPattern[0]))
            iterationIndex.append(steadyStateIterations)

        if klDivergence[len(klDivergence) - 1] < klDivergenceThreshold:
            iterationFlag = 1

        if steadyStateIterations == maxIterations - 1 or iterationFlag == 1:
            plt.figure(iProblem)
            print(klDivergence[len(klDivergence) - 1])
            print("Iterations required = ", steadyStateIterations + 1)
            plt.plot(iterationIndex, klDivergence)
            plt.xlabel("Iterations")
            plt.ylabel("KL Divergence")
            plotTitle = "KL Divergence plot for hidden neuron = " + str(numHiddenNeurons[iProblem])
            plt.title(plotTitle)
            fileName = "KLDivergencePlot_" + str(numHiddenNeurons[iProblem])
            plt.savefig("D:\\Masters Program Chalmers\\Projects and Labs\\ANN\\%s.png" % fileName)
            klDivergenceVectorM.append(klDivergence.copy())

        if steadyStateIterations == maxIterations - 1 or iterationFlag == 1:
            print("Boltzmann", boltzmannFrequencyOfPattern)
            print("Data", dataFrequencyOfPattern)

        weightIncrement.clear()
        thresholdVisibleIncrement.clear()
        thresholdHiddenIncrement.clear()

        steadyStateIterations += 1

    # Complete Patterns - Feed incomplete pattern and let the network converge in 10 steps
    fileHandler = open('D:\\Masters Program Chalmers\\Projects and Labs\\ANN\\HW3_Task3.txt', 'a')
    fileHandler.write("\nNeuron = %s\n" % str(numHiddenNeurons[iProblem]))
    fileHandler.write("KL Divergence = %s\n" % str(klDivergence[len(klDivergence) - 1]))
    fileHandler.close()
    for iIncomplete in range(len(incompletePattern)):
        fileHandler = open('D:\\Masters Program Chalmers\\Projects and Labs\\ANN\\HW3_Task3.txt', 'a')
        fileHandler.write("\nIncomplete Pattern = %s\n" % str(incompletePattern[iIncomplete]))
        fileHandler.write("McCulloh states are\n")

        visibleMcCullohStates = np.asarray(incompletePattern[iIncomplete].copy())
        visibleMcCullohStates = np.reshape(visibleMcCullohStates, (1, numVisibleNeurons))

        for iIterates in range(10):
            localFieldHidden = np.matmul(visibleMcCullohStates, np.transpose(weights)) - thresholdHidden
            localFieldHidden = 1/(1 + np.exp(-2 * localFieldHidden))
            for iHiddenNeuron in range(numHiddenNeurons[iProblem]):
                if localFieldHidden[0][iHiddenNeuron] > np.random.rand():
                    hiddenMcCullohStates[0][iHiddenNeuron] = 1
                else:
                    hiddenMcCullohStates[0][iHiddenNeuron] = -1
            localFieldVisible = np.matmul(hiddenMcCullohStates, weights) - thresholdVisible
            localFieldVisible = 1/(1 + np.exp(-2 * localFieldVisible))
            for iVisibleNeuron in range(numVisibleNeurons):
                if localFieldVisible[0][iVisibleNeuron] >= np.random.rand():
                    visibleMcCullohStates[0][iVisibleNeuron] = 1
                else:
                    visibleMcCullohStates[0][iVisibleNeuron] = -1
            for iVisible in range(3):
                visibleMcCullohStates[0][indicesOfComplete[iIncomplete][iVisible]] = \
                    incompletePattern[iIncomplete][int(indicesOfComplete[iIncomplete][iVisible])]

            iMcCullohPattern = 0
            McCullohFlag = 0
            while iMcCullohPattern < numOfPatterns and McCullohFlag == 0:
                mcCullohComparison = visibleMcCullohStates[0] == patterns[iMcCullohPattern]
                if all(mcCullohComparison) is True:
                    McCullohFlag = 1
                else:
                    iMcCullohPattern += 1
            if McCullohFlag == 1:
                fileHandler.write("Converged to pattern %s\n" % iMcCullohPattern)
            else:
                fileHandler.write("%s\n" % visibleMcCullohStates)
        fileHandler.close()

    iterationIndex.clear()
    klDivergence.clear()

colorVector = ['r', 'b', 'g', 'm']
plt.figure(len(numHiddenNeurons) + 1)
for iPlot in range(len(numHiddenNeurons)):
    plt.plot(klDivergenceVectorM[iPlot], colorVector[iPlot], label="M = " + str(numHiddenNeurons[iPlot]))
plt.xlabel("Iterations")
plt.ylabel("KL Divergence")
plt.legend(loc=5)
plotTitle = "KL Divergence plot"
plt.title(plotTitle)
fileName = "KLDivergencePlot"
plt.savefig("D:\\Masters Program Chalmers\\Projects and Labs\\ANN\\%s.png" % fileName)
