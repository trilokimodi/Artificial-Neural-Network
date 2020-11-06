import numpy as np
import matplotlib.pyplot as plt
import csv

# Discount factor is considered as 1 throughout and hence not included.
actionProbability = 1
learningRate = 0.1
boardDimension = 3
stateMatrixPlayer1 = list()
stateMatrixPlayer2 = list()
qMatrixPlayer1 = list()
qMatrixPlayer2 = list()
qTablePlayer1 = list()
qTablePlayer2 = list()
baseInitializationMatrix = np.zeros((1, boardDimension ** 2), float)
initializeBoard = np.zeros((1, boardDimension ** 2), int)
maxGames = 100000
baseIndexArray = np.reshape(np.array(np.arange(boardDimension ** 2)), (1, boardDimension ** 2))
probPlayer1Wins = list()
probPlayer2Wins = list()
probDraw = list()
freqPlayer1Wins = 0
freqPlayer2Wins = 0
freqDraw = 0

def fn_check_state(current_state, player_number):
    flagState = 1
    if player_number == 1:
        for iMatrix in range(len(stateMatrixPlayer1)):
            comparison = current_state == stateMatrixPlayer1[iMatrix]
            if all(comparison[0]) is True:
                return iMatrix
            else:
                flagState = 1  # Not found
        if flagState == 1:
            return fn_add_state_add_q(current_state, 1)

    elif player_number == 2:
        for iMatrix in range(len(stateMatrixPlayer2)):
            comparison = current_state == stateMatrixPlayer2[iMatrix]
            if all(comparison[0]) is True:
                return iMatrix
            else:
                flagState = 1  # Not found
        if flagState == 1:
            return fn_add_state_add_q(current_state, 2)


def fn_add_state_add_q(current_state, player_number):
    if player_number == 1:
        stateMatrixPlayer1.append(current_state.copy())
        initializationMatrix = initialization_matrix(current_state)
        qMatrixPlayer1.append(initializationMatrix.copy())
        return len(stateMatrixPlayer1) - 1

    elif player_number == 2:
        stateMatrixPlayer2.append(current_state.copy())
        initializationMatrix = initialization_matrix(current_state)
        qMatrixPlayer2.append(initializationMatrix.copy())
        return len(stateMatrixPlayer2) - 1


def initialization_matrix(current_state):
    unavailableSpots = current_state != 0
    tempInitializeMatrix = baseInitializationMatrix.copy()
    tempInitializeMatrix[unavailableSpots] = np.nan
    return tempInitializeMatrix


def check_game_status(current_state):
    sums = np.zeros(boardDimension * 2 + 2, int)
    tempMatrix = np.reshape(current_state, (boardDimension, boardDimension))
    for i in range(boardDimension):
        sums[i] = tempMatrix[i][0] + tempMatrix[i][1] + tempMatrix[i][2]
        sums[i+3] = tempMatrix[0][i] + tempMatrix[1][i] + tempMatrix[2][i]
        sums[6] += tempMatrix[i][i]
        sums[7] += tempMatrix[i][2-i]
    comparePlayer1Wins = sums == boardDimension
    comparePlayer2Wins = sums == -boardDimension
    compare2 = current_state == 0
    if any(comparePlayer1Wins) is True or any(comparePlayer2Wins) is True:
        return 1  # Win
    elif any(compare2[0]) is True:
        return 0  # Game On
    else:
        return -1  # Game Draw


def fn_optimal_action_location(matrix_location, player_number):
    maxQ = np.NINF
    jLoc = 0
    if player_number == 1:
        for iLocation in range(boardDimension ** 2):
            if qMatrixPlayer1[matrix_location][0][iLocation] is not np.nan:
                if qMatrixPlayer1[matrix_location][0][iLocation] > maxQ:
                    maxQ = qMatrixPlayer1[matrix_location][0][iLocation]
                    jLoc = iLocation
    elif player_number == 2:
        for iLocation in range(boardDimension ** 2):
            if qMatrixPlayer2[matrix_location][0][iLocation] is not np.nan:
                if qMatrixPlayer2[matrix_location][0][iLocation] > maxQ:
                    maxQ = qMatrixPlayer2[matrix_location][0][iLocation]
                    jLoc = iLocation
    return jLoc


def fn_update_with_reward(state_location, action_location, reward, player_number):
    if player_number == 1:
        qMatrixPlayer1[state_location][0][action_location] += learningRate * (reward - qMatrixPlayer1[state_location][0][action_location])
    elif player_number == 2:
        qMatrixPlayer2[state_location][0][action_location] += learningRate * (reward - qMatrixPlayer2[state_location][0][action_location])


def fn_update_without_reward(previous_state_location, current_state_location, action_location, player_number):
    if player_number == 1:
        maxAction = fn_optimal_action_location(current_state_location, player_number)
        qMatrixPlayer1[previous_state_location][0][action_location] += \
            learningRate * (qMatrixPlayer1[current_state_location][0][maxAction] -
                            qMatrixPlayer1[previous_state_location][0][action_location])
    elif player_number == 2:
        maxAction = fn_optimal_action_location(current_state_location, player_number)
        qMatrixPlayer2[previous_state_location][0][action_location] += \
            learningRate * (qMatrixPlayer2[current_state_location][0][maxAction] -
                            qMatrixPlayer2[previous_state_location][0][action_location])


for iGame in range(maxGames):
    if (iGame + 1) % 500 == 0 and iGame >= 2499:
        actionProbability = 0.9 * actionProbability
    board = initializeBoard.copy()
    flagGame = 0
    historyStatePlayer1 = list()
    historyActionPlayer1 = list()
    historyStatePlayer2 = list()
    historyActionPlayer2 = list()
    whoWon = 0
    whoLost = 0
    gameStatus = 0
    moveNumber = 0
    while flagGame == 0:
        moveNumber += 1
        playerTurn = 1
        if len(historyStatePlayer1) == 2:
            historyStatePlayer1.pop(0)
            historyActionPlayer1.pop(0)
        matrixLocation = fn_check_state(board, playerTurn)  # matrix Location implies location of state in Q Table
        historyStatePlayer1.append(matrixLocation)
        if moveNumber > 1:
            fn_update_without_reward(historyStatePlayer1[0], historyStatePlayer1[1], historyActionPlayer1[0],
                                     playerTurn)
        optimalActionLocation = fn_optimal_action_location(matrixLocation, playerTurn)
        if np.random.rand() < actionProbability:
            availableLocationsBoolean = board == 0
            availableLocations = baseIndexArray[0][availableLocationsBoolean[0]]
            optimalActionLocation = np.random.choice(availableLocations, 1)
            board[0][optimalActionLocation] = 1
            historyActionPlayer1.append(optimalActionLocation)
        else:
            board[0][optimalActionLocation] = 1
            historyActionPlayer1.append(optimalActionLocation)
        gameStatus = check_game_status(board)
        if gameStatus == 1:
            whoWon = 1
            whoLost = 2
            freqPlayer1Wins += 1
            if iGame > 40000:
                print("Player 1 wins at ", iGame)
            break
        elif gameStatus == -1:
            freqDraw += 1
            break
        # Player 2
        playerTurn = 2
        if len(historyStatePlayer2) == 2:
            historyStatePlayer2.pop(0)
            historyActionPlayer2.pop(0)
        matrixLocation = fn_check_state(board, playerTurn)  # matrix Location implies location of state in Q Table
        historyStatePlayer2.append(matrixLocation)

        if moveNumber > 1:
            fn_update_without_reward(historyStatePlayer2[0], historyStatePlayer2[1], historyActionPlayer2[0],
                                     playerTurn)
        optimalActionLocation = fn_optimal_action_location(matrixLocation, playerTurn)
        if np.random.rand() < actionProbability:
            availableLocationsBoolean = board == 0
            availableLocations = baseIndexArray[0][availableLocationsBoolean[0]]
            optimalActionLocation = np.random.choice(availableLocations, 1)
            board[0][optimalActionLocation] = -1
            historyActionPlayer2.append(optimalActionLocation)
        else:
            board[0][optimalActionLocation] = -1
            historyActionPlayer2.append(optimalActionLocation)
        gameStatus = check_game_status(board)
        if gameStatus == 1:
            whoWon = 2
            whoLost = 1
            freqPlayer2Wins += 1
            if iGame > 40000:
                print("Player 2 wins at ", iGame)
            break
        elif gameStatus == -1:
            freqDraw += 1
            break

    if gameStatus == 1 and whoWon == 1 and whoLost == 2:
        fn_update_with_reward(historyStatePlayer1[1], historyActionPlayer1[1], 1, whoWon)
        fn_update_with_reward(historyStatePlayer2[1], historyActionPlayer2[1], -1, whoLost)
    elif gameStatus == 1 and whoWon == 2 and whoLost == 1:
        fn_update_with_reward(historyStatePlayer2[1], historyActionPlayer2[1], 1, whoWon)
        fn_update_with_reward(historyStatePlayer1[1], historyActionPlayer1[1], -1, whoLost)

    probPlayer1Wins.append(freqPlayer1Wins/(iGame + 1))
    probPlayer2Wins.append(freqPlayer2Wins/(iGame + 1))
    probDraw.append(freqDraw/(iGame + 1))

for x in range(2):
    qTablePlayer1.append([])
    qTablePlayer2.append([])
for iList in range(len(stateMatrixPlayer1)):
    qTablePlayer1[0].append(np.reshape(stateMatrixPlayer1[iList], (boardDimension, boardDimension)))
    qTablePlayer1[1].append(np.reshape(qMatrixPlayer1[iList], (boardDimension, boardDimension)))

for iList in range(len(stateMatrixPlayer2)):
    qTablePlayer2[0].append(np.reshape(stateMatrixPlayer2[iList], (boardDimension, boardDimension)))
    qTablePlayer2[1].append(np.reshape(qMatrixPlayer2[iList], (boardDimension, boardDimension)))

csvMatrixPlayer1 = np.zeros((6, len(stateMatrixPlayer1) * 3))
for iPosition in range(len(stateMatrixPlayer1)):
    for iList in range(3):
        for jList in range(3):
            csvMatrixPlayer1[iList][iPosition * 3 + jList] = qTablePlayer1[0][iPosition][iList][jList]
            csvMatrixPlayer1[iList + 3][iPosition * 3 + jList] = qTablePlayer1[1][iPosition][iList][jList]

csvMatrixPlayer2 = np.zeros((6, len(stateMatrixPlayer2) * 3))
for iPosition in range(len(stateMatrixPlayer2)):
    for iList in range(3):
        for jList in range(3):
            csvMatrixPlayer2[iList][iPosition * 3 + jList] = qTablePlayer2[0][iPosition][iList][jList]
            csvMatrixPlayer2[iList + 3][iPosition * 3 + jList] = qTablePlayer2[1][iPosition][iList][jList]

fileHandler = open('D:\\Masters Program Chalmers\\Projects and Labs\\ANN\\HW3_Task4_Player1.csv', 'a', newline="")
with fileHandler:
    write = csv.writer(fileHandler)
    write.writerows(csvMatrixPlayer1)
fileHandler.close()

fileHandler = open('D:\\Masters Program Chalmers\\Projects and Labs\\ANN\\HW3_Task4_Player2.csv', 'a', newline="")
with fileHandler:
    write = csv.writer(fileHandler)
    write.writerows(csvMatrixPlayer2)
fileHandler.close()

print(probPlayer1Wins[len(probPlayer1Wins) - 1])
print(probPlayer2Wins[len(probPlayer2Wins) - 1])
print(probDraw[len(probDraw) - 1])

plt.figure()
xAxis = np.arange(maxGames)
xAxis = xAxis + 1
plt.plot(xAxis, probPlayer1Wins, 'r', label="Player 1 Wins")
plt.plot(xAxis, probPlayer2Wins, 'g', label="Player 2 Wins")
plt.plot(xAxis, probDraw, 'b', label="Draws")
plt.xlabel("Iterations")
plt.ylabel("Frequency ratio")
plt.legend(loc=5)
plotTitle = " Learning curve for tic-tac-toe"
plt.title(plotTitle)

fileName = "HW3_Task4_learningCurve"
plt.savefig("D:\\Masters Program Chalmers\\Projects and Labs\\ANN\\%s.png" % fileName)
