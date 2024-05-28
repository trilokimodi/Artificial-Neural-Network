import numpy as np
import matplotlib.pyplot as plt
import csv
# import ColabTurtle as turtle
from ColabTurtle.Turtle import *

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
        sums[i + 3] = tempMatrix[0][i] + tempMatrix[1][i] + tempMatrix[2][i]
        sums[6] += tempMatrix[i][i]
        sums[7] += tempMatrix[i][2 - i]
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
        qMatrixPlayer1[state_location][0][action_location] += learningRate * (
                reward - qMatrixPlayer1[state_location][0][action_location])
    elif player_number == 2:
        qMatrixPlayer2[state_location][0][action_location] += learningRate * (
                reward - qMatrixPlayer2[state_location][0][action_location])


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

def DrawSquare(board):
    initializeTurtle()
    speed(10)
    penup()
    xi = 0
    y = 0
    x = xi
    boardState = np.reshape(board, (3, 3))
    for iRow in range(boardDimension):
        for iCol in range(boardDimension):
            if boardState[iRow][iCol] == 1:
                color("black")
                bgcolor("red")
            elif boardState[iRow][iCol] == -1:
                color("black")
                bgcolor("blue")
            else:
                color("black")
            x += 60
            goto(x, y)
            pendown()
            for i in range(4):
                forward(50)
                left(90)
            penup()
        y += 60
        x = xi
    done()

filePath1 = "/content/Artificial-Neural-Network/Reinforcement Learning/HW3_Task4_Player1.csv"
filePath2 = "/content/Artificial-Neural-Network/Reinforcement Learning/HW3_Task4_Player2.csv"
qMatrixPlayer1 = list()
stateMatrixPlayer1 = list()
qTablePlayer1 = list()
matrixPlayer1 = list()
qMatrixPlayer2 = list()
stateMatrixPlayer2 = list()
qTablePlayer2 = list()
matrixPlayer2 = list()
boardDimension = 3
learningRate = 0.1
baseIndexArray = np.reshape(np.array(np.arange(boardDimension ** 2)), (1, boardDimension ** 2))
baseInitializationMatrix = np.zeros((1, boardDimension ** 2), float)
initializeBoard = np.zeros((1, boardDimension ** 2), int)
for iTable in range(2):
    qTablePlayer1.append([])
    qTablePlayer2.append([])

rowLen1 = 0
fileHandler = open(filePath1 , 'r', newline="")
with fileHandler:
    read = csv.reader(fileHandler)
    for row in read:
        matrixPlayer1.append(row)
        rowLen1 = len(row)
fileHandler.close()

rowLen2 = 0
fileHandler = open(filePath2, 'r', newline="")
with fileHandler:
    read = csv.reader(fileHandler)
    for row in read:
        matrixPlayer2.append(row)
        rowLen2 = len(row)
fileHandler.close()


csvMatrixPlayer1 = np.zeros((6, rowLen1))
for iRow in range(len(csvMatrixPlayer1)):
    csvMatrixPlayer1[iRow] = matrixPlayer1[iRow].copy()

csvMatrixPlayer2 = np.zeros((6, rowLen2))
for iRow in range(len(csvMatrixPlayer2)):
    csvMatrixPlayer2[iRow] = matrixPlayer2[iRow].copy()

for iPosition in range(int(rowLen1 / 3)):
    qTablePlayer1[0].append(np.zeros((3, 3)))
    qTablePlayer1[1].append(np.zeros((3, 3)))
    for iList in range(3):
        for jList in range(3):
            qTablePlayer1[0][iPosition][iList][jList] = csvMatrixPlayer1[iList][iPosition * 3 + jList]
            qTablePlayer1[1][iPosition][iList][jList] = csvMatrixPlayer1[iList + 3][iPosition * 3 + jList]

for iPosition in range(int(rowLen2 / 3)):
    qTablePlayer2[0].append(np.zeros((3, 3)))
    qTablePlayer2[1].append(np.zeros((3, 3)))
    for iList in range(3):
        for jList in range(3):
            qTablePlayer2[0][iPosition][iList][jList] = csvMatrixPlayer2[iList][iPosition * 3 + jList]
            qTablePlayer2[1][iPosition][iList][jList] = csvMatrixPlayer2[iList + 3][iPosition * 3 + jList]

for iList in range(int(rowLen1 / 3)):
    stateMatrixPlayer1.append(np.reshape(qTablePlayer1[0][iList], (1, boardDimension ** 2)))
    qMatrixPlayer1.append(np.reshape(qTablePlayer1[1][iList], (1, boardDimension ** 2)))

for iList in range(int(rowLen2 / 3)):
    stateMatrixPlayer2.append(np.reshape(qTablePlayer2[0][iList], (1, boardDimension ** 2)))
    qMatrixPlayer2.append(np.reshape(qTablePlayer2[1][iList], (1, boardDimension ** 2)))

actionProbability = 0.000001
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

print("Tossing a coin, head implies you start and tails implies AI starts")
whoStarts = np.random.randint(0, 2, 1)
if whoStarts == 0:
    print("AI starts")
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
            print(np.reshape(board, (boardDimension, boardDimension)))
            print("AI Wins")
            break
        elif gameStatus == -1:
            print(np.reshape(board, (boardDimension, boardDimension)))
            print("We shared equal pie")
            break
        # Player 2
        playerTurn = 2
        print(np.reshape(board, (boardDimension, boardDimension)))
        availableLocationsBoolean = board == 0
        availableLocations = baseIndexArray[0][availableLocationsBoolean[0]]
        print("availableLocations = ", end="")
        print(availableLocations)
        DrawSquare(board)
        optimalActionLocation = int(input("Enter your location in 0 to 9\n"))
        while optimalActionLocation not in availableLocations:
            optimalActionLocation = int(input("Enter among the free and possible positions\n"))
        board[0][optimalActionLocation] = -1
        gameStatus = check_game_status(board)
        if gameStatus == 1:
            print(np.reshape(board, (boardDimension, boardDimension)))
            print("Well there's a bug. You won against my AI")
            break
        elif gameStatus == -1:
            print(np.reshape(board, (boardDimension, boardDimension)))
            print("We shared equal pie")
            break
else:
    print("You start")
    while flagGame == 0:
        moveNumber += 1
        playerTurn = 1
        print(np.reshape(board, (boardDimension, boardDimension)))
        availableLocationsBoolean = board == 0
        availableLocations = baseIndexArray[0][availableLocationsBoolean[0]]
        print("availableLocations = ", end="")
        print(availableLocations)
        DrawSquare(board)
        optimalActionLocation = int(input("Enter your location in 0 to 9\n"))
        while optimalActionLocation not in availableLocations:
            optimalActionLocation = int(input("Enter among the free and possible positions\n"))
        board[0][optimalActionLocation] = 1
        gameStatus = check_game_status(board)
        if gameStatus == 1:
            print(np.reshape(board, (boardDimension, boardDimension)))
            print("Well there's a bug. You won against my AI")
            break
        elif gameStatus == -1:
            print(np.reshape(board, (boardDimension, boardDimension)))
            print("Draw!! We shared equal pie")
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
            print(np.reshape(board, (boardDimension, boardDimension)))
            print("AI Wins")
            break
        elif gameStatus == -1:
            print(np.reshape(board, (boardDimension, boardDimension)))
            print("Draw!! We shared equal pie")
            break
