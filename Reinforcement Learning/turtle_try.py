from ColabTurtle.Turtle import *
import numpy as np

def DrawSquare(board):
    initializeTurtle()  # Initialize the turtle graphics environment
    
    speed(10)  # Set the drawing speed
    penup()  # Lift the pen
    xi = 0
    y = 0
    x = xi
    boardState = np.reshape(board, (3, 3))  # Reshape the board array
    for iRow in range(boardDimension):
        for iCol in range(boardDimension):
            if boardState[iRow][iCol] == 1:
                color("black")  # Set the pen color to black
                bgcolor("red")  # Set the background color to red
            elif boardState[iRow][iCol] == -1:
                color("black")  # Set the pen color to black
                bgcolor("blue")  # Set the background color to blue
            else:
                # If the cell value is not 1 or -1, use default color
                color("black")  # Set the pen color to black
            x += 60
            goto(x, y)
            pendown()
            for i in range(4):
                forward(50)
                left(90)
            penup()
        y += 60
        x = xi
    done()  # Finish drawing

# Example usage:
DrawSquare([[1, -1, 1], [-1, 1, -1], [1, -1, 1]])
