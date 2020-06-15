# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        
        "*** YOUR CODE HERE ***"
        if action == "Stop":
            return -999
        dis = manhattanDistance(newGhostStates[0].getPosition(), newPos)
        closeFoodPath = closestFood(newFood.asList(), newPos)
        numFood = len(newFood.asList())

        if (dis <= 2):
            return dis*2
        else:
            return - closeFoodPath - numFood * 99 + 9999

def closestFood(listFood, pacmanPosition):
    if len(listFood) == 0:
        return -1
    min = 9999
    for x in listFood:
        tmp = manhattanDistance(x, pacmanPosition)
        if (min > tmp):
            min = tmp
    return min

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # run maxValue in there for easy to get action
        legalMoves = gameState.getLegalActions(0)
        value = -99999
        action = ""
        for act in legalMoves:
            childState = gameState.generateSuccessor(0, act)
            childValue = self.minValue(childState, 1)
            if (value < childValue):
                value = childValue
                action = act
        return action
        # util.raiseNotDefined()
    
    def maxValue(self, gameState, turn):
        # decrease self.depth
        self.depth = self.depth - 1
        if gameState.isWin() or gameState.isLose() or self.depth == 0:
            self.depth = self.depth + 1
            return self.evaluationFunction(gameState)
        
        value = -99999
        childState = 0
        
        for act in gameState.getLegalActions(0):
            childState = gameState.generateSuccessor(0, act)
            value = max(value, self.minValue(childState, turn + 1))
        # increase the self.depth when run out
        self.depth = self.depth + 1         
        return value

    def minValue(self, gameState, turn):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        # create numAgent to check how many player
        numAgent = gameState.getNumAgents()
        value = 99999
        childValue = 0

        for act in gameState.getLegalActions(turn):
            childState = gameState.generateSuccessor(turn, act)
            # check the next player is max or min
            if ((turn + 1) % numAgent == 0):
                childValue = self.maxValue(childState, 0) 
            else:
                childValue = self.minValue(childState, turn + 1)
            value = min(value, childValue)

        return value

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        legalMoves = gameState.getLegalActions(0)
        value = -99999
        action = ""
        alpha = -99999
        beta = 99999

        for act in legalMoves:
            childState = gameState.generateSuccessor(0, act)
            childValue = self.min_value(childState, alpha, beta, 1)
            if (value < childValue):
                value = childValue
                action = act
            alpha = max(alpha, value)
        return action      

        # util.raiseNotDefined()

    def max_value(self, gameState, alpha, beta, turn):
        self.depth = self.depth - 1
        if gameState.isWin() or gameState.isLose() or self.depth == 0:
            self.depth = self.depth + 1
            return self.evaluationFunction(gameState)

        value = -99999
        for act in gameState.getLegalActions(0):
            childState = gameState.generateSuccessor(0, act)
            value = max(value, self.min_value(childState, alpha, beta, 1))
            if value > beta:
                self.depth = self.depth + 1 
                return value
            alpha = max(alpha, value)
        self.depth = self.depth + 1
        return value

    def min_value(self, gameState, alpha, beta, turn):

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        numAgent = gameState.getNumAgents()
        value = 99999
        childValue = 0

        for act in gameState.getLegalActions(turn):
            childState = gameState.generateSuccessor(turn, act)
            if ((turn + 1) % numAgent == 0):
                childValue = self.max_value(childState, alpha, beta, 0)
            else:
                childValue = self.min_value(childState, alpha, beta, turn + 1)
            value = min(value, childValue)
            if value < alpha: return value
            beta = min(beta, value)
        return value   

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        legalMoves = gameState.getLegalActions(0)
        value = -9999
        action = ""
        for act in legalMoves:
            childState = gameState.generateSuccessor(0, act)
            childValue = self.chance_value(childState, 1)
            if (value < childValue):
                value = childValue
                action = act
        return action
        # util.raiseNotDefined()

    def max_value(self, gameState, turn):
        self.depth = self.depth - 1
        if gameState.isWin() or gameState.isLose() or self.depth == 0:
            self.depth = self.depth + 1
            return self.evaluationFunction(gameState)

        value = -9999
        legalMoves = gameState.getLegalActions(0)
        for act in legalMoves:
            childState = gameState.generateSuccessor(0, act)
            value = max(value, self.chance_value(childState, 1))
        self.depth = self.depth + 1
        return value

    def chance_value(self, gameState, turn):
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        numAgent = gameState.getNumAgents()
        values = []
        legalMoves = gameState.getLegalActions(turn)
        for act in legalMoves:
            childState = gameState.generateSuccessor(turn, act)
            if (turn + 1) % numAgent == 0:
                values.append(self.max_value(childState, 0))
            else:
                values.append(self.chance_value(childState, turn + 1))

        numWays = len(legalMoves)
        probabilities = [1/numWays for x in range(numWays)]
        return self.expect(values, probabilities)
        
    def expect(self, values, probabilities):
        expectValue = 0
        for i in range(len(values)):
            expectValue = expectValue + values[i] * probabilities[i]
        return expectValue


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPosition = currentGameState.getPacmanPosition()
    # x,y = currentGameState.getGhostPosition(1)
    # x = round(x)
    # y = round(y)

    ghostPositionList = roundGhostPositionList(currentGameState.getGhostPositions())

    foodList = currentGameState.getFood().asList()
    closestFood = breadFirstSearchCost(pacmanPosition, foodList, currentGameState)
    
    capsuleList = currentGameState.getCapsules()
    closestCapsule = breadFirstSearchCost(pacmanPosition, capsuleList, currentGameState)
    scaredTimer = currentGameState.getGhostState(1).scaredTimer

    disToGhost = breadFirstSearchCost(pacmanPosition, ghostPositionList, currentGameState)

    if (scaredTimer > 0 and disToGhost < 10):
        # return currentGameState.getScore() + 100 / (disToGhost + 1)
        return currentGameState.getScore() + 100 / (disToGhost + 1) - 10*len(foodList)

    if (disToGhost < 10):
        if (closestCapsule < 3 and len(capsuleList) != 0):
            # return 20 / closestCapsule + currentGameState.getScore()
            return 20 / closestCapsule + currentGameState.getScore() - 10*len(foodList)
    
    # return -closestFood + currentGameState.getScore()
    return -closestFood + currentGameState.getScore() - 10*len(foodList)


def breadFirstSearchCost(point1, foodList, gameState):

    startState = point1
    closed = util.Counter()
    fringe = fringeQueue()

    fringe.push(Node(startState, 0))
    while True:
        if (fringe.isEmpty()):
            # print("Not Solution", foodList)
            return 0
        
        node = fringe.pop()
        cost = node.getCost()
        state = node.getState()

        closed[state] = 1
        if (state in foodList):
            return cost

        legalChildState = getLegalActions(state, gameState.getWalls())
        for x in legalChildState:
            if (closed[x] != 1 and not fringe.isExist(x)):
                fringe.push(Node(x, cost + 1))

def roundGhostPositionList(ghostPositionList):
    PositionList = []
    for position in ghostPositionList:
        x,y = position
        x = round(x)
        y = round(y)
        PositionList.append((x,y))

    return PositionList

def getLegalActions(state, walls):
    successors = []
    for direction in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        x, y = state
        dx, dy = direction
        nextx, nexty = int(x + dx), int(y + dy)
        if not walls[nextx][nexty]:
            successors.append((nextx, nexty))
    return successors

class Node:
    def __init__(self, state, cost):
        self.state = state
        self.cost = cost

    def getState(self):
        return self.state

    def getCost(self):
        return self.cost

class fringeQueue:
    def __init__(self):
        self.list = []

    def push(self, item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0, item)

    def pop(self):
        """
        Dequeue the earliest enqueued item still in the queue. This
        operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0

    def isExist(self, state):
        for x in self.list:
            if (x.getState() == state):
                return True
        return False
    
# Abbreviation
better = betterEvaluationFunction
