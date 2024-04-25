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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        """
            到最近的豆子距离越近，分越高
            离鬼的距离在1以内，一定不能去
            离鬼怪的距离在1以外，无所谓鬼怪
            因为第一题测试的图全部没有墙，所以直接用曼哈顿距离
            
            暂时不管吃到胶囊
        """
        newNumFood = successorGameState.getNumFood()
        curFood = currentGameState.getFood()
        curWalls = currentGameState.getWalls()

        import math
        def disToClosestFood():
            ret = min([manhattanDistance(newPos, food) for food in curFood.asList()])
            if ret == 0:
                return 0.1
            return ret

        def disToClosestGhost():
            # 暂时先用曼哈顿距离，方便
            ret = min([manhattanDistance(newPos, ghost.configuration.pos) for ghost in newGhostStates])
            return ret

        score = 0
        if disToClosestGhost() <= 1:
            score = -math.inf
        score += 9 / disToClosestFood()
        return score
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        # 获取当前有多少个智能体
        numAgents = gameState.getNumAgents()
        GhostIndex = [i for i in range(1, numAgents)]

        # 当前状态是否为游戏结束或者搜索到了限制层数
        def is_over(State, deep):
            return deep == self.depth or State.isWin() or State.isLose()
        
        # 采用递归方式把ghost每个走过的状态检查一遍
        def min_value(state, deep, ghost):  # minimizer
            if is_over(state, deep):
                return self.evaluationFunction(state)

            v = 1e6   # β初始值为无穷大
            for action in state.getLegalActions(ghost): # 递归的查找最小值
                if ghost == GhostIndex[-1]: # 如果ghost==ghostindex[-1]的话，说明这一层的幽灵已经查找结束，下一层查找吃豆人
                    v = min(v, max_value(state.generateSuccessor(ghost, action), deep + 1))
                else:                       # 反之寻找下一个幽灵，取最小值
                    v = min(v, min_value(state.generateSuccessor(ghost, action), deep, ghost + 1))
            return v

        def max_value(state, deep):  # maximizer

            if is_over(state, deep):
                return self.evaluationFunction(state)

            v = -1e6  # α初始值为无穷小
            for action in state.getLegalActions(0): # 递归查找最大值
                v = max(v, min_value(state.generateSuccessor(0, action), deep, 1)) # 在下一层查找幽灵
            return v

        # 从第一层开始查找，为max_value
        best_act = []
        best_val = -1e6
        actions = gameState.getLegalActions(0) # 0是吃豆人的代理
        for act in actions:
            val = min_value(gameState.generateSuccessor(0, act),0,1)
            if val > best_val:
                best_val = val
                best_act = act

        return best_act

        # 智能体层取max，幽灵层取min，但是幽灵层有numAgents个幽灵
        def dfs(curGameState: GameState, deep, agentIndex):
            # 判断当前是否为叶节点，以及是否为胜利或失败状态
            if deep == self.depth or curGameState.isWin() or curGameState.isLose():
                return self.evaluationFunction(curGameState), "None"

            # 如果当前智能体不是最后一个，那么到下一个智能体；否则换到下一层
            nxtDeep = deep if agentIndex < numAgents - 1 else deep + 1
            nxtAgentIndex = agentIndex + 1 if agentIndex < numAgents - 1 else 0

            results = []
            actions = curGameState.getLegalActions(agentIndex)
            for act in actions:
                # 生成下一状态
                newGameState = curGameState.generateSuccessor(agentIndex, act)
                # 获取下一状态评分和对应的动作
                results.append((dfs(newGameState, nxtDeep, nxtAgentIndex)[0], act))

            tmp:tuple = results[0]
            if agentIndex == 0:
                # 如果当前智能体是 Pacman，则选择最大评分的动作
                for res in results:
                    if tmp[0] < res[0]:
                        tmp = res
            else:
                # 如果当前智能体是 Ghost，则选择最小评分的动作
                for res in results:
                    if tmp[0] > res[0]:
                        tmp = res
            return tmp

        # 返回最优动作
        return dfs(gameState, 0, 0)[1]
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
    
        numAgents = gameState.getNumAgents()
        GhostIndex = [i for i in range(1, numAgents)]

        # 当前状态是否为游戏结束或者搜索到了限制层数
        def is_over(State, deep):
            return deep == self.depth or State.isWin() or State.isLose()
        
        # 采用递归方式把ghost每个走过的状态检查一遍
        def min_value(state, deep, ghost, alpha, beta):  # minimizer计算这一层的beta   alpha是上一层的alpha
            if is_over(state, deep):
                return self.evaluationFunction(state)

            v = 1e100   # β初始值为无穷大
            for action in state.getLegalActions(ghost): # 递归的查找最小值
                if ghost == GhostIndex[-1]: # 如果ghost==ghostindex[-1]的话，说明这一层的幽灵已经查找结束，下一层查找吃豆人
                    v = min(v, max_value(state.generateSuccessor(ghost, action), deep + 1, alpha, beta))
                    if v < alpha:
                        return v
                else:                       # 反之寻找下一个幽灵，取最小值
                    v = min(v, min_value(state.generateSuccessor(ghost, action), deep, ghost + 1, alpha, beta))
                
                # 减枝
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        def max_value(state, deep, alpha, beta):  # maximizer计算这一层的alpha    beta是上一层的beta

            if is_over(state, deep):
                return self.evaluationFunction(state)

            v = -1e100  # α初始值为无穷小
            for action in state.getLegalActions(0): # 递归查找最大值
                v = max(v, min_value(state.generateSuccessor(0, action), deep, 1, alpha, beta)) # 在下一层查找幽灵
                # 减枝
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v
            
        # 从第一层开始查找，为min_value
        best_act = []
        # best_val = -1e6
        alpha = -1e6
        beta = 1e6
        actions = gameState.getLegalActions(0) # 0是吃豆人的代理
        for act in actions:
            val = min_value(gameState.generateSuccessor(0, act), 0, 1, alpha, beta)
            if val > alpha:
                alpha = val
                best_act = act
        return best_act
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"


        # 本题考虑了不是每一次ghost都能做出对ghost最好的动作，因此不是对ghost采取action后所有state的各值取最小，而是对他们求期望。
        # 由题意，ghost采取各action概率相同，所以只需要对各state的值求平均即可。其他部分和minimax中的一致。                        
        numAgents = gameState.getNumAgents()
        GhostIndex = [i for i in range(1, numAgents)]

        # 当前状态是否为游戏结束或者搜索到了限制层数
        def is_over(State, deep):
            return deep == self.depth or State.isWin() or State.isLose()
                        
        # 采用递归方式把ghost每个走过的状态检查一遍
        def evg_value(state, deep, ghost):  # minimizer计算这一层的beta   alpha是上一层的alpha
            if is_over(state, deep):
                return self.evaluationFunction(state)

            v = 0   # β初始值为无穷大
            prob = 1 / len(state.getLegalActions(ghost))
            for action in state.getLegalActions(ghost): # 递归的查找平均值
                if ghost == GhostIndex[-1]: # 如果ghost==ghostindex[-1]的话，说明这一层的幽灵已经查找结束，下一层查找吃豆人
                    v += max_value(state.generateSuccessor(ghost, action), deep + 1)
                else:                       # 反之寻找下一个幽灵，取最小值
                    v += evg_value(state.generateSuccessor(ghost, action), deep, ghost + 1)
            v = v * prob
            return v
        
        def max_value(state, deep):  # maximizer计算这一层的alpha    beta是上一层的beta
            if is_over(state, deep):
                return self.evaluationFunction(state)

            v = -1e100  # α初始值为无穷小
            for action in state.getLegalActions(0): # 递归查找最大值
                v = max(v, evg_value(state.generateSuccessor(0, action), deep, 1)) # 在下一层查找幽灵
            return v
            
        # 从第一层开始查找，为max
        best_act = []
        best_val = -1e6
        actions = gameState.getLegalActions(0) # 0是吃豆人的代理
        for act in actions:
            val = evg_value(gameState.generateSuccessor(0, act),0,1)
            if val > best_val:
                best_val = val
                best_act = act

        return best_act
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    # newCapsules = currentGameState.getCapsules()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    walls = currentGameState.getWalls()

    # 如果不是新的ScaredTimes，则新状态为ghost：返回最低值
    newFood = newFood.asList()
    ghostPos = [(G.getPosition()[0], G.getPosition()[1]) for G in newGhostStates]
    scared = min(newScaredTimes) > 0
    if currentGameState.isLose():
        return float('-inf')
    if newPos in ghostPos:
        return float('-inf')

    # 计算食物距离和ghost距离分数
    closestFoodDist = sorted(newFood, key=lambda fDist: util.manhattanDistance(fDist, newPos))
    closestGhostDist = sorted(ghostPos, key=lambda gDist: util.manhattanDistance(gDist, newPos))
    # closestCapsules = sorted(newCapsules,key=lambda cDist: util.manhattanDistance(cDist, newPos))
    score = 0
    fd = lambda fDis: util.manhattanDistance(fDis, newPos)
    gd = lambda gDis: util.manhattanDistance(gDis, newPos)
    # cd = lambda cDis: util.manhattanDistance(cDis, newPos)
   
    # 越接近鬼得分越低
    if gd(closestGhostDist[0]) <3: 
        score-=300
    if gd(closestGhostDist[0]) <2:
        score-=1000
    if gd(closestGhostDist[0]) <1:
        return float('-inf')
    
    # 离胶囊越近得分越高
    # if len(closestCapsules) != 0 and cd(closestCapsules[0]) <3:
    #     score += 200
    # if len(closestCapsules) != 0 and cd(closestCapsules[0]) <2:
    #     score += 200
    if len(currentGameState.getCapsules()) < 2:
        score+=100
    
    # 越接近食物得分越高
    if len(closestFoodDist)==0 or len(closestGhostDist)==0 :
        score += scoreEvaluationFunction(currentGameState) + 10
    else:
        score += (scoreEvaluationFunction(currentGameState) + 10/fd(closestFoodDist[0]) + 1/gd(closestGhostDist[0]) + 1/gd(closestGhostDist[-1])  )

    return score


# Abbreviation
better = betterEvaluationFunction
