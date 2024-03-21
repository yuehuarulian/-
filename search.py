# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

# def mydfs(actions_mincos:list, actions:util.Stack, state:tuple[int,int], problem:SearchProblem, vis:dict, cos:int):
#     if problem.isGoalState(state):
#         if len(actions_mincos) > cos or len(actions_mincos) == 0:
#             actions_mincos.clear()
#             actions_mincos.extend(actions.list)
#         return
#     # 剪枝
#     if cos >= len(actions_mincos) and len(actions_mincos) != 0:
#         return
#     successors = problem.getSuccessors(state)
#     # 遍历下一步可能的状态
#     for next_state, action, cost in successors:
#         # 如果已经访问过就跳过
#         if vis.get(next_state, False):
#             continue
#         actions.push(action)
#         vis[next_state] = True
#         cos += cost
#         mydfs(actions_mincos, actions, next_state, problem, vis, cos)
#         # 回溯
#         actions.pop()
#         vis.pop(next_state)
#         cos -= cost
#     return

# def depthFirstSearch(problem: SearchProblem):
#     """
#     Search the deepest nodes in the search tree first.
#     Your search algorithm needs to return a list of actions that reaches the
#     goal. Make sure to implement a graph search algorithm.
#     To get started, you might want to try some of these simple commands to
#     understand the search problem that is being passed in:
#     print("Start:", problem.getStartState())
#     print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
#     print("Start's successors:", problem.getSuccessors(problem.getStartState()))
#     """
#     "*** YOUR CODE HERE ***"
#     actions = util.Stack()  # 动作
#     actions_mincos = []  # 记录消耗最小的动作  记录最小消耗
#     start = problem.getStartState()  # 起始状态
#     vis = {start: True}  # 标记是否已访问的字典
#     mydfs(actions_mincos, actions, start, problem, vis, 0)  # 深搜
#     # print(actions_mincos)
#     return actions_mincos  # 返回动作列表
#     util.raiseNotDefined()

def mydfs(actions, state, problem, vis):
    # 找到目标状态返回
    if problem.isGoalState(state):
        return True
    # 获取下一步可能的所有状态
    successors = problem.getSuccessors(state)
    # 遍历下一步可能的状态
    for nxt_state, action, _ in successors:
        # 如果该状态已经访问，跳过
        if vis.get(nxt_state, False):
            continue
        # 记录动作
        actions.append(action)
        # 标记状态已访问
        vis[nxt_state] = True
        # 深搜，如果找到目标状态返回
        if mydfs(actions, nxt_state, problem, vis):
            return True
        # 回溯动作
        actions.pop()
    return False

def depthFirstSearch(problem: SearchProblem):
    actions = []  # 动作
    start = problem.getStartState()  # 起始状态
    vis = {start: True}  # 标记是否已访问的字典
    mydfs(actions, start, problem, vis)  # 深搜
    return actions  # 返回动作列表

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    myqueue = util.Queue()
    actions = []
    start = problem.getStartState()  # 起始状态
    vis = {start: True}  # 标记是否已访问的字典
    myqueue.push([start, []])
    while not myqueue.isEmpty():
        state, actions = myqueue.pop()
        # 已经找到
        if(problem.isGoalState(state)):
            return actions
        successors = problem.getSuccessors(state)
        # 遍历所有方向
        for state_t, direction, cos in successors:
            # 是否已经经过
            if vis.get(state_t, False):
                continue
            myqueue.push([state_t, actions + [direction]])
            vis[state_t] = True
    return actions
    util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    myqueue = util.PriorityQueue()
    actions = []
    start = problem.getStartState()  # 起始状态
    vis = {start: True}  # 标记是否已访问的字典
    myqueue.push([start, [], 0], 0)  # 当前状态，活动列表，代价     优先级
    while not myqueue.isEmpty():
        state, actions, cost = myqueue.pop()
        # 已经找到
        if problem.isGoalState(state):
            return actions
        successors = problem.getSuccessors(state)
        # 遍历所有方向
        for state_t, direction, cost_t in successors:
            # 是否已经经过
            if state_t in vis:
                continue
            myqueue.push([state_t, actions + [direction], cost_t + cost], cost_t + cost)
            if not problem.isGoalState(state_t):
                vis[state_t] = True
    return actions
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    myqueue = util.PriorityQueue()
    actions = []
    start = problem.getStartState()  # 起始状态
    vis = {start: True}  # 标记是否已访问的字典
    myqueue.push([start, [], 0], 0)  # 当前状态，活动列表，代价     优先级
    while not myqueue.isEmpty():
        state, actions, cost = myqueue.pop()
        # 已经找到
        if problem.isGoalState(state):
            return actions
        
        # 遍历所有方向
        for state_t, direction, cost_t in problem.getSuccessors(state):
            # 是否已经经过
            if state_t in vis:
                continue
            
            #状态里存储当前路径的代价，而优先队列里比较 代价+启发式函数值
            myqueue.push([state_t, actions + [direction], cost_t + cost], cost_t + cost + heuristic(state_t, problem))
            if not problem.isGoalState(state_t):
                vis[state_t] = True
    return actions
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
