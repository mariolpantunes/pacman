from abc import ABC, abstractmethod
from queue import PriorityQueue


class SearchDomain(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def actions(self, state):
        pass

    @abstractmethod
    def result(self, state, action):
        pass

    @abstractmethod
    def cost(self, state, action):
        pass

    @abstractmethod
    def heuristic(self, state, goal_state):
        pass

    @abstractmethod
    def equivalent(self, state1, state2):
        pass

    @abstractmethod
    def satisfies(self, state, goal):
        pass


# Search problem with multiple goals
class SearchProblem:
    def __init__(self, domain, initial, goals):
        self.domain = domain
        self.initial = initial
        self.goals = goals
    
    def goal_test(self, state):
        for goal in self.goals:
            if self.domain.satisfies(state, goal):
                return True
        return False
        

class SearchNode:
    def __init__(self,state,parent,heuristic,cost=0):
        self.state = state
        self.parent = parent
        self.heuristic = heuristic
        self.cost = cost
        self.f = cost + heuristic
    
    def isParent(self, state, domain):
        if domain.equivalent(self.state, state):
            return True
        if self.parent is None:
            return False
        return self.parent.isParent(state, domain) 
    
    def __str__(self):
        return "SN(" + str(self.state) + "," + str(self.parent) + "," + str(self.cost) + ","+ str(self.heuristic) + ")"
    
    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

    def __eq__(self, other):
        if other is None:
            return False
        return (self.cost + self.heuristic) == (other.cost + other.heuristic)

# Support for multiple goals
# And uses a Priority Queue to speed up the retrieval of open nodes
# The code has a maximum number of iteration in order to decrease time of execution.
class SearchTree:
    def __init__(self, problem, it=4e3): 
        self.problem = problem
        self.open_nodes = PriorityQueue()
        self.close_nodes = set()
        self.it = it
        heuristics = []
        for goal in problem.goals:
            heuristics.append(problem.domain.heuristic(problem.initial, goal))
        node = SearchNode(problem.initial, None, min(heuristics))
        self.open_nodes.put((node.f, node))

    def get_cost_path(self, node):
        return (node.cost, self.get_path(node))

    def get_path(self,node):
        if node.parent == None:
            return [node.state]
        path = self.get_path(node.parent)
        path += [node.state]
        return(path)

    def search(self):
        while not self.open_nodes.empty():
            node = self.open_nodes.get()[1]
            if self.problem.goal_test(node.state) or self.it == 0:
                return self.get_cost_path(node)
            self.close_nodes.add(node.state)
            
            lnewnodes = []
            for a in self.problem.domain.actions(node.state):
                newstate = self.problem.domain.result(node.state,a)
                #if not node.isParent(newstate, self.problem.domain):
                # Since Manhattan Distance is Monotonic
                # we can simply use a close set to discard explored nodes
                # node.isParent() was the most expensive function of the code
                if newstate not in self.close_nodes:
                    heuristics = []
                    for goal in self.problem.goals:
                        heuristics.append(self.problem.domain.heuristic(self.problem.initial, goal))
                    cost = node.cost + self.problem.domain.cost(node.state, a)
                    lnewnodes += [SearchNode(newstate, node, min(heuristics), cost)]
            
            for n in lnewnodes:
                self.open_nodes.put((n.f, n))
            self.it -= 1
        return None
