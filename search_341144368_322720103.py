import numpy as np
import heapq
from itertools import permutations, zip_longest
import copy
import ex1_341144368_322720103

"""Search (Chapters 3-4)

The way to use this code is to subclass Problem to create a class of problems,
then create problem instances and solve them with calls to the various search
functions."""


from utils import (
    is_in, argmin, argmax, argmax_random_tie, probability, weighted_sampler,
    memoize, print_table, open_data, Stack, FIFOQueue, PriorityQueue, name,
    distance
)

from collections import defaultdict
import math
import random
import sys
import bisect

infinity = float('inf')

# ______________________________________________________________________________


class Problem(object):

    """The abstract class for a formal problem.  You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal.  Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError
# ______________________________________________________________________________


class Node:

    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state.  Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node.  Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        # was state<state
        return self.path_cost < node.path_cost

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next = problem.result(self.state, action)
        return Node(next, self, action,
                    problem.path_cost(self.path_cost, self.state,
                                      action, next))

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)

# ______________________________________________________________________________

def hashable(state):
    hash_map = tuple(tuple((key, tuple(value)) for key, value in cell.items()) for row in state[0] for cell in row)

    hash_ships = tuple((key, (value[0],tuple(value[1]))) for key, value in state[1].items())
    hash_treasures = tuple((k, v) for k, v in state[2].items())

    return hash_map + hash_ships + hash_treasures

"""def print_map(arr):
    for row in arr:
        for cell in row:
            symbols = []
            for i, pirate_ship in enumerate(cell['pirate_ships'], start=1):
                symbols.append(pirate_ship[0])
                symbols.append(pirate_ship[-1])

            for i, marine_ship in enumerate(cell['marine_ships'], start=1):
                symbols.append(marine_ship[0])
                symbols.append(marine_ship[-1])
            for i, treasure in enumerate(cell['treasures'], start=1):
                symbols.append(treasure[0])
                symbols.append(treasure[-1])
            if not symbols:
                symbols.append('_')  # Empty cell
            print(''.join(symbols), end=' ')
        print()  """


def create_symmetric_states(state):
    location_map = state[0]
    pirate_ships = state[1]
    num_ships = len(pirate_ships.keys())
    if num_ships == 1:
        return []
    ship_permutations = list(permutations(pirate_ships.keys()))
    ship_permutations.remove(tuple(pirate_ships.keys()))
    symmetric=[]
    for perm in ship_permutations:
        new_location_map = copy.deepcopy(location_map)
        new_pirate_ships = copy.deepcopy(pirate_ships)
        # Create a copy of the original location map and pirate ships
        # Iterate over each ship name and its corresponding position
        for i, ship_name in enumerate(perm):
            original_ship = 'pirate_ship_' + str(i+1)
            x, y = pirate_ships[original_ship][0]
            new_location_map[x][y]['pirate_ships'].remove(original_ship)
            new_location_map[x][y]['pirate_ships'].append(ship_name)
            new_pirate_ships[ship_name] = pirate_ships[original_ship]
            new_state = (new_location_map, new_pirate_ships,copy.deepcopy(state[2]), copy.deepcopy(state[3]))
            symmetric.append(new_state)
    return symmetric


def astar_search(problem, h=None):
    # For 2 and more ships - turning the problem into easier one
    num_pirates = len(list(problem.initial_dict['pirate_ships'].keys()))
    if len(list(problem.initial_dict['pirate_ships'].keys()))>1:
        results = []
        lengths = []

        result = {}
        for i, ship in enumerate(problem.initial_dict['pirate_ships'].keys()):
            new_problem_dict = {'map': copy.deepcopy(problem.initial_dict['map']),
                                'pirate_ships': {ship: copy.deepcopy(problem.initial_dict['pirate_ships'][ship])},
                                'treasures': {},
                                'marine_ships': copy.deepcopy(problem.initial_dict['marine_ships'])
                                }
            treasures = {}

            # dividing the treasures between the ships
            for j, t in enumerate(problem.initial_dict['treasures'].keys()):
                if j%num_pirates == i:
                    treasures[t]=problem.initial_dict['treasures'][t]

            new_problem_dict['treasures']=treasures
            new_problem = ex1_341144368_322720103.create_onepiece_problem(new_problem_dict)
            result[ship] = astar_search(new_problem)
            lengths.append(len(result[ship]))

        max_result_length = max(lengths)


        for j in range(max_result_length):
            action = []
            for ship in problem.initial_dict['pirate_ships'].keys():
                if len(result[ship]) < j+1:
                    result[ship].append(('wait', ship))
                    action.append(result[ship][j])
                else:
                    action.append(result[ship][j][0])
            action = tuple(action)
            results.append(action)
        return results

    # Memoize the heuristic function for better performance
    h = memoize(h or problem.h, 'h')

    # Function to calculate f(n) = g(n) + h(n)
    # Memoize this function for better performance
    f = memoize(lambda n: n.path_cost + h(n), 'f')

    opened = [] # ordered by f(problem)
    root_node = Node(problem.initial)
    opened.append((f(root_node), root_node))
    closed = []
    distance = {}
    while bool(opened):
        _, sigma_node = heapq.heappop(opened)
        if sigma_node.state not in closed or sigma_node.path_cost < distance.get(hashable(sigma_node.state), -math.inf):
            closed.append(sigma_node.state)
            symmetric = create_symmetric_states(sigma_node.state)
            distance[hashable(sigma_node.state)] = sigma_node.path_cost
            for s in symmetric:
                distance[hashable(s)] = sigma_node.path_cost
                closed.append(s)
            if problem.goal_test(sigma_node.state):
                return sigma_node.solution()
            children = sigma_node.expand(problem)
            for i, c in enumerate(children):
                if h(c) < np.inf:
                    heapq.heappush(opened, (f(c), c))

    return None
