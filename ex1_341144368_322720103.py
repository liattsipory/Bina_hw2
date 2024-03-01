import search_341144368_322720103
import copy
from itertools import product
import random
import math
import numpy as np

ids = ["341144368", "322720103"]


class OnePieceProblem(search_341144368_322720103.Problem):
    """This class implements a medical problem according to problem description file"""

    def __init__(self, initial):
        """"
        Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""
        self.initial_dict = copy.deepcopy(initial)
        self.map = initial["map"]
        self.rows = len(self.map)
        self.cols = len(self.map[0]) if self.rows > 0 else 0
        self.location_map = [[{} for _ in range(self.cols)] for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                self.location_map[i][j] = {'pirate_ships': [],
                                           "marine_ships": [],
                                           'treasures': []}
        self.pirate_ships = initial["pirate_ships"]
        self.treasures = initial["treasures"]
        self.marine_ships = initial["marine_ships"]

        k = 0
        for p_ship, p_loc in self.pirate_ships.items():
            i, j = p_loc
            self.location_map[i][j]['pirate_ships'].append(p_ship)
            # 2 free hands : {"pirate_ship_1":(2, 0, [])}
            self.pirate_ships[p_ship] = (p_loc, ['', ''])
            k += 1

        for m_ship, m_path in self.marine_ships.items():
            reverse_path = m_path[::-1][1:-1]
            all_path = m_path + reverse_path
            i, j = m_path[0]
            self.location_map[i][j]['marine_ships'].append(m_ship)
            self.marine_ships[m_ship] = (all_path, 0)

        for treasure, t_loc in self.treasures.items():
            i, j = t_loc
            self.location_map[i][j]['treasures'].append(treasure)

        initial = (self.location_map, self.pirate_ships, self.treasures, self.marine_ships)
        search_341144368_322720103.Problem.__init__(self, initial)

    def has_this_treasure(self, ship_info, treasure):
        hands = ship_info[1]
        for h in hands:
            if h == treasure:
                return True
        return False

    def has_free_hand(self, ship_info):
        hands = ship_info[1]
        return (hands[0] == '' or hands[1] == '')


    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        location_map = state[0]
        map_rows = len(location_map)
        map_columns = len(location_map[0])
        pirate_ships = state[1]
        actions_dict = {}
        # sail
        for pirate_ship, ship_info in pirate_ships.items():
            actions_dict[pirate_ship]=[]
            i,j = ship_info[0]

            # left
            if j != 0:
                if self.map[i][j - 1] != 'I':
                    actions_dict[pirate_ship].append(("sail", pirate_ship, (i, j - 1)))
                else:  # island to the left
                    if (len(location_map[i][j - 1]['treasures']) > 0 and self.has_free_hand(ship_info)):
                        for t in location_map[i][j - 1]['treasures']:
                            if not (self.has_this_treasure(ship_info, t)):
                                actions_dict[pirate_ship].append(("collect_treasure", pirate_ship, t))
                                break

            # right
            if j != map_columns - 1:
                if self.map[i][j + 1] != 'I':
                    actions_dict[pirate_ship].append(("sail", pirate_ship, (i, j + 1)))
                else:
                    # there is a treasure
                    if (len(location_map[i][j+1]['treasures']) > 0 and self.has_free_hand(ship_info)):
                        for t in location_map[i][j+1]['treasures']:
                            if not (self.has_this_treasure(ship_info, t)):
                                actions_dict[pirate_ship].append(("collect_treasure", pirate_ship, t))
                                break

            # up
            if i != 0:
                if self.map[i - 1][j] != 'I':
                    actions_dict[pirate_ship].append(("sail", pirate_ship, (i - 1, j)))

                else:
                    if (len(location_map[i - 1][j]['treasures']) > 0 and self.has_free_hand(ship_info)):
                        for t in location_map[i - 1][j]['treasures']:
                            if not (self.has_this_treasure(ship_info, t)):
                                actions_dict[pirate_ship].append(("collect_treasure", pirate_ship, t))
                                break

            # down
            if i != map_rows - 1:
                if self.map[i + 1][j] != 'I':
                    actions_dict[pirate_ship].append(("sail", pirate_ship, (i + 1, j)))

                else:
                    if (len(location_map[i + 1][j]['treasures']) > 0 and self.has_free_hand(ship_info)):
                        for t in location_map[i + 1][j]['treasures']:
                            if not (self.has_this_treasure(ship_info,t)):
                                actions_dict[pirate_ship].append(("collect_treasure", pirate_ship, t))
                                break
            # deposit_treasure
            if self.map[i][j] == 'B' and (ship_info[1][0] != '' or ship_info[1][1] != ''):
                actions_dict[pirate_ship].append(("deposit_treasures", pirate_ship))

            # wait
            actions_dict[pirate_ship].append(("wait", pirate_ship))

        actions = list(actions_dict.values())
        action_combinations = list(product(*actions))

        return action_combinations

    def marine_moves(self, location_map, marine_ships):
        for marine_ship, marine_info in marine_ships.items():
            m_path = marine_info[0]
            # current location
            x, y = m_path[marine_info[1]]

            location_map[x][y]['marine_ships'].remove(marine_ship)
            # changing the location
            if marine_info[1] == len(m_path) - 1:
                marine_ships[marine_ship] = (marine_info[0], 0)
            else:
                next_index = marine_info[1]+1
                marine_ships[marine_ship] = (marine_info[0], next_index)
            x, y = m_path[marine_ships[marine_ship][1]]
            location_map[x][y]['marine_ships'].append(marine_ship)

        return location_map, marine_ships

    def result(self, state, action_comb):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""

        # moving the marines to the next step
        state_pirates = copy.deepcopy(state[1])
        state_treasures = copy.deepcopy(state[2])
        state_locations, state_marine_ships = self.marine_moves(copy.deepcopy(state[0]),copy.deepcopy(state[3]))

        if action_comb not in self.actions(state):
            return ("action not in possible actions from this state")

        for action in action_comb:

            pirate_ship = action[1]
            x,y = state_pirates[pirate_ship][0]

            if action[0] == 'sail':

                state_locations[x][y]['pirate_ships'].remove(pirate_ship)
                x, y = action[2]
                state_locations[x][y]['pirate_ships'].append(pirate_ship)
                state_pirates[pirate_ship] = ((x, y), state_pirates[pirate_ship][1])

            # if marine and pirate are in the same cell after they both move
            marine_flag = len(state_locations[x][y]['marine_ships'])

            if action[0] == 'collect_treasure':
                # if there is no marine ship in the same cell as me
                if not marine_flag:
                    treasure = action[2]
                    # choosing hand
                    if state_pirates[pirate_ship][1][0] == '':
                        state_pirates[pirate_ship] = (state_pirates[pirate_ship][0], [treasure,''])
                    else:

                        left_hand_treasure = state_pirates[pirate_ship][1][0]
                        state_pirates[pirate_ship] = (state_pirates[pirate_ship][0], [left_hand_treasure, treasure])


            if action[0] == 'deposit_treasures':
                for t in state_pirates[pirate_ship][1]:
                    if t == '':
                        continue
                    else:
                        t_x, t_y = self.treasures[t]
                        if t in state_treasures.keys():
                            del state_treasures[t]
                            state_locations[t_x][t_y]['treasures'].remove(t)
                state_pirates[pirate_ship] = (state_pirates[pirate_ship][0], ['', ''])

            # marines taking the treasures
            if marine_flag:
                #checking for treasure in 2 hands
                for i in range(2):
                    if state_pirates[pirate_ship][1][i] != '':
                        treasure = state_pirates[pirate_ship][1][i]
                        # original location of the treasure
                        t_x, t_y = self.treasures[treasure]
                        if treasure not in state_locations[t_x][t_y]['treasures']:
                            state_locations[t_x][t_y]['treasures'].append(treasure)

                state_pirates[pirate_ship] = (state_pirates[pirate_ship][0], ['', ''])

        state = (state_locations, state_pirates, state_treasures, state_marine_ships)
        return state

    def goal_test(self, state):
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""
        pirate_ships=state[1]
        state_treasures = state[2]
        treasure_empty = not bool(state_treasures)
        num_ships = len(self.pirate_ships)
        count = 0
        for pirate_ship, ship_info in pirate_ships.items():
            i, j = ship_info[0]
            if self.map[i][j] == 'B':
                count+=1
        if count == num_ships and treasure_empty:
            return True

        else:
            return False


    def h_1(self, node):
        state = node.state
        pirate_ships = state[1]
        num_pirate_ships = len(pirate_ships)
        state_treasures = set(state[2].keys())
        treasures_in_hands = set()
        for _, hands_list in pirate_ships.values():
            treasures_in_hands.update(set(hands_list))
        num_treasures = len(state_treasures - treasures_in_hands)

        return num_treasures/num_pirate_ships

    def h_2(self, node):
        """
                    [['S', 'S', 'I', 'S'],
                    ['S', 'B', 'S', 'S'],
                    ['S', 'S', 'I', 'I'],
                    ['S', 'S', 'I', 'I']],
        # self.treasures = all the treasures
        # state.treasures = all the not deposited treasures
        """
        distances_sum  = 0
        state_pirate_ships = node.state[1]
        num_pirate_ships = len(state_pirate_ships)
        state_treasures = node.state[2]
        first_key =  list(self.pirate_ships.keys())[0]
        base_loc = self.pirate_ships[first_key][0]
        # hands = state_pirate_ships.values()[1]
        treasure_locations = {}
        for t, t_initial_loc in state_treasures.items():
            treasure_locations[t] = []
            min_dist = float('inf')

            #checking if the treasure is on ship
            for p, p_info in state_pirate_ships.items():
                hands = p_info[1]
                if t in hands:
                    treasure_locations[t].append(p_info[0])
            # it's not in the hands
            if len(treasure_locations[t]) == 0:
                i, j = t_initial_loc
                seas = []
                if j!=0:
                    if self.map[i][j-1] == 'S':
                        seas.append((i, j-1))
                if j != self.cols-1:
                    if self.map[i][j+1] == 'S':
                        seas.append((i, j+1))
                if i != self.rows - 1:
                    if self.map[i+1][j] == 'S':
                        seas.append((i+1, j))
                if i != 0:
                    if self.map[i-1][j] == 'S':
                        seas.append((i-1, j))
                treasure_locations[t] = seas

            if len(treasure_locations[t]) == 0:
                return float('inf')

            # list of all locations where treasure is (adjacent seas / pirate ships that carry it)
            for loc in treasure_locations[t]:
                dist = abs(loc[0]-base_loc[0]) + abs(loc[1]-base_loc[1])
                if loc == base_loc:
                    dist += 1
                if dist < min_dist:
                    min_dist = dist

            treasure_locations[t] = min_dist
            distances_sum+=min_dist

        return distances_sum/num_pirate_ships


    def h(self, node):
        """ This is the heuristic. It gets a node (not a state,
        state can be accessed via node.state)
        and returns a goal distance estimate"""
        return self.h_2(node)

    """Feel free to add your own functions
    (-2, -2, None) means there was a timeout"""


def create_onepiece_problem(game):
    return OnePieceProblem(game)
