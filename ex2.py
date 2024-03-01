import copy
from itertools import product
import random

ids = ["341144368", "322720103"]

RESET_PENALTY = 2
DEPOSIT_SCORE = 4
MARINE_COLLISION_PENALTY = 1
OUR_TREASURE_SCORE = 3


class OptimalPirateAgent:
    def __init__(self, initial):
        """
        "optimal": True,
        "infinite": False,
        "map": [
            ['B', 'S', 'S', 'S', 'I'],
            ['I', 'S', 'I', 'S', 'I'],
            ['S', 'S', 'I', 'S', 'S'],
            ['S', 'I', 'S', 'S', 'S'],
            ['S', 'S', 'S', 'S', 'I']
        ],
        "pirate_ships": {'pirate_ship_1': {"location": (0, 0),
                                           "capacity": 2}
                         },
        "treasures": {'treasure_1': {"location": (4, 4),
                                     "possible_locations": ((4, 4),),
                                     "prob_change_location": 0.5}
                      },
        "marine_ships": {'marine_1': {"index": 0,
                                      "path": [(2, 3), (2, 3)]}},
        "turns to go": 100
        """
        self.initial = initial
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
        """{'pirate_ship_1': {"location": (0, 0),
                                           "capacity": 2}"""
        self.treasures = initial["treasures"]
        """{'treasure_1': {"location": (4, 4),
                                     "possible_locations": ((4, 4),),
                                     "prob_change_location": 0.5}"""
        self.marine_ships = initial["marine_ships"]
        """{'marine_1': {"index": 0,
                                      "path": [(2, 3), (2, 3)]}}"""
        for p_ship in self.pirate_ships.keys():
            i, j = self.pirate_ships[p_ship]['location']
            self.location_map[i][j]['pirate_ships'].append(p_ship)

        for m_ship in self.marine_ships.keys():
            index = self.marine_ships[m_ship]['index']
            i, j = self.marine_ships[m_ship]['path'][index]
            self.location_map[i][j]['marine_ships'].append(m_ship)

        for treasure in self.treasures.keys():
            i, j = self.treasures[treasure]['location']
            self.location_map[i][j]['treasures'].append(treasure)

        initial = (self.location_map, self.pirate_ships, self.treasures, self.marine_ships)

    def act(self, s):
        state = self.create_state(s)
        r = self.r(state)
        #self.print_map(state[0])
        max = float('-inf')
        value = r
        for action in self.possible_actions(state):  # (sail, deposit, wait)
            for p_index, pirate_ship in enumerate(self.pirate_ships.keys()):
                treasures_num = 2 - state[1][pirate_ship]['capacity']
                if action[p_index][0] == 'sail':
                    i_to, j_to = action[p_index][2]
                    for marine in state[3]:
                        marine_stats = state[3][marine]
                        index = marine_stats["index"]
                        if len(marine_stats['path']) == 1:
                            continue
                        if index == 0:
                            # possible marine locations
                            if marine_stats["path"][index] == (i_to, j_to) or marine_stats["path"][index + 1] == (
                            i_to, j_to):
                                value += (0.5 * self.r_new(True, treasures_num, action[p_index][0]) +
                                          0.5 * self.r_new(False, treasures_num, action[p_index][0]))
                        elif index == len(marine_stats['path']) - 1:
                            # possible marine locations
                            if marine_stats["path"][index] == (i_to, j_to) or marine_stats["path"][index - 1] == (
                            i_to, j_to):
                                value += (0.5 * self.r_new(True, treasures_num, action[p_index][0]) +
                                          0.5 * self.r_new(False, treasures_num, action[p_index][0]))
                        elif index > 0 and len(marine_stats['path']) > 2:
                            # possible marine locations
                            if marine_stats["path"][index] == (i_to, j_to) or marine_stats["path"][index + 1] == (
                                    i_to, j_to) or marine_stats["path"][index - 1] == (i_to, j_to):
                                value += (0.3333 * self.r_new(True, treasures_num, action[p_index][0]) +
                                          0.3333 * self.r_new(False, treasures_num, action[p_index][0])
                                          + 0.3333 * self.r_new(True, treasures_num, action[p_index][0]))
            if value > max:
                max = value
                best_action = action

        return best_action

    def r(self, state):
        raise NotImplemented


class PirateAgent:
    def __init__(self, initial):
        """
        {
        "optimal": True,
        "infinite": True,
        "gamma": 0.9,
        "map": [
            ['S', 'S', 'I', 'S'],
            ['S', 'S', 'I', 'S'],
            ['B', 'S', 'S', 'S'],
            ['S', 'S', 'I', 'S']
        ],
        "pirate_ships": {'pirate_ship_2': {"location": (2, 0),
                                           "capacity": 2}
                         },
        "treasures": {'treasure_1': {"location": (0, 2),
                                     "possible_locations": ((0, 2), (1, 2), (3, 2)),
                                     "prob_change_location": 0.1}
                      },
        "marine_ships": {'marine_1': {"index": 0,
                                      "path": [(1, 1)]}},
    }"""
        self.initial = initial
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
        first_pship = list(self.pirate_ships.keys())[0]
        self.base_location = self.pirate_ships[first_pship]['location']
        """{'pirate_ship_1': {"location": (0, 0),
                                           "capacity": 2}"""
        self.treasures = initial["treasures"]
        """{'treasure_1': {"location": (4, 4),
                                     "possible_locations": ((4, 4),),
                                     "prob_change_location": 0.5}"""
        self.marine_ships = initial["marine_ships"]
        """{'marine_1': {"index": 0,
                                      "path": [(2, 3), (2, 3)]}}"""
        for p_ship in self.pirate_ships.keys():
            i, j = self.pirate_ships[p_ship]['location']
            self.location_map[i][j]['pirate_ships'].append(p_ship)

        for m_ship in self.marine_ships.keys():
            index = self.marine_ships[m_ship]['index']
            i, j = self.marine_ships[m_ship]['path'][index]
            self.location_map[i][j]['marine_ships'].append(m_ship)

        for treasure in self.treasures.keys():
            i, j = self.treasures[treasure]['location']
            self.location_map[i][j]['treasures'].append(treasure)

        initial = (self.location_map, self.pirate_ships, self.treasures, self.marine_ships)

    def create_state(self, state):
        initial = copy.deepcopy(state)
        self.initial = initial
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

        """{'pirate_ship_1': {"location": (0, 0),
                                           "capacity": 2}"""
        self.treasures = initial["treasures"]
        """{'treasure_1': {"location": (4, 4),
                                     "possible_locations": ((4, 4),),
                                     "prob_change_location": 0.5}"""
        self.marine_ships = initial["marine_ships"]
        """{'marine_1': {"index": 0,
                                      "path": [(2, 3), (2, 3)]}}"""
        for p_ship in self.pirate_ships.keys():
            i, j = self.pirate_ships[p_ship]['location']
            self.location_map[i][j]['pirate_ships'].append(p_ship)

        for m_ship in self.marine_ships.keys():
            index = self.marine_ships[m_ship]['index']
            i, j = self.marine_ships[m_ship]['path'][index]
            self.location_map[i][j]['marine_ships'].append(m_ship)

        for treasure in self.treasures.keys():
            i, j = self.treasures[treasure]['location']
            self.location_map[i][j]['treasures'].append(treasure)

        return (self.location_map, self.pirate_ships, self.treasures, self.marine_ships)

    def possible_actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        location_map = state[0]
        map_rows = len(location_map)
        map_columns = len(location_map[0])
        pirate_ships = state[1]
        actions_dict = {}
        marine_paths = []
        for marine in state[3].keys():
            marine_paths.append(state[3][marine]['path'])
        marine_paths = sum(marine_paths, [])
        # sail
        for pirate_ship in pirate_ships.keys():
            actions_dict[pirate_ship] = []
            i, j = pirate_ships[pirate_ship]['location']
            # left
            if j != 0:
                if self.map[i][j - 1] != 'I' and (i, j-1) not in marine_paths:
                    actions_dict[pirate_ship].append(("sail", pirate_ship, (i, j - 1)))
                else:  # island to the left
                    if len(location_map[i][j - 1]['treasures']) > 0 and  pirate_ships[pirate_ship]['capacity'] > 0:
                        for t in location_map[i][j - 1]['treasures']:
                            actions_dict[pirate_ship].append(("collect", pirate_ship, t))
                            break

            # right
            if j != map_columns - 1:
                if self.map[i][j + 1] != 'I' and (i, j+1) not in marine_paths:
                    actions_dict[pirate_ship].append(("sail", pirate_ship, (i, j + 1)))
                else:
                    # there is a treasure
                    if (len(location_map[i][j + 1]['treasures']) > 0 and pirate_ships[pirate_ship]['capacity'] > 0):
                        for t in location_map[i][j + 1]['treasures']:
                            actions_dict[pirate_ship].append(("collect", pirate_ship, t))
                            break

            # up
            if i != 0:
                if self.map[i - 1][j] != 'I' and (i-1, j) not in marine_paths:
                    actions_dict[pirate_ship].append(("sail", pirate_ship, (i - 1, j)))

                else:
                    if (len(location_map[i - 1][j]['treasures']) > 0 and pirate_ships[pirate_ship]['capacity'] > 0):
                        for t in location_map[i - 1][j]['treasures']:
                            actions_dict[pirate_ship].append(("collect", pirate_ship, t))
                            break

            # down
            if i != map_rows - 1:
                if self.map[i + 1][j] != 'I' and (i+1, j) not in marine_paths:
                    actions_dict[pirate_ship].append(("sail", pirate_ship, (i + 1, j)))

                else:
                    if len(location_map[i + 1][j]['treasures']) > 0 and pirate_ships[pirate_ship]['capacity'] > 0:
                        for t in location_map[i + 1][j]['treasures']:
                            actions_dict[pirate_ship].append(("collect", pirate_ship, t))
                            break

            # deposit_treasure
            if self.map[i][j] == 'B' and pirate_ships[pirate_ship]['capacity'] < 2:
                actions_dict[pirate_ship].append(("deposit", pirate_ship))

            # wait
            actions_dict[pirate_ship].append(("wait", pirate_ship))

            # reset
            actions_dict[pirate_ship].append(("reset", pirate_ship))

            # terminate
            actions_dict[pirate_ship].append(("terminate", pirate_ship))

        actions = list(actions_dict.values())
        action_combinations = list(product(*actions))

        return action_combinations


    def r(self, state):
        location_map = state[0]
        pirate_ships = state[1]
        score = 0
        for p_ship in pirate_ships.keys():
            ship_treasures_num = (2 - pirate_ships[p_ship]['capacity'])
            i, j = pirate_ships[p_ship]['location']
            if len(location_map[i][j]['marine_ships']) > 0:
                score -= (MARINE_COLLISION_PENALTY + ship_treasures_num * OUR_TREASURE_SCORE)
            if self.map[i][j] == 'B':
                score += ship_treasures_num * DEPOSIT_SCORE
            score += ship_treasures_num * OUR_TREASURE_SCORE
        return score

    def r_new(self, marine_meet, treasures_num, action):
        score = 0
        if action == 'deposit':
            score += treasures_num * DEPOSIT_SCORE
            treasures_num = 0
        elif action == 'collect':
            score += OUR_TREASURE_SCORE
            treasures_num += 1
        if marine_meet:
            score -= (MARINE_COLLISION_PENALTY + treasures_num * OUR_TREASURE_SCORE)
        return score

    def print_map(self, arr):
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
            print()
        print()

    def move_vertically(self, x, ship_x, ship_y, p_ship, action_with_locations_list):
        action = None
        if x > 0:
            if ('sail', (ship_x - 1, ship_y)) in action_with_locations_list:
                action = ('sail', p_ship, (ship_x - 1, ship_y))
        elif x < 0:
            if ('sail', (ship_x + 1, ship_y)) in action_with_locations_list:
                action = ('sail', p_ship, (ship_x + 1, ship_y))
        return action

    def move_horizontally(self, y, ship_x, ship_y, p_ship, action_with_locations_list):
        action = None
        if y > 0:
            if ('sail', (ship_x, ship_y - 1)) in action_with_locations_list:
                action = ('sail', p_ship, (ship_x, ship_y - 1))
        elif y < 0:
            if ('sail', (ship_x, ship_y + 1)) in action_with_locations_list:
                action = ('sail', p_ship, (ship_x, ship_y + 1))
        return action

    def act(self, s):
        state = self.create_state(s)
        #self.print_map(state[0])
        actions_order = ['deposit', 'collect', 'sail', 'wait', 'reset', 'terminate']
        treasures = state[2].keys()
        treasure_location = state[2][list(treasures)[0]]['location']
        best_action = []

        for p_index, p_ship in enumerate(state[1].keys()):
            ship_x, ship_y = state[1][p_ship]['location']
            full_action_list = [action[p_index] for action in self.possible_actions(state)]
            action_names_list = [action[0] for action in full_action_list]
            #print(full_action_list)
            if 'deposit' in action_names_list:
                deposit_index = action_names_list.index('deposit')
                best_action.append(full_action_list[deposit_index])
                break
            if 'collect' in action_names_list:
                #print('COLLECT IT')
                collect_index = action_names_list.index('collect')
                best_action.append(full_action_list[collect_index])
                break
            # we have something to deposit
            if state[1][p_ship]['capacity'] < 2:
                distance = (ship_x - self.base_location[0], ship_y - self.base_location[1])
            else:
                distance = (ship_x - treasure_location[0], ship_y - treasure_location[1])
            x, y = distance
            """print(distance)
            print(self.possible_actions(state))"""
            action_with_locations_list = [(action[0], action[2]) if action[0] == 'sail' else () for action in full_action_list]
            # move in the right direction
            random_int = random.randint(0, 1)
            if random_int == 0 and x != 0:
                best_action.append(self.move_vertically(x, ship_x, ship_y, p_ship, action_with_locations_list))
                if best_action[-1] is not None:
                    break
            elif (random_int == 1 and y != 0) or (x==0 and y!=0):
                best_action.append(self.move_horizontally(y, ship_x, ship_y, p_ship, action_with_locations_list))
                if best_action[-1] is not None:
                    break
            else:
                best_action.append(self.move_vertically(x, ship_x, ship_y, p_ship, action_with_locations_list))
                if best_action[-1] is not None:
                    break
            # sail in any direction
            if best_action[-1] == None:
                best_action = best_action[:-1]
                if 'sail' in action_names_list:
                    sail_indices = [i for i, action in enumerate(action_names_list) if action == 'sail']
                    # Choose a random index from the list of sail_indices
                    random_sail_index = random.choice(sail_indices)
                    # Add the corresponding action to best_action list
                    best_action.append(full_action_list[random_sail_index])
                    break
            if 'wait' in action_names_list:
                wait_index = action_names_list.index('wait')
                best_action.append(full_action_list[wait_index])
        return tuple(best_action)

class InfinitePirateAgent:
    def __init__(self, initial, gamma):
        self.initial = initial
        self.gamma = gamma

    def act(self, state):
        raise NotImplemented

    def value(self, state):
        raise NotImplemented
