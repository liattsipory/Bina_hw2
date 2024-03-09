import copy
from itertools import product
import random

ids = ["341144368", "322720103"]

RESET_PENALTY = 2
DEPOSIT_SCORE = 4
MARINE_COLLISION_PENALTY = 1



def create_state(state):
    initial = copy.deepcopy(state)
    initial = initial
    map = initial["map"]
    rows = len(map)
    cols = len(map[0]) if rows > 0 else 0
    location_map = [[{} for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            location_map[i][j] = {'pirate_ships': [],
                                       "marine_ships": [],
                                       'treasures': []}
    pirate_ships = initial["pirate_ships"]

    """{'pirate_ship_1': {"location": (0, 0),
                                       "capacity": 2}"""
    treasures = initial["treasures"]
    """{'treasure_1': {"location": (4, 4),
                                 "possible_locations": ((4, 4),),
                                 "prob_change_location": 0.5}"""
    marine_ships = initial["marine_ships"]
    """{'marine_1': {"index": 0,
                                  "path": [(2, 3), (2, 3)]}}"""
    for p_ship in pirate_ships.keys():
        i, j = pirate_ships[p_ship]['location']
        location_map[i][j]['pirate_ships'].append(p_ship)

    for m_ship in marine_ships.keys():
        index = marine_ships[m_ship]['index']
        i, j = marine_ships[m_ship]['path'][index]
        location_map[i][j]['marine_ships'].append(m_ship)

    for treasure in treasures.keys():
        i, j = treasures[treasure]['location']
        location_map[i][j]['treasures'].append(treasure)

    return map, (location_map, pirate_ships, treasures, marine_ships)

def hashable(state):
    hash_map = tuple(tuple((key, tuple(value)) for key, value in cell.items()) for row in state[0] for cell in row)
    hash_ships = tuple((ship, tuple(dict['location'])) for ship, dict in state[1].items())
    hash_treasures = tuple((k, dict['location']) for k, dict in state[2].items())
    hash_marines = tuple((k, dict['index'], tuple(dict['path'])) for k, dict in state[3].items())
    return hash_map + hash_ships + hash_treasures + hash_marines


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
        self.marine_ships = initial['marine_ships']
        self.treasures = initial['treasures']
        # create state makes a full state like in hw1
        self.map, state = create_state(initial)
        self.turns = initial["turns to go"]
        first_ship = list(state[1].keys())[0]
        self.num_of_pirates = len(state[1].keys())
        self.pirate_names = list(state[1].keys())
        self.treasure_names = list(state[2].keys())
        self.marine_names = list(state[3].keys())
        self.base_location = state[1][first_ship]['location']
        self.rows = len(self.map)
        self.cols = len(self.map[0]) if self.rows > 0 else 0
        self.initial_vi_state = self.create_initial_state(initial)

        self.value_iteration_table, self.policy_table = self.value_iteration(state, self.map)



    def r(self, state):
        pirate_ships = state[0]
        treasures = state[1]
        marine_ships = state[2]
        score = 0
        # print(pirate_ships)
        """
            ['S', 'S', 'I', 'S'],
            ['S', 'S', 'I', 'S'],
            ['B', 'S', 'S', 'S'],
            ['S', 'S', 'I', 'S']
            """
        ship_treasures_num = (2 - pirate_ships['pirate_ship']['capacity'])
        i, j = pirate_ships['pirate_ship']['location']
        # print(i, j)
        marine_ships_current_locations = []
        for m_i, marine_ship in enumerate(marine_ships.keys()):
            marine_ship_og_name = self.marine_names[m_i]
            marine_ships_current_locations.append(self.marine_ships[marine_ship_og_name]['path'][marine_ships[marine_ship]['index']])
        if (i, j) in marine_ships_current_locations:
            score -= MARINE_COLLISION_PENALTY

        #score += ship_treasures_num

        return score

    def build_loc_map(self, state):
        location_map = [[{} for _ in range(self.cols)] for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                location_map[i][j] = {'pirate_ships': [],
                                      "marine_ships": [],
                                      'treasures': []}
        pirate_ships = state[0]

        """{'pirate_ship_1': {"location": (0, 0),
                                           "capacity": 2}"""
        treasures = state[1]
        """{'treasure_1': {"location": (4, 4),
                                     "possible_locations": ((4, 4),),
                                     "prob_change_location": 0.5}"""
        marine_ships = state[2]
        """{'marine_1': {"index": 0,
                                      "path": [(2, 3), (2, 3)]}}"""
        i, j = pirate_ships['pirate_ship']['location']
        location_map[i][j]['pirate_ships'].append('pirate_ship')

        for m_i, m_ship in enumerate(marine_ships.keys()):
            index = marine_ships[m_ship]['index']
            i, j = self.marine_ships[self.marine_names[m_i]]['path'][index]
            location_map[i][j]['marine_ships'].append(m_ship)

        for treasure in treasures.keys():
            i, j = treasures[treasure]['location']
            location_map[i][j]['treasures'].append(treasure)

        return location_map

    def possible_actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        location_map = self.build_loc_map(state)
        map_rows = len(location_map)
        map_columns = len(location_map[0])
        pirate_ships = state[0]
        # sail
        pirate_ship = 'pirate_ship'
        possible_actions = []
        i, j = pirate_ships[pirate_ship]['location']
        # left
        if j != 0:
            if self.map[i][j - 1] != 'I':
                possible_actions.append(("sail", pirate_ship, (i, j - 1)))
            else:  # island to the left
                if len(location_map[i][j - 1]['treasures']) > 0 and pirate_ships[pirate_ship]['capacity'] > 0:
                    for t in location_map[i][j - 1]['treasures']:
                        possible_actions.append(("collect", pirate_ship, t))
                        break
        # right
        if j != map_columns - 1:
            if self.map[i][j + 1] != 'I':
                possible_actions.append(("sail", pirate_ship, (i, j + 1)))
            else:
                # there is a treasure
                if (len(location_map[i][j + 1]['treasures']) > 0 and pirate_ships[pirate_ship]['capacity'] > 0):
                    for t in location_map[i][j + 1]['treasures']:
                        possible_actions.append(("collect", pirate_ship, t))
                        break
        # up
        if i != 0:
            if self.map[i - 1][j] != 'I':
                possible_actions.append(("sail", pirate_ship, (i - 1, j)))
            else:
                if (len(location_map[i - 1][j]['treasures']) > 0 and pirate_ships[pirate_ship]['capacity'] > 0):
                    for t in location_map[i - 1][j]['treasures']:
                        possible_actions.append(("collect", pirate_ship, t))
                        break

        # down
        if i != map_rows - 1:
            if self.map[i + 1][j] != 'I':
                possible_actions.append(("sail", pirate_ship, (i + 1, j)))

            else:
                if len(location_map[i + 1][j]['treasures']) > 0 and pirate_ships[pirate_ship]['capacity'] > 0:
                    for t in location_map[i + 1][j]['treasures']:
                        possible_actions.append(("collect", pirate_ship, t))
                        break

        # deposit_treasure
        if self.map[i][j] == 'B' and pirate_ships[pirate_ship]['capacity'] < 2:
            possible_actions.append(("deposit", pirate_ship))

        # wait
        possible_actions.append(("wait", pirate_ship))

        # reset
        possible_actions.append('reset')

        # terminate
        possible_actions.append('terminate')

        return possible_actions

    def create_vi_state(self, i, j, capacity, m_combination, t_combination):
        """rows = len(map)
        cols = len(map[0]) if rows > 0 else 0"""
        """" The output is:
        pirate_ships = {'pirate_ship': {"location": (i, j), "capacity": capacity}}
        treasures = {'treasure_i': {"location": (4, 4)}
        marine_ships = {'marine_i': {"index": 0}}
        """
        pirate_ships = {}
        pirate_ships['pirate_ship'] = {"location": (i, j), "capacity": capacity}
        treasures = {}
        for i, t in enumerate(t_combination):
            treasures[f'treasure_{i}'] = {"location": t}
        marine_ships = {}
        for i, m_i in enumerate(m_combination):
            marine_ships[f'marine_{i}'] = {"index": m_i}

        return (pirate_ships, treasures, marine_ships)

    def hashable_vi(self, state):
        #print(state[0])
        hash_ships = tuple((ship, dict['location'], dict['capacity']) for ship, dict in state[0].items())
        #print(state[1].items())
        hash_treasures = tuple((k, dict['location']) for k, dict in state[1].items())
        hash_marines = tuple((k, dict['index']) for k, dict in state[2].items())
        return hash_ships + hash_treasures + hash_marines

    def unhash(self, hashable_state):
        ships = {}
        treasures = {}
        marines = {}

        for item in hashable_state[0]:  # Ships
            ship_name, location, capacity = item
            ships[ship_name] = {"location": location, "capacity": capacity}

        for item in hashable_state[1]:  # Treasures
            item_name, location = item
            treasures[item_name] = {"location": location}

        for item in hashable_state[2]:  # Marines
            item_name, index = item
            marines[item_name] = {"index": index}

        return [ships, treasures, marines]

    def environment_probs(self, treasure_dict, marine_dict):
        """
        in this function i want to make all combinations of treasure locations and marine locations out of all possible options
and then calculate the probability of each combination
each combination should combine all the treasures and all the marine ships

        treasure dict = {'treasure_i': {"(0,1)": 0.95
        "(1,0)": 0.05}}
        marine dict = {'marine_i': {"0": 0.5, "1": 0.5}}
        """
        # Get all keys (locations) and probabilities for treasures
        all_treasure_keys = []
        all_treasure_probs = []

        for treasure in treasure_dict.values():
            keys = list(treasure.keys())
            probs = list(treasure.values())
            all_treasure_keys.append(keys)
            all_treasure_probs.append(probs)

        # Get all keys (locations) and probabilities for marines
        all_marine_keys = []
        all_marine_probs = []

        for marine in marine_dict.values():
            keys = list(marine.keys())
            probs = list(marine.values())
            all_marine_keys.append(keys)
            all_marine_probs.append(probs)

        # Generate all combinations of treasures and marines locations
        combinations = list(product(*all_treasure_keys, *all_marine_keys))

        # Calculate probabilities for each combination
        probabilities = []
        for combination in combinations:
            prob = 1.0
            for i, loc in enumerate(combination):
                if i < len(all_treasure_keys):
                    prob *= treasure_dict[list(treasure_dict.keys())[i]][loc]
                else:
                    prob *= marine_dict[list(marine_dict.keys())[i - len(all_treasure_keys)]][loc]
            probabilities.append(prob)

        # Combine combinations with probabilities
        combinations_with_prob = list(zip(combinations, probabilities))

        return combinations_with_prob

    def environment_changes(self, comb, treasure_dict, marine_dict, state):
        # comb (('(0,1)', '(0,2)', '0'), 0.475)
        new_state = copy.deepcopy(state)
        cur_marine_locs = []
        for i, t in enumerate(treasure_dict.keys()):
            new_state[1][t]['location'] = comb[0][i]
        for j, m in enumerate(marine_dict.keys()):
            new_index = comb[0][j + i + 1]
            new_state[2][m]['index'] = new_index
            cur_marine_locs.append(self.marine_ships[self.marine_names[j]]['path'][new_index])
        # print('env changes - state', new_state)
        return new_state, cur_marine_locs

    def create_state_from_input(self, input):
        first_pirate_ship = self.pirate_names[0]
        loc = input['pirate_ships'][first_pirate_ship]['location']
        pirate_ships = {}
        pirate_ships['pirate_ship'] = {"location": loc, "capacity": input['pirate_ships'][first_pirate_ship]['capacity']}
        treasures = {}
        for i, t in enumerate(input['treasures'].values()):
            treasures[f'treasure_{i}'] = {"location": t['location']}
        marine_ships = {}
        for i, marine in enumerate(input['marine_ships'].values()):
            marine_ships[f'marine_{i}'] = {"index": marine['index']}
        # ('pirate_ship', (2, 0), 2), ('treasure_0', (0, 2)), ('marine_0', 0)
        return (pirate_ships, treasures, marine_ships)

    def create_initial_state(self, input):
        i, j = self.base_location
        pirate_ships = {}
        first_pirate_ship = self.pirate_names[0]
        pirate_ships['pirate_ship'] = {"location": (i, j), "capacity": input['pirate_ships'][first_pirate_ship]['capacity']}
        treasures = {}
        for i, t in enumerate(input['treasures'].values()):
            treasures[f'treasure_{i}'] = {"location": t['location']}
        marine_ships = {}
        for i, marine in enumerate(input['marine_ships'].values()):
            marine_ships[f'marine_{i}'] = {"index": marine['index']}

        return (pirate_ships, treasures, marine_ships)

    def transition(self, state, action):
        """ state is:
        pirate_ships = {'pirate_ship': {"location": (i, j), "capacity": capacity}}
        treasures = {'treasure_i': {"location": (4, 4)}}
        marine_ships = {'marine_1': {"index": 0}
                        'marine_2': {"index": 1}
        """

        pirate_ships = state[0]
        treasures = state[1]
        marine_ships = state[2]
        probabilities_dict = {}
        action_name = action[0]
        treasure_dict = {}
        marine_dict = {}

        for t_i, treasure in enumerate(treasures.keys()):
            treasure_dict[treasure] = {}
            original_name = self.treasure_names[t_i]
            transition_prob = self.treasures[original_name]['prob_change_location']
            for location in self.treasures[original_name]['possible_locations']:
                if location == treasures[treasure]['location']:
                    treasure_dict[treasure][location] = (1-transition_prob) + transition_prob/len(self.treasures[original_name]['possible_locations'])
                else:
                    treasure_dict[treasure][location] = transition_prob/len(self.treasures[original_name]['possible_locations'])

        current_marine_locs = []
        for m_i, marine in enumerate(marine_ships.keys()):
            original_name = self.marine_names[m_i]
            if len(self.marine_ships[original_name]['path']) == 1:
                marine_dict[marine] = {0: 1}
            else:
                if marine_ships[marine]['index'] == 0:
                    marine_dict[marine] = {0: 0.5, 1: 0.5}
                elif marine_ships[marine]['index'] == len(self.marine_ships[original_name]['path']) - 1:
                    marine_dict[marine] = {len(self.marine_ships[original_name]['path']) - 1: 0.5, len(self.marine_ships[original_name]['path']) - 2: 0.5}
                else:
                    marine_dict[marine] = {marine_ships[marine]['index'] - 1: 0.3333, marine_ships[marine]['index']: 0.3333, marine_ships[marine]['index'] + 1: 0.3333}
        """Combinations with Probabilities:
        0 , 1
        (('(0,1)', '(0,2)', '0'), 0.475)
        (('(0,1)', '(0,2)', '1'), 0.475)"""
        current_loc = pirate_ships['pirate_ship']['location']
        env_combinations = self.environment_probs(treasure_dict, marine_dict)
        if action_name == 'wait':
            for comb in env_combinations:
                new_state, current_marine_locs = self.environment_changes(comb, treasure_dict, marine_dict, state)
                if current_loc in current_marine_locs:
                    new_state[0]['pirate_ship']['capacity'] = 2
                probabilities_dict[self.hashable_vi(new_state)] = comb[1]
        if action_name == 'sail':
            new_loc = action[2]
            for comb in env_combinations:
                new_state, current_marine_locs = self.environment_changes(comb, treasure_dict, marine_dict, state)
                new_state[0]['pirate_ship']['location'] = new_loc
                if new_loc in current_marine_locs:
                    new_state[0]['pirate_ship']['capacity'] = 2
                probabilities_dict[self.hashable_vi(new_state)] = comb[1]
        if action_name == 'collect':
            for comb in env_combinations:
                new_state, current_marine_locs = self.environment_changes(comb, treasure_dict, marine_dict, state)
                new_state[0]['pirate_ship']['capacity'] -= 1
                if current_loc in current_marine_locs:
                    new_state[0]['pirate_ship']['capacity'] = 2
                probabilities_dict[self.hashable_vi(new_state)] = comb[1]
        if action_name == 'deposit':
            for comb in env_combinations:
                new_state, _ = self.environment_changes(comb, treasure_dict, marine_dict, state)
                new_state[0]['pirate_ship']['capacity'] = 2
                probabilities_dict[self.hashable_vi(new_state)] = comb[1]
        if action_name == 'reset':
            initial_state = self.initial_vi_state
            probabilities_dict[self.hashable_vi(initial_state)] = 1
        if action_name == 'terminate':
            probabilities_dict = {}
        # the keys are hashable states and the values are the probabilities for the transition
        return probabilities_dict

    """The
    output is:
    pirate_ships = {'pirate_ship': {"location": (i, j), "capacity": capacity}}
    treasures = {'treasure_i': {"location": (4, 4)}
                 marine_ships = {'marine_i': {"index": 0}}"""

    def value_iteration(self, initial, map):
        # create all possible states
        # state = (location_map, pirate_ships, treasures, marine_ships)
        location_map = initial[0]
        pirate_ships = initial[1]
        treasures = initial[2]
        marine_ships = initial[3]

        # Get combinations of 3 marine ships and 4 treasures
        marine_indices = [list(range(len(ship['path']))) for ship in marine_ships.values()]
        treasure_locations = [treasure['possible_locations'] for treasure in treasures.values()]
        combinations_of_tlocs = list(product(*treasure_locations))
        combinations_of_paths = list(product(*marine_indices))
        V = [{} for _ in range(self.turns+1)]
        pi = [{} for _ in range(self.turns+1)]
        possible_states = []
        # rows
        for i in range(len(map)):
            # cols
            for j in range(len(map[0])):
                if map[i][j] == 'I':
                    continue
                for capacity in range(3):
                    for m_combination in combinations_of_paths:
                        for t_combination in combinations_of_tlocs:
                            # print(m_combination, t_combination, capacity)
                            current_state = self.create_vi_state(i, j, capacity, m_combination, t_combination)
                            possible_states.append(current_state)
                            hash_state = self.hashable_vi(current_state)
                            V[0][hash_state] = self.r(current_state)
        # print('V0')
        #print(V[0])
        for t in range(1, self.turns+1):
            #print('turn is', t)
            for s in possible_states:
                #print(s)
                hash_state = self.hashable_vi(s)
                V[t][hash_state] = float('-inf')
                for action in self.possible_actions(s):
                    value = 0
                    # transition gets a state and an action and returns a dictionary of possible states and their probabilities
                    if action[0] == 'deposit':
                        value += 4 * (2 - s[0]['pirate_ship']['capacity'])
                    if action[0] == 'reset':
                        value -= RESET_PENALTY
                    prob_dict = self.transition(s, action)
                    for s_prime in prob_dict.keys():
                        value += prob_dict[s_prime] * V[t-1][s_prime]
                    if value > V[t][hash_state]:
                        pi[t][hash_state] = action
                        V[t][hash_state] = value
                V[t][hash_state] += self.r(s)
        return V, pi

    def act(self, s):
        state = self.create_state_from_input(s)
        current_turn = s['turns to go']
        hash_state = self.hashable_vi(state)
        best_action = self.policy_table[current_turn][hash_state]
        action_name = best_action[0]
        best_action_extended = []
        for p in self.pirate_names:
            if action_name == 'sail':
                best_true_action = (best_action[0], p, best_action[2])
                best_action_extended.append(best_true_action)
            elif action_name == 'collect':
                treasure_index = int(best_action[2].split('_')[-1])
                best_true_action = (best_action[0], p, self.treasure_names[treasure_index])
                best_action_extended.append(best_true_action)
            elif action_name == 'deposit' or action_name == 'wait':
                best_true_action = (best_action[0], p)
                best_action_extended.append(best_true_action)
            elif action_name == 'reset' or action_name == 'terminate':
                return best_action
            else:
                best_action_extended.append(best_action)
        print('turns to go - ', current_turn, ', best action - ', best_action_extended)
        return tuple(best_action_extended)


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

        self.map, state = create_state(initial)
        self.marine_ships = initial['marine_ships']
        self.treasures = initial['treasures']
        # create state makes a full state like in hw1
        new_turns_num = 10
        self.map, state = create_state(initial)
        self.turns = initial["turns to go"]
        self.num_of_pirates = len(state[1].keys())
        self.pirate_names = list(state[1].keys())
        self.treasure_names = list(state[2].keys())
        self.marine_names = list(state[3].keys())
        first_ship = list(state[1].keys())[0]
        self.base_location = state[1][first_ship]['location']
        self.marine_locations = set(cell for marine_info in self.marine_ships.values() for cell in marine_info['path'])
        print(self.marine_locations)
        self.new_marines = self.create_new_marines(state[3])
        print(self.new_marines)
        self.best_treasure_names = self.find_best_treasures(self.treasures)
        self.best_treasures = {t: state[2][t] for t in self.best_treasure_names}
        print(self.best_treasures)
        new_state = (copy.deepcopy(state[0]), copy.deepcopy(state[1]), self.best_treasures, self.new_marines)
        self.rows = len(self.map)
        self.cols = len(self.map[0]) if self.rows > 0 else 0
        initial['marine_ships'] = self.new_marines
        initial['turns to go'] = new_turns_num
        initial['treasures'] = self.best_treasures
        self.initial_vi_state = self.create_initial_state(initial)
        self.value_iteration_table, self.policy_table = self.value_iteration(new_state, self.map)

    def create_new_marines(self, marine_ships):
        print('im hereee')
        new_marines = {}
        for i, m in enumerate(marine_ships.keys()):
            new_marines[m] = {"index": marine_ships[m]['index']}
        new_marine_ships = {}
        for marine, details in marine_ships.items():
            print(marine, details)
            for cell in details["path"]:
                new_marine_ships[f"{marine}_{cell}"] = {"index": details["index"], "path": [cell]}
        return new_marine_ships

    def r(self, state):
        pirate_ships = state[0]
        treasures = state[1]
        marine_ships = state[2]
        score = 0
        # print(pirate_ships)
        """
            ['S', 'S', 'I', 'S'],
            ['S', 'S', 'I', 'S'],
            ['B', 'S', 'S', 'S'],
            ['S', 'S', 'I', 'S']
            """
        ship_treasures_num = (2 - pirate_ships['pirate_ship']['capacity'])
        i, j = pirate_ships['pirate_ship']['location']
        # print(i, j)

        if (i, j) in self.marine_locations:
            score -= MARINE_COLLISION_PENALTY

        # score += ship_treasures_num

        return score

    def build_loc_map(self, state):
        location_map = [[{} for _ in range(self.cols)] for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                location_map[i][j] = {'pirate_ships': [],
                                      "marine_ships": [],
                                      'treasures': []}
        pirate_ships = state[0]

        """{'pirate_ship_1': {"location": (0, 0),
                                           "capacity": 2}"""
        treasures = state[1]
        """{'treasure_1': {"location": (4, 4),
                                     "possible_locations": ((4, 4),),
                                     "prob_change_location": 0.5}"""
        """{'marine_1': {"index": 0,
                                      "path": [(2, 3), (2, 3)]}}"""
        i, j = pirate_ships['pirate_ship']['location']
        location_map[i][j]['pirate_ships'].append('pirate_ship')

        for m_loc in self.marine_locations:
            i, j = m_loc
            location_map[i][j]['marine_ships'].append('marine')

        for treasure in treasures.keys():
            i, j = treasures[treasure]['location']
            location_map[i][j]['treasures'].append(treasure)

        return location_map

    def possible_actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        location_map = self.build_loc_map(state)
        map_rows = len(location_map)
        map_columns = len(location_map[0])
        pirate_ships = state[0]
        # sail
        pirate_ship = 'pirate_ship'
        possible_actions = []
        # ({'pirate_ship': {'location': (3, 1), 'capacity': 2}}, {'treasure_0': {'location': (0, 2)}, 'treasure_1': {'location': (3, 2)}}, {'marine_1_(1, 1)': {'index': 0}, 'marine_1_(2, 1)': {'index': 0}, 'marine_1_(2, 2)': {'index': 0}})
        i, j = pirate_ships[pirate_ship]['location']
        # left
        if j != 0:
            if self.map[i][j - 1] != 'I':
                possible_actions.append(("sail", pirate_ship, (i, j - 1)))
            else:  # island to the left
                if len(location_map[i][j - 1]['treasures']) > 0 and pirate_ships[pirate_ship]['capacity'] > 0:
                    for t in location_map[i][j - 1]['treasures']:
                        possible_actions.append(("collect", pirate_ship, t))
                        break
        # right
        if j != map_columns - 1:
            if self.map[i][j + 1] != 'I':
                possible_actions.append(("sail", pirate_ship, (i, j + 1)))
            else:
                # there is a treasure
                if (len(location_map[i][j + 1]['treasures']) > 0 and pirate_ships[pirate_ship]['capacity'] > 0):
                    for t in location_map[i][j + 1]['treasures']:
                        possible_actions.append(("collect", pirate_ship, t))
                        break
        # up
        if i != 0:
            if self.map[i - 1][j] != 'I':
                possible_actions.append(("sail", pirate_ship, (i - 1, j)))
            else:
                if (len(location_map[i - 1][j]['treasures']) > 0 and pirate_ships[pirate_ship]['capacity'] > 0):
                    for t in location_map[i - 1][j]['treasures']:
                        possible_actions.append(("collect", pirate_ship, t))
                        break

        # down
        if i != map_rows - 1:
            if self.map[i + 1][j] != 'I':
                possible_actions.append(("sail", pirate_ship, (i + 1, j)))

            else:
                if len(location_map[i + 1][j]['treasures']) > 0 and pirate_ships[pirate_ship]['capacity'] > 0:
                    for t in location_map[i + 1][j]['treasures']:
                        possible_actions.append(("collect", pirate_ship, t))
                        break

        # deposit_treasure
        if self.map[i][j] == 'B' and pirate_ships[pirate_ship]['capacity'] < 2:
            possible_actions.append(("deposit", pirate_ship))

        # wait
        possible_actions.append(("wait", pirate_ship))

        # reset
        possible_actions.append('reset')
        # terminate
        possible_actions.append('terminate')
        return possible_actions

    def create_vi_state(self, i, j, capacity, m_combination, t_combination):
        """rows = len(map)
        cols = len(map[0]) if rows > 0 else 0"""
        """" The output is:
        pirate_ships = {'pirate_ship': {"location": (i, j), "capacity": capacity}}
        treasures = {'treasure_i': {"location": (4, 4)}
        marine_ships = {'marine_i': {"index": 0}}
        """
        pirate_ships = {}
        pirate_ships['pirate_ship'] = {"location": (i, j), "capacity": capacity}
        treasures = {}
        for i, t in enumerate(t_combination):
            treasures[f'treasure_{i}'] = {"location": t}
        marine_ships = {}
        marine_names = list(self.new_marines.keys())
        for i, m_i in enumerate(m_combination): # 00000
            marine_ships[marine_names[i]] = {"index": m_i}

        return (pirate_ships, treasures, marine_ships)

    def hashable_vi(self, state):
        # print(state[0])
        hash_ships = tuple((ship, dict['location'], dict['capacity']) for ship, dict in state[0].items())
        # print(state[1].items())
        hash_treasures = tuple((k, dict['location']) for k, dict in state[1].items())
        hash_marines = tuple((k, dict['index']) for k, dict in state[2].items())
        return hash_ships + hash_treasures + hash_marines

    def unhash(self, hashable_state):
        ships = {}
        treasures = {}
        marines = {}

        for item in hashable_state[0]:  # Ships
            ship_name, location, capacity = item
            ships[ship_name] = {"location": location, "capacity": capacity}

        for item in hashable_state[1]:  # Treasures
            item_name, location = item
            treasures[item_name] = {"location": location}

        for item in hashable_state[2]:  # Marines
            item_name, index = item
            marines[item_name] = {"index": index}

        return [ships, treasures, marines]

    def environment_probs(self, treasure_dict, marine_dict):
        """
        in this function i want to make all combinations of treasure locations and marine locations out of all possible options
and then calculate the probability of each combination
each combination should combine all the treasures and all the marine ships

        treasure dict = {'treasure_i': {"(0,1)": 0.95
        "(1,0)": 0.05}}
        marine dict = {'marine_i': {"0": 0.5, "1": 0.5}}
        """
        # Get all keys (locations) and probabilities for treasures
        all_treasure_keys = []
        all_treasure_probs = []

        for treasure in treasure_dict.values():
            keys = list(treasure.keys())
            probs = list(treasure.values())
            all_treasure_keys.append(keys)
            all_treasure_probs.append(probs)

        # Get all keys (locations) and probabilities for marines
        all_marine_keys = []
        all_marine_probs = []

        for marine in marine_dict.values():
            keys = list(marine.keys())
            probs = list(marine.values())
            all_marine_keys.append(keys)
            all_marine_probs.append(probs)

        # Generate all combinations of treasures and marines locations
        combinations = list(product(*all_treasure_keys, *all_marine_keys))

        # Calculate probabilities for each combination
        probabilities = []
        for combination in combinations:
            prob = 1.0
            for i, loc in enumerate(combination):
                if i < len(all_treasure_keys):
                    prob *= treasure_dict[list(treasure_dict.keys())[i]][loc]
                else:
                    prob *= marine_dict[list(marine_dict.keys())[i - len(all_treasure_keys)]][loc]
            probabilities.append(prob)

        # Combine combinations with probabilities
        combinations_with_prob = list(zip(combinations, probabilities))

        return combinations_with_prob

    def environment_changes(self, comb, treasure_dict, marine_dict, state):
        # comb (('(0,1)', '(0,2)', '0'), 0.475)
        new_state = copy.deepcopy(state)
        cur_marine_locs = []
        marine_names = list(self.new_marines.keys())
        for i, t in enumerate(treasure_dict.keys()):
            new_state[1][t]['location'] = comb[0][i]
        cur_marine_locs = self.marine_locations
        # print('env changes - state', new_state)
        return new_state, cur_marine_locs

    def create_state_from_input(self, input):
        first_pirate_ship = self.pirate_names[0]
        loc = input['pirate_ships'][first_pirate_ship]['location']
        pirate_ships = {}
        pirate_ships['pirate_ship'] = {"location": loc,
                                       "capacity": input['pirate_ships'][first_pirate_ship]['capacity']}
        treasures = {}
        for i, t in enumerate(self.best_treasure_names): # [treasure2, treasure1]
            treasures[f'treasure_{i}'] = {"location": input['treasures'][t]['location']}
        """for i, t in enumerate(input['treasures'].items()):
            if t[0] in self.best_treasure_names:
                treasures[f'treasure_{i}'] = {"location": t[1]['location']}"""
        marine_ships = {}
        for marine in self.new_marines.keys():
            marine_ships[marine] = {"index": 0}
        # ('pirate_ship', (2, 0), 2), ('treasure_0', (0, 2)), ('marine_0', 0)
        return (pirate_ships, treasures, marine_ships)

    def create_initial_state(self, input):
        i, j = self.base_location
        pirate_ships = {}
        first_pirate_ship = self.pirate_names[0]
        pirate_ships['pirate_ship'] = {"location": (i, j),
                                       "capacity": input['pirate_ships'][first_pirate_ship]['capacity']}
        treasures = {}
        for i, t in enumerate(input['treasures'].values()):
            treasures[f'treasure_{i}'] = {"location": t['location']}
        marine_ships = {}

        for marine, marine_info in input['marine_ships'].items():
            marine_ships[marine] = {"index": 0}
        return (pirate_ships, treasures, marine_ships)

    def transition(self, state, action):
        """ state is:
        pirate_ships = {'pirate_ship': {"location": (i, j), "capacity": capacity}}
        treasures = {'treasure_i': {"location": (4, 4)}}
        marine_ships = {'marine_1': {"index": 0}
                        'marine_2': {"index": 1}
        """

        pirate_ships = state[0]
        treasures = state[1]
        marine_ships = state[2]
        probabilities_dict = {}
        action_name = action[0]
        treasure_dict = {}
        marine_dict = {}

        for t_i, treasure in enumerate(treasures.keys()):
            treasure_dict[treasure] = {}
            original_name = self.best_treasure_names[t_i]
            transition_prob = self.treasures[original_name]['prob_change_location']
            for location in self.treasures[original_name]['possible_locations']:
                if location == treasures[treasure]['location']:
                    treasure_dict[treasure][location] = (1 - transition_prob) + transition_prob / len(
                        self.treasures[original_name]['possible_locations'])
                else:
                    treasure_dict[treasure][location] = transition_prob / len(
                        self.treasures[original_name]['possible_locations'])

        current_marine_locs = []
        for m_i, marine in enumerate(self.new_marines.keys()):
            #original_name = self.marine_names[m_i]
            path_len = len(self.new_marines[marine]['path'])
            if path_len == 1:
                marine_dict[marine] = {0: 1}
            else:
                if marine_ships[marine]['index'] == 0:
                    marine_dict[marine] = {0: 0.5, 1: 0.5}
                elif marine_ships[marine]['index'] == path_len - 1:
                    marine_dict[marine] = {path_len - 1: 0.5,
                                           path_len - 2: 0.5}
                else:
                    marine_dict[marine] = {marine_ships[marine]['index'] - 1: 0.3333,
                                           marine_ships[marine]['index']: 0.3333,
                                           marine_ships[marine]['index'] + 1: 0.3333}
        """Combinations with Probabilities:
        0 , 1
        (('(0,1)', '(0,2)', '0'), 0.475)
        (('(0,1)', '(0,2)', '1'), 0.475)"""
        current_loc = pirate_ships['pirate_ship']['location']
        env_combinations = self.environment_probs(treasure_dict, marine_dict)
        if action_name == 'wait':
            for comb in env_combinations:
                new_state, current_marine_locs = self.environment_changes(comb, treasure_dict, marine_dict, state)
                if current_loc in current_marine_locs:
                    new_state[0]['pirate_ship']['capacity'] = 2
                probabilities_dict[self.hashable_vi(new_state)] = comb[1]
        if action_name == 'sail':
            new_loc = action[2]
            for comb in env_combinations:
                new_state, current_marine_locs = self.environment_changes(comb, treasure_dict, marine_dict, state)
                new_state[0]['pirate_ship']['location'] = new_loc
                if new_loc in current_marine_locs:
                    new_state[0]['pirate_ship']['capacity'] = 2
                probabilities_dict[self.hashable_vi(new_state)] = comb[1]
        if action_name == 'collect':
            for comb in env_combinations:
                new_state, current_marine_locs = self.environment_changes(comb, treasure_dict, marine_dict, state)
                new_state[0]['pirate_ship']['capacity'] -= 1
                if current_loc in current_marine_locs:
                    new_state[0]['pirate_ship']['capacity'] = 2
                probabilities_dict[self.hashable_vi(new_state)] = comb[1]
        if action_name == 'deposit':
            for comb in env_combinations:
                new_state, _ = self.environment_changes(comb, treasure_dict, marine_dict, state)
                new_state[0]['pirate_ship']['capacity'] = 2
                probabilities_dict[self.hashable_vi(new_state)] = comb[1]
        if action_name == 'reset':
            initial_state = self.initial_vi_state
            probabilities_dict[self.hashable_vi(initial_state)] = 1
        if action_name == 'terminate':
            probabilities_dict = {}
        # the keys are hashable states and the values are the probabilities for the transition
        return probabilities_dict

    """The
    output is:
    pirate_ships = {'pirate_ship': {"location": (i, j), "capacity": capacity}}
    treasures = {'treasure_i': {"location": (4, 4)}
                 marine_ships = {'marine_i': {"index": 0}}"""

    def value_iteration(self, initial, map):
        # create all possible states
        # state = (location_map, pirate_ships, treasures, marine_ships)
        location_map = initial[0]
        pirate_ships = initial[1]
        treasures = initial[2]
        marine_ships = initial[3]

        # Get combinations of 3 marine ships and 4 treasures
        marine_indices = [list(range(len(ship['path']))) for ship in marine_ships.values()]
        treasure_locations = [treasure['possible_locations'] for treasure in treasures.values()]
        combinations_of_tlocs = list(product(*treasure_locations))
        combinations_of_paths = list(product(*marine_indices))
        V = [{} for _ in range(self.turns + 1)]
        pi = [{} for _ in range(self.turns + 1)]
        possible_states = []
        # rows
        for i in range(len(map)):
            # cols
            for j in range(len(map[0])):
                if map[i][j] == 'I':
                    continue
                for capacity in range(3):
                    for m_combination in combinations_of_paths:
                        for t_combination in combinations_of_tlocs:
                            # print(m_combination, t_combination, capacity)
                            current_state = self.create_vi_state(i, j, capacity, m_combination, t_combination)
                            possible_states.append(current_state)
                            hash_state = self.hashable_vi(current_state)
                            V[0][hash_state] = self.r(current_state)
        #print(possible_states)
        #print(V[0].keys())
        #print(V[0])
        # (('pirate_ship', (0, 1), 0), ('treasure_0', (1, 2)), ('treasure_1', (0, 2)), ('marine_1_(1, 1)', 0), ('marine_1_(2, 1)', 0), ('marine_1_(2, 2)', 0))
        for t in range(1, self.turns + 1):
            print('turn is', t)
            for s in possible_states:
                #print(s)
                hash_state = self.hashable_vi(s)
                V[t][hash_state] = float('-inf')
                for action in self.possible_actions(s):
                    value = 0
                    # transition gets a state and an action and returns a dictionary of possible states and their probabilities
                    if action[0] == 'deposit':
                        value += 4 * (2 - s[0]['pirate_ship']['capacity'])
                    if action[0] == 'reset':
                        value -= RESET_PENALTY
                    prob_dict = self.transition(s, action)
                    for s_prime in prob_dict.keys():
                        value += prob_dict[s_prime] * V[t - 1][s_prime]
                    if value > V[t][hash_state]:
                        pi[t][hash_state] = action
                        V[t][hash_state] = value
                V[t][hash_state] += self.r(s)
            #print(t)
            #print(V[t])
        return V, pi

    def act(self, s):
        #print(s)
        state = self.create_state_from_input(s)
        #print(state)
        current_turn = s['turns to go']
        hash_state = self.hashable_vi(state)
        #print(self.policy_table[current_turn].keys())
        best_action = self.policy_table[current_turn][hash_state]
        #print('best action in act', best_action)
        action_name = best_action[0]
        best_action_extended = []
        for p in self.pirate_names:
            #print(best_action)
            if action_name == 'sail':
                best_true_action = (best_action[0], p, best_action[2])
                best_action_extended.append(best_true_action)
            elif action_name == 'collect':
                treasure_index = int(best_action[2].split('_')[-1])
                best_true_action = (best_action[0], p, self.best_treasure_names[treasure_index])
                best_action_extended.append(best_true_action)
            elif action_name == 'deposit' or action_name == 'wait':
                best_true_action = (best_action[0], p)
                best_action_extended.append(best_true_action)
            elif action_name == 'reset' or action_name == 'terminate':
                return best_action
            else:
                best_action_extended.append(best_action)
        print('turns to go - ', current_turn, ', best action - ', best_action_extended)
        return tuple(best_action_extended)

    def calculate_treasure_probability(self, treasures, t):
        location_probabilities = {}

        for treasure in treasures.values():
            for location in treasure["possible_locations"]:
                if location not in location_probabilities:
                    location_probabilities[location] = 0
                location_probabilities[location] += 1 - treasure["prob_change_location"]

        total_treasures = len(treasures)
        for location, probability in location_probabilities.items():
            location_probabilities[location] = min(1, probability * total_treasures)

        return location_probabilities


    def are_seas_nearby(self, map, t_info):
        i, j = t_info['location']

        for loc in t_info['possible_locations']:
            i, j = loc
            seas = []
            if i != 0:
                if map[i-1][j] == 'S' and (i-1, j) not in self.marine_locations:
                    seas.append((i-1, j))
            if i != len(map) - 1:
                if map[i+1][j] == 'S' and (i+1, j) not in self.marine_locations:
                    seas.append((i+1, j))
            if j != 0:
                if map[i][j-1] == 'S' and (i, j-1) not in self.marine_locations:
                    seas.append((i, j-1))
            if j != len(map[0]) - 1:
                if map[i][j+1] == 'S' and (i, j+1) not in self.marine_locations:
                    seas.append((i, j+1))
            # there is a possible location you cant access
            if not bool(seas):
                return None
        return True

    def dist_from_base(self, loc):
        return abs(loc[0] - self.base_location[0]) + abs(loc[1] - self.base_location[1])

    def most_stable(self, treasures):
        sorted_treasures = sorted(treasures.items(), key=lambda x: x[1]['prob_change_location'])
        most_stable_treasures = [t for t, t_info in sorted_treasures[:2]]
        return most_stable_treasures

    def least_possible_locs(self, treasures):
        sorted_treasures = sorted(treasures.items(), key=lambda x: len(x[1]['possible_locations']))
        least_possible_locs = [t for t, t_info in sorted_treasures if
                               len(t_info['possible_locations']) == len(sorted_treasures[0][1]['possible_locations'])]
        return least_possible_locs[:2]

    def closest_treasures(self, treasures):
        sorted_treasures = sorted(treasures.items(), key=lambda x: self.dist_from_base(x[1]['location']))
        min_dist = sorted_treasures[0][1]['location']
        closest_treasures = [t for t, t_info in sorted_treasures if self.dist_from_base(t_info['location']) == min_dist]
        return closest_treasures[:2]

    def find_best_treasures(self, treasures):
        treasures_set = set(treasures.keys())
        new_treasures_set = set(treasures.keys())
        # removing the treasures that have non-accessible locations
        for t, t_info in treasures.items():
            if not self.are_seas_nearby(self.map, t_info):
                new_treasures_set -= {t}

        treasure_counts = {t: 0 for t in treasures.keys()}
        # Get the sets of treasures from each function
        most_stable_treasures = self.most_stable(treasures)
        least_possible_locs = self.least_possible_locs(treasures)
        closest_treasures = self.closest_treasures(treasures)
        # Count occurrences of each treasure in the sets
        for treasure_set in [most_stable_treasures, least_possible_locs, closest_treasures, new_treasures_set]:
            for treasure in treasure_set:
                treasure_counts[treasure] += 1

        top_2_best_treasures = sorted(treasure_counts.keys(), key=lambda x: treasure_counts[x], reverse=True)[:2]
        print('top 2 best treasures', top_2_best_treasures)
        return top_2_best_treasures



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


class InfinitePirateAgent:
    def __init__(self, initial, gamma):
        self.initial = initial
        self.gamma = gamma

    def act(self, state):
        raise NotImplemented

    def value(self, state):
        raise NotImplemented
