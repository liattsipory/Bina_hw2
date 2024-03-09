additional_inputs = [
    # an infinite game with 5 X 5 map and one pirate ship, 1 treasure and one marine ship
    {
        "optimal": True,
        "infinite": False,
        "gamma": 0.9,
        "map": [['B', 'S', 'S', 'S', 'I'],
                ['I', 'S', 'I', 'S', 'I'],
                ['S', 'S', 'I', 'S', 'S'],
                ['S', 'I', 'S', 'S', 'S'],
                ['S', 'S', 'S', 'S', 'I']],

        "pirate_ships": {'pirate_ship_1': {"location": (0, 0),
                                           "capacity": 2}
                         },
        "treasures": {'treasure_1': {"location": (4, 4),
                                     "possible_locations": ((4, 4),),
                                     "prob_change_location": 0.5}},
        "marine_ships": {'marine_1': {"index": 1,
                                      "path": [(2, 3), (2, 3)]}},
        'turns to go': 100
    },
    {
        "optimal": True,
        "infinite": False,
        "map": [['S', 'S', 'I', 'S'],
                ['S', 'S', 'I', 'S'],
                ['B', 'S', 'S', 'S'],
                ['S', 'S', 'I', 'S']],
        "pirate_ships": {'pirate_ship_1': {"location": (2, 0),
                                           "capacity": 2}
                         },
        "treasures": {'treasure_1': {"location": (0, 2),
                                     "possible_locations": ((0, 2), (1, 2), (3, 2)),
                                     "prob_change_location": 0.1},
                      'treasure_2': {"location": (3, 2),
                                     "possible_locations": ((0, 2), (3, 2)),
                                     "prob_change_location": 0.1}
                      },
        "marine_ships": {'marine_1': {"index": 0,
                                      "path": [(1, 1), (2, 1), (2, 2), (2, 1)]}},
        "turns to go": 100
    },
    # a finite large game - not optimal
    {
        "optimal": False,
        "infinite": False,
        "map": [['S', 'S', 'I', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
                ['S', 'S', 'I', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
                ['B', 'S', 'I', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
                ['S', 'S', 'I', 'S', 'S', 'S', 'S', 'S', 'I', 'S'],
                ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
                ['S', 'S', 'S', 'S', 'I', 'S', 'S', 'S', 'S', 'I']],

        "pirate_ships": {'pirate_ship_1': {"location": (2, 0),
                                           "capacity": 2},
                         'pirate_bob': {"location": (2, 0),
                                        "capacity": 2},
                         'bob the pirate': {"location": (2, 0),
                                            "capacity": 2}
                         },
        "treasures": {'treasure_1': {"location": (0, 2),
                                     "possible_locations": ((0, 2), (1, 2), (3, 2)),
                                     "prob_change_location": 0.2},
                      'treasure_2': {"location": (2, 2),
                                     "possible_locations": ((0, 2), (2, 2), (3, 2)),
                                     "prob_change_location": 0.1},
                      'treasure_3': {"location": (3, 8),
                                     "possible_locations": ((3, 8), (3, 2), (5, 4)),
                                     "prob_change_location": 0.3},
                      'magical treasure': {"location": (5, 9),
                                           "possible_locations": ((5, 9), (5, 4)),
                                           "prob_change_location": 0.4}

                      },
        "marine_ships": {'marine_1': {"index": 0,
                                      "path": [(1, 1), (2, 1)]},
                         "larry the marine": {"index": 0,
                                              "path": [(5, 6), (4, 6), (4, 7)]},
                         },
        "turns to go": 100
    },

{
        "optimal": False,
        "infinite": False,
        "map": [['S', 'S', 'I', 'I', 'I', 'S', 'S', 'S', 'S', 'B'],
                ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
                ['S', 'S', 'S', 'S', 'S', 'S', 'I', 'I', 'I', 'S'],
                ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'I', 'S'],
                ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
                ['S', 'S', 'S', 'S', 'I', 'I', 'I', 'S', 'S', 'S']],

        "pirate_ships": {'pirate_ship_1': {"location": (0, 9),
                                           "capacity": 2},
                         'pirate_bob': {"location": (0, 9),
                                        "capacity": 2},
                         'bob the pirate': {"location": (0, 9),
                                            "capacity": 2}
                         },
        "treasures": {'magical treasure': {"location": (3, 8),
                                           "possible_locations": ((3, 8), (2, 8), (2, 7), (2, 6)),
                                           "prob_change_location": 0.4},

                        'treasure_hii': {"location": (0, 2),
                                     "possible_locations": ((0, 2), (0, 3), (0, 4)),
                                     "prob_change_location": 0.2},
                      'treasure_yeees': {"location": (5, 6),
                                     "possible_locations": ((5, 6), (5, 5), (5, 4)),
                                     "prob_change_location": 0.1}

                      },
        "marine_ships": {'marine_1': {"index": 0,
                                      "path": [(4, 6)]},
                         "larry the marine": {"index": 0,
                                              "path": [(1, 6), (1, 7), (1, 8), (1, 9)]},
                         },
        "turns to go": 100
    },
{
        "optimal": False,
        "infinite": False,
        "map": [['S', 'S', 'I', 'I', 'I', 'S', 'S', 'S', 'S', 'S'],
                ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
                ['S', 'S', 'S', 'S', 'B', 'S', 'I', 'I', 'I', 'S'],
                ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'I', 'S'],
                ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
                ['S', 'S', 'S', 'S', 'I', 'I', 'I', 'S', 'S', 'S']],

        "pirate_ships": {'pirate_ship_1': {"location": (2, 4),
                                           "capacity": 2},
                         'pirate_bob': {"location": (2, 4),
                                        "capacity": 2},
                         'bob the pirate': {"location": (2, 4),
                                            "capacity": 2}
                         },
        "treasures": {'treasure_2': {"location": (0, 2),
                                     "possible_locations": ((0, 2), (0, 3), (0, 4)),
                                     "prob_change_location": 0.2},
                      'treasure_3': {"location": (5, 6),
                                     "possible_locations": ((5, 6), (5, 5), (5, 4)),
                                     "prob_change_location": 0.1},
                      'treasure_1': {"location": (3, 8),
                                           "possible_locations": ((3, 8), (2, 8), (2, 7), (2, 6)),
                                           "prob_change_location": 0.4}
                      },
        "marine_ships": {'marine_1': {"index": 0,
                                      "path": [(0, 1), (1, 1), (1, 2), (1, 3), (1, 4), (1, 6), (1, 7), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7)]},
                         "larry the marine": {"index": 0,
                                              "path": [(2, 5), (3, 5)]},
                         },
        "turns to go": 100
    },
{
        "optimal": False,
        "infinite": False,
        "map": [['S', 'S', 'I', 'I', 'I', 'S', 'S', 'S', 'S', 'S'],
                ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
                ['S', 'S', 'S', 'S', 'B', 'S', 'I', 'I', 'I', 'S'],
                ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'I', 'S'],
                ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
                ['S', 'S', 'S', 'S', 'I', 'I', 'I', 'S', 'S', 'S']],

        "pirate_ships": {'pirate_ship_1': {"location": (2, 4),
                                           "capacity": 2},
                         'pirate_bob': {"location": (2, 4),
                                        "capacity": 2},
                         'bob the pirate': {"location": (2, 4),
                                            "capacity": 2}
                         },
        "treasures": {'treasure_liat': {"location": (0, 2),
                                     "possible_locations": ((0, 2), (0, 3), (0, 4)),
                                     "prob_change_location": 0.4},
                      'treasure_anna': {"location": (5, 6),
                                     "possible_locations": ((5, 6), (5, 5), (5, 4)),
                                     "prob_change_location": 0.3},
                      'magical treasure': {"location": (3, 8),
                                           "possible_locations": ((3, 8), (2, 8), (2, 7), (2, 6)),
                                           "prob_change_location": 0.4}
                      },
        "marine_ships": {'marine_1': {"index": 0,
                                      "path": [(1, 1), (1, 2), (1, 3), (1, 4), (1, 6), (1, 7), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7)]},
                         "larry the marine": {"index": 0,
                                              "path": [(2, 5), (3, 5)]},
                         },
        "turns to go": 100
    }

    # a finite game with 4 X 4 map and one pirate ship, 2 treasures and one marine ship
]
