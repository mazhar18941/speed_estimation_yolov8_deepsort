import numpy as np

dark_colors = [
            (235, 219, 11),
            (243, 243, 243),
            (183, 223, 0),
            (221, 111, 255),
            (0, 237, 204),
            (68, 243, 0),
            (255, 255, 0),
            (179, 255, 1),
            (11, 255, 162),
        ]
light_colors = [
            (255, 42, 4),
            (79, 68, 255),
            (255, 0, 189),
            (255, 180, 0),
            (186, 0, 221),
            (0, 192, 38),
            (255, 36, 125),
            (104, 0, 123),
            (108, 27, 255),
            (47, 109, 252),
            (104, 31, 17),
        ]

'''
SOURCE = np.array([[395, 370], [868, 370], [1254, 612], [0, 612]])

TARGET_WIDTH = 15
TARGET_HEIGHT = 20

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH, 0],
        [TARGET_WIDTH, TARGET_HEIGHT],
        [0, TARGET_HEIGHT],
    ]
)
'''

SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])

TARGET_WIDTH = 20
TARGET_HEIGHT = 120

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)

