import cv2
import numpy as np
import config

class draw():
    def __init__(self):
        self.dark_colors = config.dark_colors
        self.light_colors = config.light_colors

    def get_color_space(self):
        return np.array(list(self.dark_colors) + list(self.light_colors))


    def draw_text(self, image, text, location, bb_color):
        if bb_color in np.array(self.dark_colors):
            text_color = (0, 0, 0)
        else:

            text_color = (255, 255, 255)
        font_scale = 5
        font = cv2.FONT_HERSHEY_PLAIN
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=10)[0]
        text_offset_x = location[0]
        text_offset_y = location[1]
        box_coords = ((text_offset_x+2, text_offset_y+2), (text_offset_x + text_width + 4, text_offset_y - text_height - 2))
        cv2.rectangle(image, box_coords[0], box_coords[1], bb_color, cv2.FILLED)
        cv2.putText(image, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(text_color), thickness=10)
