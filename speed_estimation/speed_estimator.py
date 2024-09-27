import cv2
import numpy as np
#from keras.layers.preprocessing.image_preprocessing import transform
from scipy.spatial import distance
from . import transform
import config

SOURCE = config.SOURCE
TARGET = config.TARGET

view_transformer = transform.ViewTransformer(source=SOURCE, target=TARGET)

class Estimate:
    def __init__(self, frame_limit, video_fps):
        self.default_bbox_prev = [0,0,0,0]
        self.default_frame_n_prev = 0
        self.default_time_s = 0
        self.default_speed = 0
        self.tid_list = []
        self.bbox_list = []
        self.frame_n_list = []
        self.time_list = []
        self.speed_list = []
        self.frame_limit = frame_limit
        self.video_fps = video_fps

    def calculate_speed(self, bbox, bbox_prev, frame_n, frame_n_prev, time_s, time_e):
        center = self.find_centroid(bbox)
        center = view_transformer.transform_points(points=center).astype(float)
        center_prev = self.find_centroid(bbox_prev)
        center_prev = view_transformer.transform_points(points=center_prev).astype(float)
        #dist_dif = distance.euclidean(center, center_prev)
        dist_dif = abs(center.item(1)-center_prev.item(1))
        frame_dif = abs(frame_n-frame_n_prev)
        time_dif_vid = self.frame_limit/self.video_fps
        #speed = round(dist_dif / time_dif, 1)
        speed = round(dist_dif / time_dif_vid, 2) * 3.6
        return speed




    def estimator(self, tid, bbox, frame_n, time):
        if tid not in self.tid_list:
            self.tid_list.append(tid)
            self.bbox_list.append(bbox)
            self.frame_n_list.append(frame_n)
            self.time_list.append(time)
            self.speed_list.append(self.default_speed)
            speed = self.calculate_speed(bbox, self.default_bbox_prev, frame_n, self.default_frame_n_prev, self.default_time_s, time)
            self.speed_list[self.tid_list.index(tid)] = speed

        else:
            bbox_prev = self.bbox_list[self.tid_list.index(tid)]
            frame_n_prev = self.frame_n_list[self.tid_list.index(tid)]
            time_s = self.time_list[self.tid_list.index(tid)]
            speed = self.calculate_speed(bbox, bbox_prev, frame_n, frame_n_prev, time_s, time)

            self.bbox_list[self.tid_list.index(tid)] = bbox
            self.frame_n_list[self.tid_list.index(tid)] = frame_n
            self.time_list[self.tid_list.index(tid)] = time
            self.speed_list[self.tid_list.index(tid)] = speed
        return self.tid_list, self.speed_list



    def find_centroid(self, box):
        center = np.array([((box[2]-box[0])/2)+box[0], ((box[3] - box[1]) / 2) + box[1]])
        return center