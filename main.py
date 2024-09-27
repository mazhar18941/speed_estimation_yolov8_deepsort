from tools.generate_detections import create_box_encoder
from deep_sort.detection import Detection
from application_util import preprocessing
from deep_sort.tracker import Tracker
from deep_sort import nn_matching
from application_util import plotting

import numpy as np
import cv2
import argparse
import time
from ultralytics import YOLO

from speed_estimation import speed_estimator
import config

def main(min_confidence, nms_max_overlap, max_cosine_distance,
         nn_budget, descriptor, object_detector, video):

    #COLORS = np.random.randint(0, 255, size=(200, 3),
     #                          dtype="uint8")
    color_cl = plotting.draw()
    COLORS = color_cl.get_color_space()

    encoder = create_box_encoder(descriptor, batch_size=1)
    yolo_model = YOLO(object_detector)

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    counter = []

    cap = cv2.VideoCapture(video)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    width = int(cap.get(3))  # float `width`
    height = int(cap.get(4))  # float `height`
    video_fps = int(cap.get(5))
    out = cv2.VideoWriter('deepsort.mp4', fourcc, video_fps, (width, height))
    # speed estimation
    n_frames = 0
    tid_list = []
    speed_list = []
    frame_limit = 6       # after frame_limit speed will be updated
    se = speed_estimator.Estimate(frame_limit, video_fps)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video file")

    # Read until video is completed
    s = 0
    while (cap.isOpened()):
        ret, image = cap.read()

        if ret == True:

            # any vehicle under this line will be tracked
            cv2.line(image, (config.SOURCE[0,0], config.SOURCE[0,1]), (config.SOURCE[1,0], config.SOURCE[1,1]), (0, 255, 0), 8)
            results = yolo_model.predict(image)
            result = results[0]
            box_coord_list = []
            detection_list = []

            for box in result.boxes:
                if result.names[box.cls[0].item()] in ['car','truck','bus']:
                    box_coord = box.xyxy[0].tolist()
                    # converting boxes into [x,y,w,h] (x,y)-top left corner
                    box_coord = [box_coord[0], box_coord[1], box_coord[2] - box_coord[0], box_coord[3] - box_coord[1]]
                    box_coord = [round(x) for x in box_coord]
                    box_conf = round(box.conf[0].item(), 2)
                    # box_class = result.names[box.cls[0].item()]
                    # any vehicle under this line will be tracked
                    if box_coord[1] > config.SOURCE[0,1]:
                        feature = encoder(image, np.array([box_coord]))
                        feature = np.reshape(feature, (128,))

                        detection_list.append(Detection(box_coord, box_conf, feature))

            detection_list = [d for d in detection_list if d.confidence >= min_confidence]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detection_list])
            scores = np.array([d.confidence for d in detection_list])
            indices = preprocessing.non_max_suppression(
                boxes, nms_max_overlap, scores)
            detections = [detection_list[i] for i in indices]

            # Update tracker.
            tracker.predict()
            tracker.update(detections)

            i = int(0)
            indexIDs = []
            bbox_list = []
            color_list = []

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bbox_ = track.to_tlbr()
                bbox = [int(b) for b in bbox_]

                indexIDs.append(int(track.track_id))
                bbox_list.append(bbox)
                counter.append(int(track.track_id))
                color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
                color_list.append(color)

                # speed estimation
                if n_frames%frame_limit==0 or n_frames==0:
                    tid_list, speed_list = se.estimator(track.track_id, bbox_, n_frames, time.time())
                    #cv2.putText(image, str(round(speed_list[tid_list.index(track.track_id)],2)), (bbox[0]+20, bbox[1]), 0, 5e-3 * 150, (color), 2)


                #cv2.circle(image, (bbox_center[0],bbox_center[1]), 7, (color), -1)
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (color), thickness=5)
                #cv2.putText(image, str(track.track_id), (bbox[0], bbox[1]), 0, 5e-3 * 150, (color), 2)
                # cv2.imwrite(savefilepath+str(track.track_id)+".jpg", image[bbox[1]:bbox[3], bbox[0]:bbox[2]])
                i += 1

            # display speed
            for t, s in zip(tid_list, speed_list):
                if t in indexIDs and s<250: # if speed s less than 250km/h
                    b = bbox_list[indexIDs.index(t)]
                    c = color_list[indexIDs.index(t)]
                    color_cl.draw_text(image, str(t)+'#'+str(round(s, 2))+'km/h',
                                       (int(b[0]), int(b[1])), c)

            n_frames += 1

            cv2.namedWindow("YOLO8_Deep_SORT", cv2.WINDOW_NORMAL)
            cv2.resizeWindow('YOLO8_Deep_SORT', 1024, 768)
            cv2.imshow("YOLO8_Deep_SORT", image)
            out.write(image)
            # Press Q on keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Break the loop
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")

    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--descriptor", help="Path to Descriptor Model "
                            "", type=str, default="resources/networks/mars-small128.pb")
    parser.add_argument(
        "--object_detector", help="Path to  YOLOv8 Model "
                                        "", type=str, default="resources/yolo/yolov8x.pt")
    parser.add_argument(
        "--video", help="Path to Video  "
                                        "", type=str, default='resources/incoming_traffic.mp4')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.min_confidence, args.nms_max_overlap, args.max_cosine_distance,
         args.nn_budget, args.descriptor, args.object_detector, args.video)
