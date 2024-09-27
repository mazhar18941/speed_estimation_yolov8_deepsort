# speed_estimation_yolov8_deepsort
This project contains code for speed calculation of vehicles using yolov8 object detector and deepsort object tracker.

## Dependencies

The code is compatible with Python 2.7 and 3. The following dependencies are
needed to run the tracker:

```
pip install -r requirements.txt
```
Additionally, feature generation requires TensorFlow (>= 1.0).

## Installation

First, clone the repository:
```
https://github.com/mazhar18941/deepSort-Yolov8.git
```
Then, download the CNN checkpoint file from
[here](https://drive.google.com/open?id=18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp).



We have replaced the appearance descriptor with a custom deep convolutional
neural network (see below).

