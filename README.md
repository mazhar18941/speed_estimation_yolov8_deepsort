# speed_estimation_yolov8_deepsort
This project contains code for speed estimation of vehicles using yolov8 object detector and deepsort object tracker.


https://github.com/user-attachments/assets/759390dc-eec7-41f0-b458-d69562de65dc




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
https://github.com/mazhar18941/speed_estimation_yolov8_deepsort.git
```
Then, download the CNN checkpoint file from
[here](https://drive.google.com/open?id=18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp).



Appearance descriptor is replaced with a custom deep convolutional
neural network.

## Running the estimator

```
python main.py --descriptor "path to descriptor" --object-detector "path to yolov8" --video "path to video"
```
Only "car","truck","bus" classes are being tracked in this code. In order to track other class like "bike" change code lne no 65 in main.py to following:

if result.names[box.cls[0].item()] == ['car','truck','bus']:

Check `python main.py -h` for an overview of available options.

## Configuration

SOURCE and TARGET points are taken for specific video used in this project. In order to use another video edit SOURCE and TARGET points in config.py file.

## Reference

https://github.com/nwojke/deep_sort
https://github.com/Qidian213/deep_sort_yolov3
https://github.com/roboflow/supervision/tree/develop/examples/speed_estimation
