from roboflow import Roboflow
rf = Roboflow(api_key="jDPp37Y4h5lf4I2ipCjv")
project = rf.workspace("safewalkbd").project("safewalkbd-l8jbn")
version = project.version(9)
dataset = version.download("yolov8")
                