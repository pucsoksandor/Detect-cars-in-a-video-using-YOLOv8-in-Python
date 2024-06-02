# Import the YOLO class from the ultralytics library
from ultralytics import YOLO

# Initialize the YOLO object and provide the path to the trained model weights
model = YOLO("yolov8s.pt")

# Perform training of the model with the specified data and for one epoch
# The data is stored in a YAML file containing paths to training images and labels
results = model.train(data="pathsForTraining.yaml", epochs=1)

