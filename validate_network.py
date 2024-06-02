# Import the YOLO class from the ultralytics library
from ultralytics import YOLO

# Initialize the YOLO object and provide the path to the trained model weights
model = YOLO(r"C:\Users\pucso\Desktop\Vehicle_detecting\runs\detect\train2\weights\best.pt")

# Perform training of the model with the specified data and for one epoch
# The data is stored in a YAML file containing paths to training images and labels
results = model.val(data="evaluateNetwork.yaml", epochs=1)
