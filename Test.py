from ultralytics import YOLO
# Load a model
model = YOLO('weights/MRD.pt')  # load a custom model
 
# Validate the model
metrics = model.val(data='dataset/melon.yaml',split='test',batch=1)  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category
