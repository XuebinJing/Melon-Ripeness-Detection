from ultralytics import YOLO
# Load a model
model = YOLO('weights/MRD.pt')  # load a custom model
 
# Validate the model
metrics = model.val(data='dataset/melon.yaml',split='test',batch=1)  # Please remember to replace the path in melon.yaml with your own and note the space between the colon and the path
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category
