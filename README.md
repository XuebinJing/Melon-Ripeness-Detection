# <div align="center">Melon Ripeness Detection</div>



The weight of the MRD-YOLO model is submitted to `weights/MRD.pt`.  You can follow the steps in **Evaluation** to validate the model's parameters as well as the floating-point operations, and to test the MRD-YOLO model on a dataset containing 300 images.



Please make sure you have the ultralytics package installed.

```py
# Install the ultralytics package from PyPI
pip install ultralytics
```

For full documentation on installation, training, validation, prediction, and deployment, please refer to the [YOLOv8 GitHub repository](https://github.com/ultralytics/ultralytics) and [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/).



### Dataset

The partially data used as examples are stored in  `dataset/images/test` directory,  and the corresponding annotations are stored in `dataset/labels/test`.  The complete original dataset is available on request.

**Dataset Structure:**

```lua
dataset
|-- images: Stores images used for training and testing.
|-- labels: Stores the annotations of the dataset in the YOLO format.
|-- melon.yaml: YAML files required for MRD-YOLO training. Please remember to replace the path with your own and note the space between the colon and the path.

```

### Evaluation

👉 Refer to the following code (Test.py) to test the MRD-YOLO model.

```python
from ultralytics import YOLO
# Load a model
model = YOLO('weights/MRD.pt')  # load a custom model
 
# Validate the model
metrics = model.val(data='dataset/melon.yaml',split='test',batch=1)  #Please remember to replace the path in melon.yaml with your own and note the space between the colon and the path.

metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category

```

Or you can just run the command:

```python
python Test.py
```



### Log

👉 The experimental results of the comparison experiments as well as the ablation experiments mentioned in the paper are documented in the `log/` directory.
