# <div align="center">LRD-YOLO</div>

LRD-YOLO is a high-throughput YOLOv8-based method designed for leaf rolling detection in maize. It incorporates two significant enhancements: the Convolutional Block Attention Module (CBAM) and Deformable ConvNets v2 (DCNv2). This method demonstrates exceptional performance in occluded scenes and complex environments, facilitating the effective recognition of leaf rolling as a plant phenotype.



For full documentation on installation, training, validation, prediction, and deployment, please refer to the [YOLOv8 GitHub repository](https://github.com/ultralytics/ultralytics).



### Dataset

The partially annotated data used as examples are stored in the  `dataset/` directory, the complete original dataset is available on request.

**Dataset Structure:**

```lua
dataset
|-- images: Stores images used for training and testing.
|-- labels: Stores the annotations of the dataset in the YOLO format.
|-- leaf.yaml: YAML files required for LRD-YOLO training.

```

 

### Training

👉  **Models Directory (`ultralytics/cfg/models/v8/`):** Here you will find a variety of pre-configured model profiles (.ymls) for creating LRD-YOLO models. These include the baseline YOLOv8 model, LRD-YOLO, and various models used in the ablation experiments of the paper. Change the `.yml` file in the command `model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")` to switch between different models for training.

```python
from ultralytics import YOLO
# Load a model
model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")

# Use the model
results = model.train(data="dataset/leaf.yaml", epochs=100, batch=16)  # train the model
```

Refer to the YOLOv8 GitHub repository for more details about epochs, batch size, and other arguments used during the training process. Check the `runs/detect/` directory for the training results.



### Evaluation

👉 LRD-YOLO models automatically remember their training settings, allowing you to validate a model at the same image size and on the original dataset by simply using the following commands. The `best.pt` produced by the model training is stored in the corresponding folder in the directory `runs/detect/`.

```python
from ultralytics import YOLO
# Load a model
model = YOLO('weights/best.pt')  # load a custom model
 
# Validate the model
metrics = model.val(split='test',batch=1)  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 of each category

```



### Log

👉 The experimental results of the comparison experiments as well as the ablation experiments mentioned in the paper are documented in the `log/` directory.
