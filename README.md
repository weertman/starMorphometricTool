# starMorphometricTool
Measures star areas on a checkerboard

![Local Image](images/demo.png)

# Install instructions

### Set up anaconda env
```bash
conda create -n starMorphometricTool python=3.9 -y
conda activate starMorphometricTool
```

### Clone the Repository
First, clone this repository to your local machine:
```bash
git clone https://github.com/weertman/starMorphometricTool.git
cd starMorphometricTool
```

### Install pytorch from source
https://pytorch.org/get-started/locally/

### Install packages
```bash
pip install PySide6 opencv-python-headless ultralytics numpy matplotlib scipy
```

# Choosing a model
Currently the package uses a nano sized model by default (its ok, not great, good for demos)
if you wish to use a different model I've placed options into a dropbox folder
```bash
https://www.dropbox.com/scl/fo/gynp911wspftbuyzmoqxe/AG2gyWISAqav4282zeYvQdE?rlkey=t8ve0p8feh94i28a9669l43ov&st=wxzyixv7&dl=0
```
You will then have to manually change the path to the model path in Yolov8WebCamStreamGUI.py

```bash
# Load YOLOv8 model
path_model = os.path.join('..', '..', 'models', 'yolov8-n-pretrained.pt')
self.yolo_model = YOLO(path_model)
```
change this to..
```bash
path_model = os.path.join('PATH TO YOUR MODEL')
```

# Running the tool
The tool is currently a demo and so you can run it simply by running the python script Yolov8WebCamStreamGUI.py
e.g.,
```bash
cd starMorphometricTool
python src/starMorphometricTool/Yolov8WebCamStreamGUI.py
```



