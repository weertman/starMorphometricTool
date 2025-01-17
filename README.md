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


