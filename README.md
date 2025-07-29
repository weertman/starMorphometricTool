# starMorphometricTool
Use of photos for morphometrics often relies on manual measurement of a calibration object (ruler) to get a px/measurement conversion for the image. 

This app uses a combination of a sea star detecting and segmenting yolo11 instance segmentation model and opencv checkerboard calibration module to measure the area of stars, estimate arm lengths from center, and get a measurement of star shape anisotropy. The calibration accounts for camera angle and optical distortion. Behind the scenes the webcam's perspective is projected onto the checkerboard flattening the image onto it. This corrects for perspective error and provides accurate measurements if the star is in the middle of the checker board. The operation also neccesarily causes some reduction in resolution of the projected image as it is warped to the checkerboard, this is OK. 

To use this tool the first step is to install it and verify it works (see below instructions). 

Once installed place a non-square checker board of known checker sizes and counts infront of a static web camera (we use a Logi C270 webcam). The app can then be triggered to find the checkerboard, its pretty good at this the checkerboard MUST be flat, if it fail to find the checkerboard it is either (A) your input dimensions for the checker board are wrong or (B) your lighting sucks or (C) you are occluding parts of the checkboard it must see the whole thing. 

Then, keeping the same static camera arrangement, you can place stars onto the checker board to measure them. NOTE the best measurements will be within the middle of the checkerboard, if the star is off the checkerboard it will miss the part of the star that is off the board. There is increasing measurement error on the sides of the checker board. Rule of thumb PLACE THE STARS IN THE MIDDLE OF THE CHECKER BOARD then get detection.

To find the arm tips the tool uses a contour finding and peak finding algorithms, you can adjust the parameters of these models on the fly to best find the arm tips. It will only work well for long arms (small arms are usually missed) and it is important to tune the parameters, although I've chosen pretty good defaults. Segmentation quality has a bit impact on the results. You can also rotate which found arm is the first arm based on visual reference to the madreporite, if you care.

![Local Image](images/demo.png)

# Install instructions

## Prerequisites

### Installing Anaconda

This tool requires Python 3.9, which is installed and managed via Anaconda. Anaconda is a free, open-source distribution for scientific computing that includes Python, the conda package manager, and many useful libraries. If you don't have Anaconda installed, follow these steps to download and install it.

1. **Download Anaconda**:
   - Visit the official [Anaconda download page](https://www.anaconda.com/download).
   - Select the latest version of Anaconda for your operating system (Windows, macOS, or Linux). Choose the graphical installer if you're new to this process.
   - Note: The tool uses Python 3.9, but Anaconda's latest version includes a newer Python (e.g., 3.13) by default. You can create a specific environment with Python 3.9 during setup (as shown in the installation instructions below).

2. **Install Anaconda**:
   Follow the platform-specific instructions below. The installation typically takes 10-20 minutes and requires about 3-5 GB of disk space.

   #### Windows
   - Double-click the downloaded `.exe` file (e.g., `Anaconda3-2025.06-1-Windows-x86_64.exe`).
   - Follow the installation wizard:
     - Agree to the license.
     - Select "Just Me" (recommended for most users).
     - Choose an installation location (default is fine).
     - **Do not** check "Add Anaconda to my PATH environment variable" (recommended to avoid conflicts; use the Anaconda Prompt instead).
     - Check "Register Anaconda as my default Python" if desired.
   - Click "Install" and wait for completion.
   - After installation, search for and open the "Anaconda Prompt" from the Start menu.

   #### macOS
   - Note: Anaconda 2025.06 is the last version with support for Intel-based macOS (osx-64). For Apple Silicon (arm64), use the appropriate installer.
   - Double-click the downloaded `.pkg` file (e.g., `Anaconda3-2025.06-1-MacOSX-arm64.pkg` for Apple Silicon or `Anaconda3-2025.06-1-MacOSX-x86_64.pkg` for Intel).
   - Follow the installation wizard:
     - Agree to the license.
     - Select an installation location (default is fine).
   - The installer will add Anaconda to your PATH automatically.
   - After installation, open the Terminal app (found in Applications > Utilities).

   #### Linux
   - Open a terminal and navigate to the download location (e.g., `cd ~/Downloads`).
   - Run the installer script:
     ```
     bash Anaconda3-2025.06-1-Linux-x86_64.sh
     ```
     (Replace the filename with the one you downloaded.)
   - Follow the prompts:
     - Agree to the license by typing `yes`.
     - Choose an installation location (default is `~/anaconda3`).
     - Allow the installer to add Anaconda to your PATH by typing `yes` (this updates your `~/.bashrc` file).
   - Close and reopen the terminal for changes to take effect.

3. **Verify the Installation**:
   - Open the Anaconda Prompt (Windows) or terminal (macOS/Linux).
   - Run the following command:
     ```
     conda --version
     ```
     - You should see output like `conda 25.5.1` (version may vary).
   - If it doesn't work, ensure Anaconda is added to your PATH or restart your computer.

4. **Update Anaconda** (Optional but Recommended):
   - In the Anaconda Prompt or terminal, run:
     ```
     conda update conda
     ```
   - Then update all packages:
     ```
     conda update --all
     ```

Once Anaconda is installed, proceed to the setup instructions below to create the environment for this tool.

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
Currently the package uses a small sized model by default 
if you wish to use a different model I've placed options into a dropbox folder
```bash
https://www.dropbox.com/scl/fo/gynp911wspftbuyzmoqxe/AG2gyWISAqav4282zeYvQdE?rlkey=t8ve0p8feh94i28a9669l43ov&st=wxzyixv7&dl=0
```
You will then have to manually change the path to the model path in main.py

```bash
# Load YOLOv8 model
path_model = os.path.join('..', '..', 'models', 'best.pt')
self.yolo_model = YOLO(path_model)
```
change this to..
```bash
path_model = os.path.join('PATH TO YOUR MODEL')
```

# Running the tool
the newest version of the tool can be run 
```bash
cd $full_path_project$/starMorphometricTool/src/starMorphometricTool
python main.py
```
The older version can be run by running
Yolov8WebCamStreamGUI.py
e.g.,
```bash
cd $full_path_project$/starMorphometricTool/src/starMorphometricTool
python Yolov8WebCamStreamGUI.py
```

# Future TODO list

### 1. Expand list USB cameras that can work 
(e.g., GeT cameras and Basler Cameras) for higher resolution imaging. This is a bit.. complex to do and would require a fair bit of script reworking for each backend added but would make the tool more user friendly. o1 is really good at doing this :)

### 2. Add auto madreporite segmentation for body axis detection
madreporiteSegmentor now works 95% of the time, see..
```bash
https://github.com/weertman/madreporiteSegmentor
```
would be cool to add a one click auto rotation feature which the user can override if it fails.

### 3. Add volume estimation
So this is a bit speculation, and would have varying results with camera model and lighting, but under the correct conditions in theory it could work well..
Not sure, but would be fun to test.. check out..
```bash
https://github.com/apple/ml-depth-pro
```
I've tested into before in other contexts and it seems to work remarkably well, would be interesting to see how close it's measurements can be to volumetric measures.
To do this we will use the checkerboard as a distance calibration and the segmentation to cut the volume where it meets the checkerboard. This should technically be possible, the big unknown is accuracy. While the other measure of this project have prinicipled underpining (camera calibration is reliable) this method would rely on a neural network which could have varying performance with lighting, checkboard, background info etc. Infact the best images probably would be fairly high resolution with some background context to help the model 'see'.

### 4. Expand morphometric analyses within and across measurement dates
I have started this, but feel that I can make solutions as data becomes available. 

### 5. Add hidden class exclusion for working with the models (e.g., star/prey models exclude prey detections)






