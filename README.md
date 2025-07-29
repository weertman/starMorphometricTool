# starMorphometricTool
Use of photos for morphometrics often relies on manual measurement of a calibration object (ruler) to get a px/measurement conversion for the image. 

This app uses a combination of a sea star detecting and segmenting yolo11 instance segmentation model and a opencv checkerboard calibration module to measure the area of stars, estimate arm lengths from center, and get a measurement of star shape anisotropy. Checkerboard calibration accounts for camera angle and optical distortion but comes with some caveats. The webcamera's perspective is projected onto the checkerboard flattening the image onto it. This corrects for perspective error and provides accurate measurements if the star is in the middle of the checker board. But, operation also neccesarily causes some reduction in resolution of the projected image as it is warped to the checkerboard, this is OK. If you move the camera after calibration you must recalibrate to the new location.

## How to Use

### Initial Setup
- Connect a USB webcam (tested with Logitech C270)
- Launch the application: `python main.py` from the `src/starMorphometricTool` directory
- Prepare a **flat, non-square** checkerboard with known square dimensions

### Step 1: Calibrate with Checkerboard
- **Position checkerboard**: Place it flat in front of the camera
- **Enter parameters** in the left panel:
  - Checkerboard rows (number of squares)
  - Checkerboard columns (number of squares)  
  - Square size in millimeters
- Click **"Start Stream"** to view live feed
- Click **"Detect Checkerboard"**
  - Green overlay confirms successful detection
  - If detection fails: improve lighting, ensure full board visibility, verify dimensions
  - You need to calibrate with your container full over water over the checkerboard so that it can take into account optical distortion of water

### Step 2: Capture Specimen
- **Keep camera and checkerboard in exact same position**
- Place specimen in **center of checkerboard** for best accuracy
- Click **"Start Detections"** to activate YOLO model
- Position specimen until detection box appears
- Click **"Get Detection"** to capture

### Step 3: Analyze Morphometrics
- Click **"Run Morphometrics"** to process the specimen
- **Adjust parameters** using sliders:
  - **Smoothing Factor** (1-15): Reduces noise in contour
  - **Prominence Factor** (0.01-1.0): Sensitivity for arm tip detection  
  - **Distance Factor** (0-15): Minimum separation between arms
  - **Arm Rotation**: Rotate which arm is labeled as #1
- **Interactive editing** on polar plot:
  - Click to add missed arm tips
  - Shift+click to remove incorrect detections

### Step 4: Save Results
- Enter your 3-letter initials
- Add any relevant notes
- Click **"Save Morphometrics"**
- Data saved in organized folder structure:

```
measurements/
└── group_name/
└── specimen_id/
└── mm_dd_yyyy/
└── mFolder_N/
├── morphometrics.json
├── corrected_mask.png
└── corrected_object.png
```

### Tips for Best Results
- ⚠️ **Critical**: Keep specimen within checkerboard boundaries
- 📏 Measurement accuracy decreases toward board edges
- 💡 Use consistent, diffuse lighting to avoid shadows
- 🎯 For specimens with short arms, manually adjust detection parameters
- 🔄 Use "Re-run Morphometrics" tab to reanalyze saved specimens

### Measurements Collected
- Total area (mm²)
- Number of arms detected
- Individual arm lengths (mm)
- Major/minor axis dimensions
- Arm tip positions
- Full contour coordinates

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
the tool can be run 
```bash
cd $full_path_project$/starMorphometricTool/src/starMorphometricTool
python main.py
```








