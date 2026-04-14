# AfroDance Learn
 
AfroDance Learn is a desktop-based computer vision application that helps users practice Afro dance movements through real-time pose tracking, adaptive instructor-reference visualization, live scoring, and detailed analysis tools.
 
---
 
## Features
 
### Live Training
The main user mode. Provides:
- Webcam-based body tracking
- Adaptive instructor skeleton overlay
- Side-by-side training view
- Live movement scoring
- End-of-session summary
 
### Detailed Analysis Mode
The technical comparison mode. Provides:
- Deeper pose comparison
- Overlap-based visual analysis
- Technical feedback for debugging and validation
 
### Reference Pose Viewer
Allows frame-by-frame inspection of the saved instructor reference skeleton.
 
---
 
## System Requirements
 
### Hardware
- Windows 10 or 11
- Webcam
- Open space for full-body movement
- Good lighting for pose detection
 
### Software
- Python 3.12.x
- Git
 
---
 
## Project Setup Guide
 
### Step 1: Download the project
 
Open Command Prompt or PowerShell and navigate to the folder where you want the project:
 
```powershell
cd C:\Users\YourName\Downloads
```
 
Clone the repository:
 
```powershell
git clone https://github.com/JustinSui624/Afro-Dance.git
```
 
Move into the project folder:
 
```powershell
cd Afro-Dance
```
 
### Step 2: Create a Python virtual environment
 
```powershell
py -3.12 -m venv .venv
```
 
This creates a local environment so project dependencies don't interfere with other Python installations on your computer.
 
### Step 3: Activate the virtual environment
 
```powershell
.venv\Scripts\activate
```
 
After activation, your terminal should show `(.venv)` at the beginning of the command line.
 
### Step 4: Install dependencies
 
```powershell
pip install -r requirements.txt
```
 
This installs: NumPy, OpenCV, MediaPipe, and Matplotlib.
 
### Step 5: Verify project files
 
Make sure the project folder contains these files:
 
```
ui_prototype.py
live_score.py
extract_reference.py
overlay.py
main.py
SLoader.py
pose_utils.py
AfroDanceLearnPose/
```
 
Also verify that the `data` folder exists. If `data/` or `data/references/` are missing, create them manually:
 
```powershell
mkdir data
mkdir data\references
```
 
### Step 6: Provide an instructor video
 
The project expects an instructor dance video at `data/instructor.mp4`.
 
**Option A — Use the included prototype video (recommended):**
When the dashboard opens, click **Use Included Prototype Video**.
 
**Option B — Add your own video:**
Place your video at `data/instructor.mp4`. For best results, use a video that is:
- Full body visible
- Front-facing
- Well lit
- Landscape orientation
- Minimal background clutter
 
### Step 7: Generate reference data and Starting the Application
 
Before live training can work, pose data must be extracted from the instructor video.

After running the command:
```powershell
python ui_prototype.py
```
The user interface then appears to navigate across the application.
To get a dance started, click **Generate Reference Data**.
 
This creates: `data/references/instructor_reference.json`
 
**Normal workflow:**
1. Select Instructor Video or Use Included Prototype Video
2. Generate Reference Data
3. Start Live Training
 
---
 
## Controls
 
### Live Training
 
| Key | Action |
|-----|--------|
| `S` | Sequence mode |
| `[` | Previous step |
| `]` | Next step |
| `SPACE` | Restart current step |
| `P` | Pause |
| `M` | Fullscreen |
| `Q` | Quit |
 
### Reference Pose Viewer
 
| Key | Action |
|-----|--------|
| `SPACE` | Play / Pause |
| `← →` | Previous / Next frame |
| `T` | Toggle quality display |
| `Q` | Quit |
 
---
 
## Color Guide
 
### Live Training
 
| Color | Meaning |
|-------|---------|
| Green | Strong alignment |
| Yellow / Cyan | Decent alignment |
| Orange / Red | Needs improvement |
| Bright Cyan | Instructor reference skeleton |
 
### Detailed Analysis Mode
Uses technical color-based visual comparison to show overlap quality.
 
---
 
## Notes
 
- **Live Training** is the primary end-user feature.
- **Detailed Analysis Mode** and **Reference Pose Viewer** are secondary tools for analysis, debugging, and validation.
- Good camera position and lighting significantly improve tracking quality.
- Python 3.12 is recommended — project dependencies were tested with this version.
 
---
 
## Summary
 
AfroDance Learn combines instructor motion reference generation, real-time pose tracking, adaptive reference alignment, live performance scoring, and technical analysis tools to support interactive Afro dance learning in a desktop application environment.
