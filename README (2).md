# AfroDance Learn 
**An Interactive AI-Based Afro Dance Learning Platform**

AfroDance Learn is a desktop-based computer vision application designed to help users learn Afro dance styles through real-time pose tracking, instructor skeleton overlays, step-by-step instruction, and movement accuracy scoring.

---

## System Requirements

### Hardware
- Windows 10 or 11
- Webcam
- Open space 5–8 feet from camera

### Software
- Python 3.12.x (required)

---

## Installation (Step-by-Step)

1. Install Python 3.12 from python.org
2. Open PowerShell in the project folder
3. Create virtual environment:
   py -3.12 -m venv .venv
4. Activate environment:
   .venv\Scripts\activate
5. Install dependencies:
   pip install -r requirements.txt

---

## Video Input Requirements

Place instructor video at:
data/instructor.mp4

The video must show full body, front-facing, good lighting, landscape orientation.

One instructor video has already been included into this folder. If you would like, you can add your own instructor video.
---

## Running the Project

1. Extract reference:
   python extract_reference.py
2. Run live trainer:
   python live_score.py

---

## Controls

"["  Previous step  
"]"  Next step  
"SPACE" Restart step  
"S"  Sequence mode  
"P"  Pause  
"O"  Overlay  
"M"  Fullscreen  
"Q"  Quit  

---

AfroDance Learn — making Afro dance education accessible and interactive.
