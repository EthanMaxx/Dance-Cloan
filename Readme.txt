# 🎶 Dance Clone

Cyberpunk-style real-time **pose detection + dancing clone effect** built with [MediaPipe](https://github.com/google/mediapipe), OpenCV, and Matplotlib.

---

## ✨ Features

* Real-time pose detection using **MediaPipe**
* Glowing **cyberpunk stick-figure clone**
* Dynamic color transitions every 3 seconds
* Real-time **3D pose rendering** using `matplotlib`
* Smooth overlay blending for polished visuals
* Works with any standard webcam

---

## ⚙️ Requirements

* Python **3.11** (⚠️ MediaPipe does not yet support Python 3.12/3.13)
* A working webcam
* VS Code (recommended) or any terminal

---

## 📦 Installation

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/Dance-Clone.git
cd Dance-Clone
```

### 2. Create virtual environment

**Windows (Command Prompt):**

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux (bash):**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r req.txt
```

---

## 📄 req.txt

This project was tested with these pinned versions:

```
mediapipe==0.10.21
opencv-contrib-python==4.11.0.86
numpy==1.26.4
matplotlib==3.10.6
```

---

## ▶️ Run the project

```bash
python dance.py
```

* A webcam window will open with the clone overlay.
* Press **q** to quit.

---

## 📂 File Structure

```
Dance-Clone/
│── dance.py           # main script
│── req.txt            # dependencies
│── README.md          # project documentation
│── sampleviddance.mp4 # sample output video
│── .venv/             # virtual environment (ignored in git)
```

---

## ❓ Troubleshooting

* **PowerShell activation error**: use Command Prompt instead (`.venv\Scripts\activate.bat`).
* **Camera not opening**: edit `main(cam_index=0)` in `dance.py` → try `cam_index=1`.
* **Slow FPS**: comment out matplotlib 3D plotting lines if you only need 2D overlay.
* **Reinstall cleanly**:

  ```bash
  pip install --force-reinstall -r req.txt
  ```

---

## 📸 Demo

(Add a GIF or screenshot of the clone effect if you want!)

---
