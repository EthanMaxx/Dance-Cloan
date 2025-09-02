Good — let’s fix this step-by-step inside **VS Code**. Here’s the thing: your PowerShell blocked activation (that’s normal on Windows), and some packages got installed outside the virtual environment. I’ll show the simplest, bulletproof path: switch the VS Code terminal to **Command Prompt**, activate the venv, install a conflict-free set of packages, verify, and run the script.

Follow every step exactly (copy-paste the commands). I’ll mark optional PowerShell commands if you prefer to stay in PowerShell.

---

## 0) Quick summary of what happened

* PowerShell blocked `Activate.ps1` because of Execution Policy.
* You installed packages while **not** in the `.venv`, so they landed in your global Python. That’s OK — we’ll set the venv and re-install the right packages inside it.

---

## 1) Open a **Command Prompt** terminal inside VS Code (preferred)

1. In VS Code, open Terminal: \`Ctrl + \`\` (backtick)
2. In the terminal panel, click the small down-arrow next to the **+** button → choose **Command Prompt**.

   * If Command Prompt is not listed, choose **Select Default Profile** → pick **Command Prompt**, then **Terminal → New Terminal**.

You should now see something like:

```
C:\Users\rohit\Desktop\ANDC_workshop_g2-main\Dance-Clone-main>
```

---

## 2) Activate your virtual environment (Command Prompt)

If the `.venv` folder already exists (you created it earlier), run:

```cmd
cd C:\Users\rohit\Desktop\ANDC_workshop_g2-main\Dance-Clone-main
.venv\Scripts\activate.bat
```

After this, your prompt should change and show `(.venv)` at the start.

If `.venv` does **not** exist or you want to recreate it with your system Python 3.11:

```cmd
cd C:\Users\rohit\Desktop\ANDC_workshop_g2-main\Dance-Clone-main
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
```

> If you *must* stay in PowerShell instead of switching to Command Prompt, run this (PowerShell):
>
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\.venv\Scripts\Activate.ps1
> ```

---

## 3) Upgrade pip and install a stable, compatible dependency set

While the `(.venv)` is active, copy–paste these exact commands:

```cmd
python -m pip install --upgrade pip setuptools wheel
pip uninstall opencv-python -y       # remove conflicting opencv if previously installed globally
pip install mediapipe==0.10.21 opencv-contrib-python==4.11.0.86 numpy==1.26.4 matplotlib==3.10.6
```

Why this set?

* `mediapipe==0.10.21` is compatible with Python 3.11.
* `opencv-contrib-python==4.11.0.86` is the OpenCV build mediapipe expects.
* `numpy==1.26.4` matches mediapipe/opencv expectations.
* `matplotlib==3.10.6` for the 3D plotting.

(If pip asks to upgrade/install other packages, allow it — that’s normal.)

---

## 4) Verify the environment imports (sanity check)

With the venv still active, run:

```cmd
python -c "import cv2, mediapipe as mp, numpy as np, matplotlib; print('OK', cv2.__version__, mp.__version__, np.__version__, matplotlib.__version__)"
```

Expected output starts with `OK` and prints versions. If you see `OK`, you’re ready.

---

## 5) Run your script

Still in the same terminal:

```cmd
python dance.py
```

* A webcam window should open with the cyberpunk skeleton overlay.
* Focus the OpenCV window and press `q` to quit.

---

## 6) Make `req.txt` stable (copy & paste into the file and save)

Open `req.txt` in VS Code and replace its content with this (then save):

```
mediapipe==0.10.21
opencv-contrib-python==4.11.0.86
numpy==1.26.4
matplotlib==3.10.6
```

Later you can run:

```cmd
pip install -r req.txt
```

to reproduce the environment.

---

## 7) If something fails, common fixes

* **Activate still blocked**: switch to Command Prompt (steps above). That’s the easiest workaround.
* **“Module not found” for mediapipe after install**: make sure `(.venv)` is shown in prompt. If not active, the packages are not in venv. Activate then reinstall.
* **Camera fails to open**: test camera index:

  ```cmd
  python -c "import cv2; cap=cv2.VideoCapture(0); print(cap.isOpened()); cap.release()"
  ```

  If `False`, try `VideoCapture(1)` in the test.
* **I installed packages before activating venv**: that installed them globally. You can ignore them — reinstalling inside venv is safe. If you want to clean global installs later, I can give exact uninstall commands.

---

## 8) Exact full command sequence (copy-paste this into a **Command Prompt** terminal in VS Code)

```cmd
cd C:\Users\rohit\Desktop\ANDC_workshop_g2-main\Dance-Clone-main
.venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
pip uninstall opencv-python -y
pip install mediapipe==0.10.21 opencv-contrib-python==4.11.0.86 numpy==1.26.4 matplotlib==3.10.6
python -c "import cv2, mediapipe as mp, numpy as np, matplotlib; print('OK', cv2.__version__, mp.__version__, np.__version__, matplotlib.__version__)"
python dance.py
```

---

Do these steps now. If anything prints an error, copy the exact terminal output here and I’ll give the exact next command to fix it.
