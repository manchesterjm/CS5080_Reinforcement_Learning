# Parking Lot Demo — Setup Instructions

> **For Claude Code:** Run all steps below automatically. To find this folder, run:
> `$folder = Get-ChildItem -Path "$env:USERPROFILE\OneDrive\Desktop" -Recurse -Directory -Filter "HW2_QLearning" | Select-Object -First 1 -ExpandProperty FullName`
> Then use `$folder` as the working directory for all subsequent commands. If the venv already exists and works, skip to step 5.

This folder syncs via OneDrive. The path depends on which machine you're on.

## 1. Find your folder path

Open PowerShell and navigate to this folder. The path varies by machine:

- **Desktop (D: drive):** `D:\OneDrive\Desktop\CS5080_Reinforcement_Learning\Homework\HW2_QLearning`
- **Surface Pro (OneDrive):** `C:\Users\<username>\OneDrive\Desktop\CS5080_Reinforcement_Learning\Homework\HW2_QLearning`

```powershell
cd "<your path here>\CS5080_Reinforcement_Learning\Homework\HW2_QLearning"
```

## 2. Create a virtual environment

```powershell
python -m venv venv
```

## 3. Activate it

```powershell
venv\Scripts\activate
```

## 4. Install dependencies (CPU-only torch to save space)

```powershell
pip install numpy matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 5. Run the demo

```powershell
python demo.py
```

## 6. Using the demo

1. Pick a grid size (7x6, 10x12, 14x16, or 18x20)
2. Pick an algorithm (Q-Learning, Double Q-Learning, or DQN)
3. Pick a parking spot number
4. Watch the car navigate to the spot (animated matplotlib window)
5. Close the window to pick another spot or quit with 'q'
