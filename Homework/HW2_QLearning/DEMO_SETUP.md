# Parking Lot Demo — Setup Instructions

## 1. Open PowerShell or Command Prompt

Navigate to this folder:
```
cd D:\OneDrive\Desktop\CS5080_Reinforcement_Learning\Homework\HW2_QLearning
```

## 2. Create a virtual environment

```
python -m venv venv
```

## 3. Activate it

```
venv\Scripts\activate
```

## 4. Install dependencies (CPU-only torch to save space)

```
pip install numpy matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 5. Run the demo

```
python demo.py
```

## 6. Using the demo

1. Pick a grid size (7x6, 10x12, 14x16, or 18x20)
2. Pick an algorithm (Q-Learning, Double Q-Learning, or DQN)
3. Pick a parking spot number
4. Watch the car navigate to the spot (animated matplotlib window)
5. Close the window to pick another spot or quit with 'q'
