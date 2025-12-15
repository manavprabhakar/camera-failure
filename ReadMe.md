# Fractured Glass, Failing Cameras: Simulating Physics-Based Adversarial Samples for Autonomous Driving Systems

This is the official github repository for the AAAI 2026 paper: https://arxiv.org/pdf/2405.15033?

Steps:-

## Setup
1. Start by cloning the repository.
  ```
  git clone https://github.com/manavprabhakar/camera-failure.git
  cd camera-failure
  ```
2. Make a python virtual environment.
   ```
   python -m venv .venv
   source .venv/bin/activate
   ```
   For Windows
   ```
   .\venv\Scripts\activate
   ```
4. Install the project specific requirements in the virtual environment.

  ```python
  pip install -r requirements.txt
  ```

## Creating the fractured glass image.
### For simulation
For simulating glass patterns. Set the parameters as desired or use the default parameters and run

```python
python simulation.py
```

###  For PBR
To overlay the simulated pattern on an existing image
```
python PBR.py
```
