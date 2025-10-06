# Getting Started (for MATLAB Users)

This guide is intended for MATLAB users who want to exploit the features of the **Python Anesthesia Simulator**.
It walks you through setting up your environment and instantiating a `Patient` object from MATLAB using the Python backend.


## 1. Prerequisites

* MATLAB R2024b or later
* Python 3.12 (recommended: use a virtual environment)
* Required Python packages (see below)

## 2. Set Up Python Environment

### 1. Create a virtual environment (optional but recommended):

```bash
    python -m venv your_environment
```

### 2. Activate the environment:

* On Windows:

```bash
    your_environment\Scripts\activate
```

* On MacOS/Linux:
```bash
    source your_environment/bin/activate
```

### 3. Install required packages:

Clone this repository and install the package with:

```bash
    pip install .
```

## 3. Using the Simulator in MATLAB

### Step-by-step Example

```matlab
    
% Set up Python environment in MATLAB
% Windows
env = pyenv('Version', ...
    'your_path\your_environment\Scripts\python.exe');

% Linux/macOS
% env = pyenv('Version', ...
%    '/Users/your_username/your_environment/bin/python');


% Import the simulator module
simulator = py.importlib.import_module('python_anesthesia_simulator.simulator');

% Define patient parameters
age = 18;      % years
height = 170;  % cm
weight = 60;   % kg
gender = 0;    % 0 = female, 1 = male


% Instantiate a Patient object
George = simulator.Patient([age, height, weight, gender]);

```
