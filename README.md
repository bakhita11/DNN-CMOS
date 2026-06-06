# DNN-CMOS

Physics-Informed Deep Learning and Bayesian Optimization for Nanosheet and Forksheet Transistor Design
Overview
This project presents a data-driven framework for modeling and optimization of advanced transistor architectures, specifically nanosheet (NS) and forksheet (FS) devices. The approach integrates deep learning and Bayesian optimization to enable efficient and accurate exploration of the transistor design space under realistic operating conditions.
Key Features
•	Large-scale synthetic dataset generation using LTSpice simulations
•	Variability-aware modeling with noise and process variations
•	Deep neural network (DNN) surrogate model for nonlinear transistor behavior
•	Feature engineering for improved generalization
•	Bayesian optimization for efficient design space exploration
•	Comparative analysis of nanosheet and forksheet transistors
Methodology
1.	Data Generation
Circuit-level simulations are performed using LTSpice to generate a dataset of approximately 150,000 samples. Key parameters include channel length, channel width, supply voltage, temperature, and threshold voltage. Variability and noise effects are incorporated to reflect realistic device behavior.
2.	Feature Engineering
Additional features such as parameter ratios and transformations are introduced to capture nonlinear relationships and improve model performance.
3.	Model Development
A fully connected deep neural network is trained on normalized data to learn the relationship between input parameters and output voltage behavior.
4.	Optimization
Bayesian optimization is applied to identify optimal transistor configurations while minimizing computational cost.
Results
•	High predictive accuracy with $R^2 \approx 0.91$
•	Accurate modeling of transient switching behavior
•	Strong generalization across varying operating conditions
•	Reduced computational cost compared to simulation-based approaches
•	Clear trade-offs identified between NS and FS devices
Requirements
•	Python 3.x
•	NumPy
•	Pandas
•	TensorFlow or PyTorch
•	Scikit-learn
•	Matplotlib
Usage
1.	Generate or load the dataset
2.	Preprocess and normalize input features
3.	Train the DNN model
4.	Evaluate model performance
5.	Apply Bayesian optimization for design exploration
Applications
•	Semiconductor device modeling
•	Circuit design optimization
•	Variability-aware design analysis
•	AI-assisted electronic design automation
Notes
•	The dataset is synthetically generated and based on circuit-level simulations
•	Results depend on simulation parameters and modeling assumptions
•	The framework can be extended to other transistor architectures
Citation
If you use this work, please cite:
Physics-Informed Deep Learning and Bayesian Optimization for Nanosheet and Forksheet Transistor Design at Advanced Nodes
Contact
DR. Bakhita Salman
TAMIU, TX



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


## gaa_device_model.py

A transparent, physics-based compact device model for gate-all-around (GAA)
nanosheet (NS) and forksheet (FS) field-effect transistors. The model
implements the standard device-physics relations used in compact modeling:

- Thermionic subthreshold conduction with a finite subthreshold slope set by
  the gate/depletion capacitance ratio (subthreshold swing above the 60 mV/dec
  room-temperature limit).
- Drain-induced barrier lowering (DIBL) via a drain-bias-dependent threshold
  shift.
- Above-threshold drift current including mobility, velocity saturation, and
  source/drain series resistance.
- GAA effective width derived from the sheet cross-sectional geometry and the
  number of stacked sheets.

The forksheet configuration is represented through its dielectric-wall
asymmetry (slightly weaker electrostatic control and higher series resistance
than the symmetric nanosheet) as a documented compact-model-level
approximation.

### Outputs
Running the script characterizes both NS and FS devices and reports:
subthreshold swing (SS), DIBL, threshold voltage (Vth), on-current (Ion),
off-current (Ioff), and the Ion/Ioff ratio. It also generates:
- `idvg_ns_fs.png` — Id–Vg comparison (linear and saturation)
- `idvd_ns.png` — Id–Vd output family for the nanosheet device

### Usage
```bash
pip install numpy matplotlib
python gaa_device_model.py
```

### Note
This is a self-contained, reproducible compact device model implementing
standard GAA subthreshold and drift transport physics. It is independent of,
and not a substitute for, the Berkeley BSIM-CMG standard model; the parameter
values used are illustrative 5 nm-class defaults and are not calibrated to a
specific foundry process.
 
