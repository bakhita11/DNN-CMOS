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

## Device-Level Analysis Scripts

The following scripts implement the device-level analyses reported in the
manuscript (Sections on device benchmarking, physical admissibility, and the
architecture-level study). They are independent of, and complementary to, the
main DNN surrogate / Bayesian optimization pipeline.

### gaa_device_model.py
A transparent, physics-based compact device model for gate-all-around (GAA)
nanosheet (NS) and forksheet (FS) field-effect transistors. It implements the
standard device-physics relations used in compact modeling: thermionic
subthreshold conduction with a finite subthreshold slope, drain-induced barrier
lowering (DIBL) via a drain-bias-dependent threshold shift, and above-threshold
drift current with mobility, velocity saturation, and source/drain series
resistance. Running the script computes Id–Vg / Id–Vd characteristics and
extracts subthreshold swing (SS), DIBL, threshold voltage, on-current (Ion),
off-current (Ioff), and the Ion/Ioff ratio for both NS and FS configurations.

*Note:* This is a self-contained compact device model implementing standard GAA
transport physics. It is independent of, and not a substitute for, the Berkeley
BSIM-CMG standard model; the parameter values used are illustrative 5 nm-class
defaults and are not calibrated to a specific foundry process. The forksheet
configuration is represented through its dielectric-wall asymmetry (slightly
weaker electrostatic control and higher series resistance than the symmetric
nanosheet) as a documented compact-model-level approximation.

### physics_bounded_training.py
Demonstrates physical admissibility of the surrogate's predictions. It compares
an unconstrained baseline DNN with a model whose output layer computes
`Vout = VDD * sigmoid(z)`, which restricts every prediction to the physical band
(0, VDD) by construction. Both models are evaluated on the in-distribution test
set and on an out-of-distribution (OOD) probe that samples parameter
combinations beyond the training ranges — the regime a Bayesian optimizer can
explore. The script reports test R2 and the fraction and magnitude of
physically out-of-bound predictions, showing that the bounded model eliminates
the OOD bound violations produced by the unconstrained baseline.

### architecture_sweep.py
Uses `gaa_device_model.py` to study two architecture-critical structural
degrees of freedom for nanosheet devices: (1) the number of stacked sheets
(1–5), showing how on-current scales with effective width; and (2) the sheet
aspect ratio (width-to-height) at fixed cross-sectional area, showing the trend
in electrostatic control (SS and DIBL). Outputs printed tables and the figure
`architecture_sweep.png`.

*Note:* In the aspect-ratio sweep, the dependence of SS and DIBL on aspect ratio
is imposed as a documented compact-model-level trend (thinner channels yield
stronger gate control), not derived from first-principles three-dimensional
electrostatics. It encodes the known qualitative behavior; quantitative values
require TCAD. The sheet-count scaling of on-current, by contrast, follows
directly from the effective-width physics in the model.

### Requirements
