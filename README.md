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
 
