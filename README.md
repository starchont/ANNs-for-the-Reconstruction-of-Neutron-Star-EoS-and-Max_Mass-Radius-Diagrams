# ANNs-for-the-Reconstruction-of-Neutron-Star-EoS-and-Max-Mass-Radius-Diagrams

**Description:**
This repository contains the code for my Master's thesis in Computational Physics. My thesis explores the application of Artificial Neural Networks (ANNs) for the reconstruction of the Equation of State (EoS) of neutron stars, with a specific focus on accurately
reproducing the Max-Mass and Radius diagrams. The data utilized for training the neural networks was generated through the numerical solution of Tolman-Oppenheimer-Volkoff (TOV) equations, a system of coupled ODEs.

**Contents:**
1. **Code_for_TOV_solutions:** This folder contains the code used to solve the Tolman-Oppenheimer-Volkoff (TOV) equations, which are a system of coupled ordinary differential equations (ODEs).
2. **Data:** Here, you can find the solutions of the TOV equations for various Equations of State used in the training process.
3. **Code_for_plot:** The code within this directory was employed to generate the plots showcasing the results of the neural network reconstructions.
4. **Code_for_test_unseen_data:** Contains the code for testing observational data obtained from Gravitational waves
5. **main_code_for_ANNs:** This directory houses the main code and necessary classes for data processing, architecture selection, and training of the Artificial Neural Networks (ANNs) utilized in this study.

**Methodology:**
To reconstruct the Equation of State (EoS), a two-stage approach using sequential ANNs and their corresponding reverse networks was employed. The first ANN reconstructs the EoS, taking the pressure at the center of the star as input and outputting the corresponding energy density. The second ANN utilizes the predicted energy density from the first network to generate the maximum mass and radius using the Tolman-Oppenheimer-Volkoff (TOV) equations. The architecture for each network was meticulously chosen through Bayesian optimization.
