# How to Re-enable PDE Loss for Physical Systems Modeling Under Partial Observation

## Official Implementation

This repository is the official implementation of "How to Re-enable PDE Loss for Physical Systems Modeling Under Partial Observation". 

## Introduction

In science and engineering, machine learning techniques are increasingly successful in physical systems modeling (predicting future states of physical systems).
Effectively integrating PDE loss as a constraint of system transition can improve the model's prediction by overcoming generalization issues due to data scarcity, especially when data acquisition is costly. However, in many real-world scenarios, due to sensor limitations, the data we can obtain is often only partial observation, making the calculation of PDE loss seem to be infeasible, as the PDE loss heavily relies on high-resolution states. We carefully study this problem and propose a novel framework named Re-enable PDE Loss under Partial Observation (RPLPO). The key idea is that although enabling PDE loss to constrain system transition solely is infeasible, we can re-enable PDE loss by reconstructing the learnable high-resolution state and constraining system transition simultaneously. Specifically, RPLPO combines an encoding module for reconstructing learnable high-resolution states with a transition module for predicting future states. The two modules are jointly trained by data and PDE loss. We conduct experiments in various physical systems to demonstrate that RPLPO has significant improvement in generalization, even when observation is sparse, irregular, noisy, and PDE is inaccurate. 

![PICL. Base-training period (left): the encoding module is trained with a physics loss without available fine-grained data, and the transition module is trained with a combination of data loss and physics loss. Inference Period (right): given a coarse-grained observation to predict the future coarse-grained observations.](fig.png)