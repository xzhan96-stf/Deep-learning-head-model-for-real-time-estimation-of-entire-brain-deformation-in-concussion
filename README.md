# Deep-learning-head-model-for-real-time-estimation-of-entire-brain-deformation-in-concussion
This file includes codes and example input output files for the paper entitled "Deep Learning Head Model for Real-time Estimation of Entire Brain Deformation in Concussion"

# Guidelines

Thank you very much for your attention to this deep learning head model. This model takes in the kinematics as input (angular velocity and angular acceleration) and output the brain strain of the entire brain (4124 brain elements), 95 percentile MPS, mean, median and std of the entire brain strain, CSDM(15%).

To use this deep learning head model, please follow the following steps:

1. Provide a .mat file with kinematics as a 1*N struct named impact, where N is the number of impacts needing to be predicted. Please refer to the Example_kinematics.mat for more details.
	The impact must have at least the following fields: 
	1)ang_vel (angular velocity, in rad/s): for each impact, provides a T * 3 matrix where T is the time step and 3 denotes the three channels (x-axis, y-axis, z-axis);
	2)ang_acc (angular acceleration, in rad/s^2): for each impact, provides a T * 3 matrix where T is the time step and 3 denotes the three channels (x-axis, y-axis, z-axis);
	3)t (time step, in s): for each impact provide a T * 1 matrix denoting the sampling time, where T is the time step.

2. Run feature_extraction.m after specifying the directory and name information of the kinematics file and where you want to store the engineered features.

3. Prepare the required python packages for deep learning head model, please refer to requirements.txt.

4. Run model_predict.py after specifying the directory and name information of the features, ground-truth MPS (if available, to check model accuracy) and where you want to store the prediction results.

Enjoy! Thanks!



Please feel free to contact us if you find some problem with this guideline: xzhan96@stanford.edu.

Please cite the following paper if you use this model for calculation in your project:

Zhan, Xianghao, et al. "Deep Learning Head Model for Real-time Estimation of Entire Brain Deformation in Concussion." arXiv preprint arXiv:2010.08527 (2020).
