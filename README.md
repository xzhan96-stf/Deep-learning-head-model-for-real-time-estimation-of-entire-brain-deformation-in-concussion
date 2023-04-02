# Deep-learning-head-model-for-real-time-estimation-of-entire-brain-deformation-in-concussion

This file includes codes and example input output files for the paper entitled "Rapid Estimation of Entire Brain Strain Using Deep Learning Models". To make it accessible to more researchers, we have packaged the models into a .exe file so that those not familiar with Python can use this model as well.

## Guidelines

Thank you very much for your attention to this deep learning head model. This model takes in the kinematics as input (angular velocity and angular acceleration) and output the brain strain of the entire brain (4124 brain elements).

To use this deep learning head model, please follow the following steps (if you are not familar with Python environment set up, skip Step 2-3, go to the easy use mode):

### Step 1: Kinematics processing

a) If you have access to MATLAB, please provide a .mat file as kinematics input:

Provide a .mat file with kinematics as a 1*N struct named impact, where N is the number of impacts needing to be predicted. Please refer to the Example_kinematics.mat for more details.
	The impact must have at least the following fields: 
	1)ang_vel (angular velocity, in rad/s): for each impact, provides a T * 3 matrix where T is the time step and 3 denotes the three channels (x-axis, y-axis, z-axis);
	2)ang_acc (angular acceleration, in rad/s^2): for each impact, provides a T * 3 matrix where T is the time step and 3 denotes the three channels (x-axis, y-axis, z-axis);
	3)t (time step, in s): for each impact provide a T * 1 matrix denoting the sampling time, where T is the time step.
	
b) If you do now have access to MATLAB you can simply input .csv files:

Provide a folder of .csv files, each .csv file represents an impact. Within each .csv, the first column is the sampling time (t, which is T-by-1), the second to the fourth columns are the angular velocity (ang_vel) at each of the sampling time, the fifth to the seventh columns are the angular acceleration (ang_acc) at each of the sampling time. There is no need to provide any column names in the .csv files. 

### Easy Use Mode (Use the model as a black box):

Download the model_predict_user.exe file is enough. Open a Windows Powershell/Linux command window in the directory where you have the kinematics .mat file/kinemaitcs folder and the .exe file. Run the following command line:

`
python model_predict_user.py {filename_kinematics.mat/foldername_kinematics}
`

The file will output the predicted MPS for all N samples and 4124 brain elements in both a "MPS Prediction.mat" file and a "MPS Prediction.csv" file (N * 4124)

### Step 2: Python environment setup

Install python and anaconda (suggested). Create a new environment using the conda create command. Specify the environment name (replace "myenv" with your desired environment name):

`
conda create -n myenv python=3.8
`

Activate the new environment:

`
conda activate myenv
`

Prepare the required python packages for deep learning head model, please refer to requirements.txt. Install the packages from the requirements.txt file using pip. Make sure you are in the directory containing the requirements.txt file or provide the full path to the file:

`
pip install -r requirements.txt
`

Now, you have created an Anaconda environment using the packages specified in the requirements.txt file, and the environment is activated. You can start using it for MLHM.



### Step 3: Run machine learning head model

Simply open a terminal with the python environment activated (with all the packages required in the requirements.txt) at the same directory with the kinematics file/kinematics folder, model_predict_user.py, model folder, then run the following command line

`
python model_predict_user.py {filename_kinematics.mat/foldername_kinematics}
`


Enjoy! Thanks!



Please feel free to contact us if you find some problem with this guideline: xzhan96@stanford.edu.

Please cite the following paper if you use this model for calculation in your project:

Zhan, Xianghao, et al. "Rapid Estimation of Entire Brain Strain Using Deep Learning Models." IEEE Transactions on Biomedical Engineering (2021).
