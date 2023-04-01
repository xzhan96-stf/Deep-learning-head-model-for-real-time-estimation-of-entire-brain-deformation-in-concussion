from keras import optimizers
from keras.models import model_from_json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os
from scipy.io import loadmat, savemat
import joblib

def YReconstruct(Y, method, Yscaler):
  if method == 'No':
      Y_out = Y
  elif method == 'LOG':
    Y_out = np.exp(Y)
  elif method == 'STD':
    Y_out = Yscaler.inverse_transform(Y)
  elif method == 'LOGSTD':
    Y_out = np.exp(Yscaler.inverse_transform(Y))
  return Y_out

current_dir = os.getcwd()
Dir_results = current_dir + '\\result'
Dir_data_X = current_dir + '\\X'
Dir_data_Y = current_dir + '\\Y'
Dir_model = current_dir + '\\model'

#Load feature of the test data
os.chdir(Dir_data_X)
X = loadmat('X_test.mat')['X_test']

#Load MPS of the test data for validation (if the MPS is available)
if Dir_data_Y:
    os.chdir(Dir_data_Y)
    Y = loadmat('Y_test.mat')['Y_test']
else:
    Y = np.empty(0)

# load json and create model
os.chdir(Dir_model)
Xscaler = joblib.load('Xscaler.joblib')
Yscaler = joblib.load('Yscaler.joblib')
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk!")


#Output prediction
os.chdir(Dir_results)
X_test_std = Xscaler.transform(X)
y_pred_raw = loaded_model.predict(X_test_std)
y_pred = YReconstruct(Y = y_pred_raw, method = 'LOGSTD', Yscaler=Yscaler)
savemat('Predicted_MPS.mat',{'y_pred':y_pred,'X':X,'X_std':X_test_std})

# Evaluate the accuracy of the model on the test set
if Y.shape[0]:
    print('The R2 of the prediction is: ',round(r2_score(Y,y_pred),ndigits=3))
    print('The MAE of the prediction is: ',round(mean_absolute_error(Y,y_pred),ndigits=3))
    print('The RMSE of the prediction is: ', round(np.sqrt(mean_squared_error(Y, y_pred)), ndigits=3))
