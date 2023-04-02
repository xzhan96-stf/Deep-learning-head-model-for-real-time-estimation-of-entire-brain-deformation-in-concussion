from keras import optimizers
from keras.models import model_from_json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os
from scipy.io import loadmat, savemat
import joblib
import scipy.signal
import pandas as pd
from scipy.signal import find_peaks
import os
import scipy.io
import sys

def x_transform(x):
    mean = np.load('X_mean.npy')
    std = np.load('X_std.npy')
    return (x - mean) / std

def y_inverse_transform(y):
    mean = np.load('Y_mean.npy')
    std = np.load('Y_std.npy')
    return y * std + mean

def YReconstruct(Y, method):
  if method == 'No':
      Y_out = Y
  elif method == 'LOG':
    Y_out = np.exp(Y)
  elif method == 'STD':
    Y_out = y_inverse_transform(Y)
  elif method == 'LOGSTD':
    Y_out = np.exp(y_inverse_transform(Y))
  return Y_out

def exp_mov_avg(raw, SR, t):
    datamatrix = raw
    smoothing_factor_1 = 1 / (SR * 100)
    smoothing_factor_2 = 1 / (SR * 10)
    smoothing_factor_3 = 1 / SR
    y_1 = np.zeros(datamatrix.shape)
    y_2 = np.zeros(datamatrix.shape)
    y_3 = np.zeros(datamatrix.shape)

    for y_index in range(1, datamatrix.shape[0]):
        delta_t = t[y_index] - t[y_index - 1]
        y_1[y_index, :] = (1 - smoothing_factor_1) * y_1[y_index - 1, :] + smoothing_factor_1 * (
                    datamatrix[y_index, :] - datamatrix[y_index - 1, :]) / delta_t
        y_2[y_index, :] = (1 - smoothing_factor_2) * y_2[y_index - 1, :] + smoothing_factor_2 * (
                    datamatrix[y_index, :] - datamatrix[y_index - 1, :]) / delta_t
        y_3[y_index, :] = (1 - smoothing_factor_3) * y_2[y_index - 1, :] + smoothing_factor_3 * (
                    datamatrix[y_index, :] - datamatrix[y_index - 1, :]) / delta_t

    exp_mov_avg_1 = np.max(y_1, axis=0)
    exp_mov_avg_2 = np.max(y_2, axis=0)
    exp_mov_avg_3 = np.max(y_3, axis=0)
    exp_mov_avg_1_min = np.min(y_1, axis=0)
    exp_mov_avg_2_min = np.min(y_2, axis=0)
    exp_mov_avg_3_min = np.min(y_3, axis=0)
    exp_mov_avg = np.hstack([exp_mov_avg_1, exp_mov_avg_1_min, exp_mov_avg_2, exp_mov_avg_2_min, exp_mov_avg_3, exp_mov_avg_3_min])
    return exp_mov_avg

def extract_magnitude(data):
    magnitude = np.zeros((data.shape[0]))
    for item in range(data.shape[0]):
        sum_squares = np.sum(data[item, :] ** 2)
        magnitude[item] = np.sqrt(sum_squares)
    return magnitude

def extract_peak(raw):
    datamatrix = raw
    _, channels = datamatrix.shape
    positive_peak_5 = np.zeros(5 * channels)
    negative_peak_5 = np.zeros(5 * channels)

    for channel in range(channels):
        sequence = datamatrix[:, channel]
        positive_peaks = np.sort(sequence[find_peaks(sequence, height = 0)[0]])[::-1] # Get peak indices with find_peaks -> extract the peak values -> sort the peak values from large to small
        negative_peaks = np.sort(sequence[find_peaks(-sequence, height= 0)[0]]) # Get negative peak indices with find_peaks -> extract the peak values -> sort the peak values from small to large

        a = len(positive_peaks)
        m = len(negative_peaks)

        # First type of peak feature: number of peaks
        positive_peak_5[5 * channel] = a
        negative_peak_5[5 * channel] = m

        # If there are more than 5 peaks or fewer than 5 peaks
        if a >= 5:
            positive_peak_5[5 * channel + 1: 5 * (channel + 1)] = positive_peaks[1:5]
        elif a >= 2:
            positive_peak_5[5 * channel + 1: 5 * channel + a] = positive_peaks[1:a]

        if m >= 5:
            negative_peak_5[5 * channel + 1: 5 * (channel + 1)] = negative_peaks[1:5]
        elif m >= 2:
            negative_peak_5[5 * channel + 1: 5 * channel + m] = negative_peaks[1:m]

    extract_peak = np.hstack([positive_peak_5, negative_peak_5])
    return extract_peak

def process_kinematics(filename_kinematics, filename_X):
    kinematics_data = scipy.io.loadmat(filename_kinematics)
    impact_data = kinematics_data["impact"]
    time_duration_lst = []
    for i in range(impact_data.shape[1]):
        impact_data[0, i]["t"] = impact_data[0, i]["t"].reshape(-1,)
        time_duration_lst.append(impact_data[0, i]["t"].shape[0]) # Extract the time duration for each impact in SI unit
    max_time = max(time_duration_lst) # Max time over the entire impact dataset
    signal_matrix = np.zeros((max_time, 8, impact_data.shape[1]))
    SR = round(1 / (impact_data[0, 0]["t"][9] - impact_data[0, 0]["t"][8]))

    for impact_id in range(impact_data.shape[1]):
        time_duration = time_duration_lst[impact_id] # The time duration of the current impact (impact_id)
        signal_matrix[:time_duration, 0:3, impact_id] = impact_data[0, impact_id]["ang_vel"]
        signal_matrix[:time_duration, 3:6, impact_id] = impact_data[0, impact_id]["ang_acc"]
        signal_matrix[:time_duration, 6, impact_id] = extract_magnitude(impact_data[0, impact_id]["ang_vel"])
        signal_matrix[:time_duration, 7, impact_id] = extract_magnitude(impact_data[0, impact_id]["ang_acc"])

    # 2. Extract Features from the signals
    feature_matrix = np.zeros((signal_matrix.shape[2], 20 * signal_matrix.shape[1]))
    for impact_id in range(signal_matrix.shape[2]):
        feature_max = np.max(signal_matrix[:, :, impact_id], axis=0) # Channels (8: 0-7)
        feature_min = np.min(signal_matrix[:, :, impact_id], axis=0) # Channels (8: 8-15)
        feature_int = np.trapz(signal_matrix[:time_duration_lst[impact_id], :, impact_id],
                               impact_data[0, impact_id]["t"], axis=0) # Channels (8: 16-23)
        feature_absint = np.trapz(np.abs(signal_matrix[:time_duration_lst[impact_id], :, impact_id]),
                                  impact_data[0, impact_id]["t"], axis=0) # Channels (8: 24-31)
        feature_ema = exp_mov_avg(signal_matrix[:time_duration_lst[impact_id], :, impact_id], SR,
                                  impact_data[0, impact_id]["t"]) #6 * Channels (48: 32-79)
        feature_peaks = extract_peak(signal_matrix[:time_duration_lst[impact_id], :, impact_id]) # 10 * Channels (80)
        feature_matrix[impact_id, :] = np.hstack([feature_max, feature_min, feature_int, feature_absint, feature_ema,
                                                  feature_peaks])

    X_test = feature_matrix
    scipy.io.savemat(filename_X, {"X_test": X_test})

    return feature_matrix

def process_kinematics_csv(t, ang_vel, ang_acc):
    time_duration_lst = [len(t[i]) for i in range(t)] # Extract the time duration for each impact in SI unit
    max_time = max(time_duration_lst)  # Max time over the entire impact dataset
    signal_matrix = np.zeros((max_time, 8, len(t)))
    SR = round(1 / (t[0][9] - t[0][8]))

    for impact_id in range(len(t)):
        time_duration = time_duration_lst[impact_id]  # The time duration of the current impact (impact_id)
        signal_matrix[:time_duration, 0:3, impact_id] = ang_vel[impact_id]
        signal_matrix[:time_duration, 3:6, impact_id] = ang_acc[impact_id]
        signal_matrix[:time_duration, 6, impact_id] = extract_magnitude(ang_vel[impact_id])
        signal_matrix[:time_duration, 7, impact_id] = extract_magnitude(ang_acc[impact_id])

    # 2. Extract Features from the signals
    feature_matrix = np.zeros((signal_matrix.shape[2], 20 * signal_matrix.shape[1]))
    for impact_id in range(signal_matrix.shape[2]):
        feature_max = np.max(signal_matrix[:, :, impact_id], axis=0)  # Channels (8: 0-7)
        feature_min = np.min(signal_matrix[:, :, impact_id], axis=0)  # Channels (8: 8-15)
        feature_int = np.trapz(signal_matrix[:time_duration_lst[impact_id], :, impact_id],
                               t[impact_id], axis=0)  # Channels (8: 16-23)
        feature_absint = np.trapz(np.abs(signal_matrix[:time_duration_lst[impact_id], :, impact_id]),
                                  t[impact_id], axis=0)  # Channels (8: 24-31)
        feature_ema = exp_mov_avg(signal_matrix[:time_duration_lst[impact_id], :, impact_id], SR,
                                  t[impact_id])  # 6 * Channels (48: 32-79)
        feature_peaks = extract_peak(signal_matrix[:time_duration_lst[impact_id], :, impact_id])  # 10 * Channels (80)
        feature_matrix[impact_id, :] = np.hstack([feature_max, feature_min, feature_int, feature_absint, feature_ema,
                                                  feature_peaks])

    X_test = feature_matrix
    scipy.io.savemat(filename_X, {"X_test": X_test})
    return feature_matrix

# Directory Definition
Dir_results = os.getcwd()
Dir_model = Dir_results + '\\model'
filename_X = "X_test.mat"

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 2: #
        filename_kinematics = str(args[0])
        filename_Y = str(args[1])
    elif len(args) == 1:
        filename_kinematics = str(args[0])
        filename_Y = None
    else:
        filename_kinematics = None
        print('Illegal input! Please provide the filename of the kinematics (and the reference MPS if there is any)!')

    # 1. Formalize signal into 3D signal matrix (time * channel * impacts)
    if filename_kinematics.endswith('.mat'):
        # Process the kinematics for MLHM
        X = process_kinematics(filename_kinematics,filename_X)
    else:
        csv_folder = filename_kinematics  # Replace with the path to your folder containing .csv files
        output_mat_file = filename_kinematics + '.mat'  # Replace with the desired output .mat filename

        # Loop through .csv files in the folder
        t = []
        ang_vel = []
        ang_acc = []
        for file in os.listdir(csv_folder):
            if file.endswith('.csv'):
                file_path = os.path.join(csv_folder, file)
                df = pd.read_csv(file_path)
                t.append(df.iloc[:, 0].values.reshape(-1,))
                ang_vel.append(df.iloc[:, 1:4].values)
                ang_acc.append(df.iloc[:, 4:7].values)

        X = process_kinematics_csv(t, ang_vel, ang_acc)



    # load json and create model
    os.chdir(Dir_model)
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk!")
    X_test_std = x_transform(X)
    y_pred_raw = loaded_model.predict(X_test_std)
    y_pred = YReconstruct(Y=y_pred_raw, method='LOGSTD')

    #Output prediction
    os.chdir(Dir_results)
    savemat('MPS Prediction.mat',{'y_pred': y_pred,'X':X,'X_std':X_test_std})
    np.savetxt('MPS Prediction.csv', y_pred, delimiter=',')


    # Evaluate the accuracy of the model on the test set
    if filename_Y:
        # Load MPS of the test data for validation (if the MPS is available)
        Y = loadmat(filename_Y)['label']
        print('The R2 of the prediction is: ',round(r2_score(Y, y_pred),ndigits=3))
        print('The MAE of the prediction is: ',round(mean_absolute_error(Y, y_pred),ndigits=3))
        print('The RMSE of the prediction is: ', round(np.sqrt(mean_squared_error(Y, y_pred)), ndigits=3))