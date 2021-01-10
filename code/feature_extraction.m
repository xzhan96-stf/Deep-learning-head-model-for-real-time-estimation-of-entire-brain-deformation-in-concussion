%%Final Code for ML feature extraction.
%%Edited by Xianghao Zhan 01/09/2021

%Directory Definition
Dir_code = 'G:\共享云端硬盘\Head Models (KTH, GHBMC, ML)\ML\Code'; %Input the directory of the code
Dir_X = 'G:\共享云端硬盘\Head Models (KTH, GHBMC, ML)\ML\Github\X'; %Input the directory of the X/kinematics data
FileName_kinematics = 'Example_kinematics.mat'; %Input the name of your kinematics file (e.g., 'Example_kinematics.mat')
FileName_X = 'X_test.mat'; %Input the name of your X file (e.g., 'X_test.mat')

%%1. Formalize signal into 3D signal matrix (time *  channel * impacts)
cd(Dir_X);
load(FileName_kinematics);
time_duration = [];
for i = 1:1:size(impact,2)
    time_duration = [time_duration,size(impact(i).t,1)];
end
max_time = max(time_duration);
signal_matrix = zeros(max_time,8,size(impact,2)); 
SR = round(1/(impact(1).t(10)-impact(1).t(9))); 

cd(Dir_code);
for impact_id = 1:1:size(impact,2)
    time_duration = size(impact(impact_id).ang_vel,1); %Impute 0 to those sequences not long enough.
    signal_matrix(1:time_duration,1:3,impact_id) = impact(impact_id).ang_vel;
    signal_matrix(1:time_duration,4:6,impact_id) = impact(impact_id).ang_acc;
    signal_matrix(1:time_duration,7,impact_id) = extract_magnitude(impact(impact_id).ang_vel);
    signal_matrix(1:time_duration,8,impact_id) = extract_magnitude(impact(impact_id).ang_acc);
end

%%2. Extract Features from the signals
cd(Dir_code);
feature_matrix = zeros(size(signal_matrix,3),20*size(signal_matrix,2));
for impact_id = 1:1:size(signal_matrix,3)
    feature_max = max(signal_matrix(:,:,impact_id)); 
    feature_min = min(signal_matrix(:,:,impact_id)); 
    feature_int = trapz(impact(impact_id).t,signal_matrix(1:size(impact(impact_id).t,1),:,impact_id)); 
    feature_absint = trapz(impact(impact_id).t,abs(signal_matrix(1:size(impact(impact_id).t,1),:,impact_id))); 
    feature_ema = exp_mov_avg(signal_matrix(1:size(impact(impact_id).t,1),:,impact_id), SR, impact(impact_id).t); 
    feature_peaks = extract_peak(signal_matrix(:,:,impact_id));  
    feature_matrix(impact_id,:) = [feature_max, feature_min, feature_int, feature_absint, feature_ema, feature_peaks];
end
cd(Dir_X);
X_test = feature_matrix;
save(FileName_X,'X_test'); 

