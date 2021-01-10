function [exp_mov_avg] = exp_mov_avg(raw, SR, t)
datamatrix = raw;
smoothing_factor_1 = 1/(SR*100);
smoothing_factor_2 = 1/(SR*10);
smoothing_factor_3 = 1/SR;
y_1 = zeros(size(datamatrix,1),size(datamatrix,2));
y_2 = zeros(size(datamatrix,1),size(datamatrix,2));
y_3 = zeros(size(datamatrix,1),size(datamatrix,2));
for y_index = 2:size(datamatrix,1) %Time point
    delta_t = t(y_index)-t(y_index-1);
    y_1(y_index,:)=(1-smoothing_factor_1)*y_1(y_index-1,:)+smoothing_factor_1*(datamatrix(y_index,:)-datamatrix(y_index-1,:))/delta_t;
    y_2(y_index,:)=(1-smoothing_factor_2)*y_2(y_index-1,:)+smoothing_factor_2*(datamatrix(y_index,:)-datamatrix(y_index-1,:))/delta_t;
    y_3(y_index,:)=(1-smoothing_factor_3)*y_2(y_index-1,:)+smoothing_factor_3*(datamatrix(y_index,:)-datamatrix(y_index-1,:))/delta_t;
end
exp_mov_avg_1 = max(y_1);
exp_mov_avg_2 = max(y_2);
exp_mov_avg_3 = max(y_3);
exp_mov_avg_1_min = min(y_1);
exp_mov_avg_2_min = min(y_2);
exp_mov_avg_3_min = min(y_3);
exp_mov_avg = [exp_mov_avg_1,exp_mov_avg_1_min, exp_mov_avg_2,exp_mov_avg_2_min, exp_mov_avg_3,exp_mov_avg_3_min];
  return;
end