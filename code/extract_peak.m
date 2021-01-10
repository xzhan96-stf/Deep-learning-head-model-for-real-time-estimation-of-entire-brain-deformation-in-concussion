function [extract_peak] = extract_peak(raw)
datamatrix = raw; %Timesteps * Channels
[~, channels] = size(datamatrix);
positive_peak_5 = zeros(1,5*channels); %5 Peak information: 1 # of peaks, 4 largest peaks (excluding max).
negative_peak_5 = zeros(1,5*channels);
for channel=1:1:channels
    sequence = datamatrix(:,channel);
    positive_peaks = sort(findpeaks(sequence),'descend'); %Find peaks and then sort in descending order.
    negative_peaks = sort(-findpeaks(-sequence),'ascend');
    positive_peaks = positive_peaks(positive_peaks>0); %Only peaks above 0
    negative_peaks = negative_peaks(negative_peaks<0); %Only peaks below 0
    [a,b] = size(positive_peaks);
    [m,n] = size(negative_peaks);
    
    positive_peak_5(1,5*(channel-1)+1) = a; %Number of positive peaks
    negative_peak_5(1,5*(channel-1)+1) = m; %Number of negative peaks
    if a >= 5
        positive_peak_5(1,(5*(channel-1)+2):5*channel) = positive_peaks(2:5,1);
    else if a >= 2%Fewer than 5 peaks and more than 2 peaks
        positive_peak_5(1,(5*(channel-1)+2):5*(channel-1)+a) = positive_peaks(2:a,1);
        end
    end
    
    if m >= 5
        negative_peak_5(1,(5*(channel-1)+2):5*channel) = negative_peaks(2:5,1);
    else if m>= 2
        negative_peak_5(1,(5*(channel-1)+2):5*(channel-1)+m) = negative_peaks(2:m,1);
        end    
    end    
end
extract_peak = [positive_peak_5, negative_peak_5];
  return;
end