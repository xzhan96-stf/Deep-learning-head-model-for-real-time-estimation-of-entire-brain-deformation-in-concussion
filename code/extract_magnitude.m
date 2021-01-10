function [magnitude] = extract_magnitude(data)
magnitude = zeros(size(data,1),1);
for item = 1:1:size(data,1) %Time steps
    sum = 0;
    for vector = 1:1:size(data,2)
       sqr = data(item,vector)^2;
       sum = sum + sqr; %a*a+b*b+c*c
    end
    magnitude(item) = sqrt(sum); 
end
return;
end