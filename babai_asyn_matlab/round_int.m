function rounded_val = round_int(val, lower, upper)

% rounded_val = round_int(val, lower, upper) rounds the entries in val to the
% nearest odd integer in a given box
% 
% Inputs:
%     val - N-dimensional real vector
%     lower - odd integer scalar, lower bound
%     upper - odd integer scalar, upper bound
% 
% Outputs:
%     rounded_val - N-dimensional odd integer vector


%rounded_val = 2*floor(val/2) + 1; 
rounded_val = round(val);
lower = 0;
for i = 1:length(rounded_val)
    if rounded_val(i) < lower
        rounded_val(i) = lower;
    elseif rounded_val(i) > upper
        rounded_val(i) = upper;
    end
end


