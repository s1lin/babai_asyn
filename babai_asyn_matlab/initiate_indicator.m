function indicator = initiate_indicator(K, N)

% indicator = initiate_indicator(K, N) gives an initial value for the
% indicator for partitioning H
% 
% Inputs:
%     K - integer scalar
%     N - integer scalar
%
% Outputs:
%     indicator - 2-by-q integer matrix (indicates submatrices of H)


q = ceil(N/K);
indicator = zeros(2, q);
lastCol = N;
i = 1;
while lastCol>0
    firstCol = max(1, lastCol-K+1);
    indicator(1,i) = firstCol;
    indicator(2,i) = lastCol;
    
    lastCol = firstCol - 1;
    i = i + 1;
end
end