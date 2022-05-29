function [s_bar, v_norm_cur] = block_opt(H, y, s_bar, N, indicator)

%Corresponds to Algorithm 12 (Block Optimal) in Report 10

% [s_bar, v_norm_cur] = block_opt(H, y, s_bar, N, indicator) applies the optimal
% soluation to each of the submatrices of H
% 
% Inputs:
%     H - K-by-N real matrix with K<N or K<=N but rank deficient
%     y - K-dimensional real vector
%     s_bar - N-dimensional real vector
%     N - integer scalar
%     indicator - 2-by-q integer matrix (indicates submatrices of H)
% 
% Outputs:
%     s_bar - N-dimensional integer vector for the updated sub-optimal solution
%     v_norm - real scalar for the norm of the residual vector corresponding to s_bar


q = size(indicator, 2);

for j=1:q
     
    %firstCol refers to the column of H where the current block starts
    %lastCol refers to the column of H where the current block ends
    firstCol = indicator(1,j);
    lastCol = indicator(2,j);
    
    % Compute y_bar in the psuedocode of the report
    if lastCol == N
        y_bar=y-H(:,1:firstCol-1)*s_bar(1:firstCol-1);
    elseif firstCol == 1
        y_bar=y-H(:,lastCol+1:N)*s_bar(lastCol+1:N);
    else
        y_bar=y-H(:,1:firstCol-1)*s_bar(1:firstCol-1) - H(:, lastCol+1:N)*s_bar(lastCol+1:N);
    end
    
    % Compute optimal solution
    y_adj = y_bar; %- H(:, firstCol:lastCol)*s_bar(firstCol:lastCol);
    H_adj = H(:, firstCol:lastCol);
    t = lastCol-firstCol+1;
    l = repelem(0, t)';
    u = repelem(7, t)';            
    z = obils(H_adj,y_adj,l,u);
    s_bar(firstCol:lastCol) = z;
end

v_norm_cur = norm(y - H*s_bar);
end