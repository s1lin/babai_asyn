function [s_bar, v_norm] = Babai_Gen(H, y, s_bar, N, Q_tilde, R_tilde, indicator, bound)

%Corresponds to Algorithm 6 (Generalized Nearest Plane) in Report 10

% [s_bar, v_norm] = Babai_Gen(H, y, s_bar, N, Q_tilde, R_tilde, indicator, bound) applies
% the nearest plane algorithm to each of the full-column rank submatrices of H
% 
% Inputs:
%     H - K-by-N real matrix
%     y - K-dimensional real vector
%     s_bar - N-dimensional real vector
%     N - integer scalar
%     Q_tilde - K-by-N real matrix (Q factors)
%     R_tilde - K-by-N real matrix (R factors)
%     indicator - 2-by-q integer matrix (indicates submatrices of H)
%     bound - integer scalar for the constraint
% 
% Outputs:
%     s_bar - N-dimensional integer vector for the updated sub-optimal solution
%     v_norm - real scalar for the norm of the residual vector corresponding to s_bar


q = size(indicator, 2);

for j=1:q 
    %Get the QR factorization corresponding to block j
    R = R_tilde(:, indicator(1, j):indicator(2, j));
    Q = Q_tilde(:, indicator(1, j):indicator(2, j));
    
    %firstCol refers to the column of H where the current block starts
    %lastCol refers to the column of H where the current block ends
    firstCol = indicator(1,j);
    lastCol = indicator(2,j);
    
    %r refers to the rank of the currenk block
    r = lastCol-firstCol+1;
    
    % Compute y_tilde
    if lastCol == N
        y_tilde=(y-H(:,1:firstCol-1)*s_bar(1:firstCol-1));
    elseif firstCol == 1
        y_tilde=(y-H(:,lastCol+1:N)*s_bar(lastCol+1:N));
    else
        y_tilde=(y-H(:,1:firstCol-1)*s_bar(1:firstCol-1) - H(:, lastCol+1:N)*s_bar(lastCol+1:N));
    end
    y_tilde = Q' * y_tilde;
    
    % Find the Babai point
    for i=r:-1:1
        if i==r
            s_temp=y_tilde(i)/(R(i,i));
        else
            s_temp=(y_tilde(i)- R(i,i+1:r)*s_bar(i+firstCol:lastCol))/(R(i,i));
        end
        s_bar(i+lastCol-r)=round_int(s_temp, 0, 7);
    end

end

v_norm = norm(y - H*s_bar);