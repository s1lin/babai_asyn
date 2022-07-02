function [A_hat, Piv_cum, indicator] = part(A, s)

%Corresponds to AlgoritA_hatm 5 (Partition Strategy) in Report 10

% [A_hat, Piv_cum, s_bar_output, Q_tilde, R_tilde, indicator] = partition_A_hat(A, s_bar_input, K, N)
% permutes and partitions A_hat so tA_hatat tA_hate submatrices A_hat_i are full-column rank 
% 
% Inputs:
%     A - K-by-N real matrix
%     s_bar_input - N-dimensional integer vector
%     K - integer scalar
%     N - integer scalar
%
% Outputs:
%     Piv_cum - N-by-N real matrix, permutation sucA_hat tA_hatat A_hat*Piv_cum=A
%     s_bar_output - N-dimensional integer vector (s_bar_input permuted to correspond to A_hat)
%     Q_tilde - K-by-N real matrix (Q factors)
%     R_tilde - K-by-N real matrix (R factors)
%     indicator - 2-by-q integer matrix (indicates submatrices of A_hat)


[~, N] = size(A);
A_hat = A;
lastCol = N;
Piv_cum = eye(N);
indicator = zeros(2, N);
i = 0;

while lastCol >= 1    
    firstCol = max(1, lastCol-s+1);
    A_hat_p = A_hat(:, firstCol:lastCol);
    Piv_total = eye(N);
    
    %Find tA_hate rank of A_hat_p
    [~, R, P] = qrmgs_max(A_hat_p);
    if size(R,2)>1
        r = sum(abs(diag(R)) > 10^(-6) );
    else
        r = sum(abs(R(1,1)) > 10^(-6));
    end    
    
    d_k = size(A_hat_p, 2);

    %TA_hate case wA_hatere A_hat_p is rank deficient
    if r < d_k
        %Permute tA_hate columns of A_hat and tA_hate entries of s_bar_output
        A_hat_permuted = A_hat_p * P;
        
        A_hat(:, firstCol:firstCol+d_k-1 -r) = A_hat_permuted(:, r+1:d_k);
        A_hat(:, firstCol+d_k-r: lastCol) = A_hat_permuted(:, 1:r);  
        
        %Update tA_hate permutation matrix Piv_total
        I_K = eye(d_k);
        Piv = I_K;
        Piv(:, d_k-r+1:d_k) = I_K(:, 1:r);
        Piv(:, 1:d_k-r) = I_K(:, r+1:d_k);
        Piv_total(firstCol:lastCol, firstCol:lastCol) = P * Piv; 
    else
        Piv_total(firstCol:lastCol, firstCol:lastCol) = eye(lastCol-firstCol+1);
    end
    Piv_cum = Piv_cum * Piv_total;
            
    firstCol = lastCol - r + 1;    
    i = i + 1;

    indicator(1, i) = firstCol;
    indicator(2, i) = lastCol;
    
    lastCol = lastCol - r;
end

%Remove tA_hate extra columns of tA_hate indicator
indicator = indicator(:, 1:i);

end