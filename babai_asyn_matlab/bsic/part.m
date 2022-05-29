function [H, Piv_cum, s_bar_output, indicator] = part(H_original, s_bar_input, K, N)

%Corresponds to Algorithm 5 (Partition Strategy) in Report 10

% [H, Piv_cum, s_bar_output, Q_tilde, R_tilde, indicator] = partition_H(H_original, s_bar_input, K, N)
% permutes and partitions H so that the submatrices H_i are full-column rank 
% 
% Inputs:
%     H_original - K-by-N real matrix
%     s_bar_input - N-dimensional integer vector
%     K - integer scalar
%     N - integer scalar
%
% Outputs:
%     Piv_cum - N-by-N real matrix, permutation such that H*Piv_cum=H_original
%     s_bar_output - N-dimensional integer vector (s_bar_input permuted to correspond to H)
%     Q_tilde - K-by-N real matrix (Q factors)
%     R_tilde - K-by-N real matrix (R factors)
%     indicator - 2-by-q integer matrix (indicates submatrices of H)



H = H_original;
s_bar_output = s_bar_input;
lastCol = N;
Piv_cum = eye(N);
R_tilde = zeros(K,N);
Q_tilde = zeros(K,N);
indicator = zeros(2, N);
i = 0;

while lastCol >= 1    
    firstCol = max(1, lastCol-K+1);
    H_cur = H(:, firstCol:lastCol);
    s_cur = s_bar_output(firstCol:lastCol);
    Piv_total = eye(N);
    
    %Find the rank of H_cur
    [Q,R,P]=qr(H_cur);
    if size(R,2)>1
        r = sum( abs(diag(R)) > 10^(-6) );
    else
        r = sum( abs(R(1,1)) > 10^(-6));
    end
    
        
    K_cur = size(H_cur, 2);

    %The case where H_cur is rank deficient
    if r < K_cur
        %Permute the columns of H and the entries of s_bar_output
        H_permuted = H_cur * P;     
        s_cur_permuted = P' * s_cur;
        H(:, firstCol:firstCol+K_cur-1 -r ) = H_permuted(:, r+1:K_cur);
        H(:, firstCol+K_cur-r: lastCol) = H_permuted(:, 1:r);
        s_bar_output(firstCol:firstCol+K_cur-1-r) = s_cur_permuted(r+1:K_cur);
        s_bar_output(firstCol+K_cur-r: lastCol) = s_cur_permuted(1:r);      
        
        %Update the permutation matrix Piv_total
        I_K = eye(K_cur);
        Piv = I_K;
        Piv(:, K_cur-r+1:K_cur) = I_K(:, 1:r);
        Piv(:, 1:K_cur-r) = I_K(:, r+1:K_cur);
        Piv_total(firstCol:lastCol, firstCol:lastCol) = P * Piv; 
    else
        %Permute the columns of H and the entries of s_bar_output
%         H(:, firstCol:lastCol) = H_permuted;
%         s_bar_output(firstCol:lastCol) = s_cur_permuted;
        
        %Update the permutation matrix Piv_total
        Piv_total(firstCol:lastCol, firstCol:lastCol) = eye(lastCol-firstCol+1);
    end
    Piv_cum = Piv_cum * Piv_total;
            
    firstCol = lastCol - r + 1;
    R_tilde(:, firstCol:lastCol) = R(:, 1:r);
    Q_tilde(:, firstCol:lastCol) = Q(:, 1:r);
    
    i = i + 1;
    indicator(1, i) = firstCol;
    indicator(2, i) = lastCol;
    
    lastCol = lastCol - r;
end

%Remove the extra columns of the indicator
indicator = indicator(:, 1:i);

end