function [H_A, P_cum, z, Q_tilde, R_tilde, indicator] = partition_H_2(H, z_B, m, n)

%Corresponds to Algorithm 5 (Partition Strategy) in Report 10

% [H_A, P_cum, z, Q_tilde, R_tilde, indicator] = partition_H(H, z_B, m, n)
% permutes and partitions H_A so that the submatrices H_i are full-column rank 
% 
% Inputs:
%     H - m-by-n real matrix
%     z_B - n-dimensional integer vector
%     m - integer scalar
%     n - integer scalar
%
% Outputs:
%     P_cum - n-by-n real matrix, permutation such that H_A*P_cum=H
%     z - n-dimensional integer vector (z_B permuted to correspond to H_A)
%     Q_tilde - m-by-n real matrix (Q_qr factors)
%     R_tilde - m-by-n real matrix (R_qr factors)
%     indicator - 2-by-q integer matrix (indicates submatrices of H_A)



H_A = H;
z = z_B;
lastCol = n;
P_cum = eye(n);
R_tilde = zeros(m,n);
Q_tilde = zeros(m,n);
indicator = zeros(2, n);
i = 0;

while lastCol >= 1    
    firstCol = max(1, lastCol-m+1);
    H_cur = H_A(:, firstCol:lastCol);
    z_cur = z(firstCol:lastCol);
    P_tmp = eye(n);
    
    %Find the rank of H_cur
    [Q_qr,R_qr,P_qr]=qr(H_cur);
    
    if size(R_qr,2)>1
        r = sum( abs(diag(R_qr)) > 10^(-6) );
    else
        r = sum( abs(R_qr(1,1)) > 10^(-6));
    end
    H_P = H_cur * P_qr;     
    z_p = P_qr' * z_cur;
        
    size_H_2 = size(H_cur, 2);

    %The case where H_cur is rank deficient
    if r < size_H_2
        %Permute the columns of H_A and the entries of z
        H_A(:, firstCol:firstCol+size_H_2-1 -r ) = H_P(:, r+1:size_H_2);
        H_A(:, firstCol+size_H_2-r: lastCol) = H_P(:, 1:r);
        z(firstCol:firstCol+size_H_2-1-r) = z_p(r+1:size_H_2);
        z(firstCol+size_H_2-r: lastCol) = z_p(1:r);      
        
        %Update the permutation matrix P_tmp
        I_K = eye(size_H_2);
        Piv = eye(size_H_2);
        Piv(:, size_H_2-r+1:size_H_2) = I_K(:, 1:r);
        Piv(:, 1:size_H_2-r) = I_K(:, r+1:size_H_2);
        P_tmp(firstCol:lastCol, firstCol:lastCol) = P_qr * Piv; 
    else
        %Permute the columns of H_A and the entries of z
        H_A(:, firstCol:lastCol) = H_P;
        z(firstCol:lastCol) = z_p;
        
        %Update the permutation matrix P_tmp
        P_tmp(firstCol:lastCol, firstCol:lastCol) = P_qr;
    end
    P_cum = P_cum * P_tmp;
            
    firstCol = lastCol - r + 1;
    R_tilde(:, firstCol:lastCol) = R_qr(:, 1:r);
    Q_tilde(:, firstCol:lastCol) = Q_qr(:, 1:r);
    
    i = i + 1;
    indicator(1, i) = firstCol;
    indicator(2, i) = lastCol;
    
    lastCol = lastCol - r;
end

%Remove the extra columns of the indicator
indicator = indicator(:, 1:i);

end