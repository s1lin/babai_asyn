function [Ahat, P, d] = partition_H_3(A)

%Corresponds to Algorithm 5 (Partition Strategy) in Report 10

% [Ahat, P, t] = partition_H(A, m, n)
% permutes and partitions Ahat so that the submatrices H_i are full-column rank
%
% Inputs:
%     A - m-by-n real matrix
%     m - integer scalar
%     n - integer scalar
%
% Outputs:
%     Ahat - m-by-n real matrix
%     P - n-by-n real matrix, permutation such that Ahat*P=A
%     t - 2-by-q integer matrix (indicates submatrices of Ahat)

[m, n] = size(A);
Ahat = A;
p = n;
f = 1;
P = eye(n);
d = zeros(1, n);
k = 0;

while f <= n
    p = min(f + m - 1, n);
    A_l = Ahat(:, f:p);
    P_l = eye(n);

    %Find the rank of A_l
    [~, R, P_hat] = qr(A_l);
    
    if size(R,2) > 1
        r = sum(abs(diag(R)) > 10^(-6));
    else
        r = sum(abs(R(1,1)) > 10^(-6));
    end
    A_P = A_l * P_hat;
    Ahat(:, f:p) = A_P;    
    
    P_l(f:p, f:p) = P_hat;
    P = P * P_l;
    
    f = f + r;   
    d(k) = r;
    k = k + 1;    
end
d = d(1:k);
end