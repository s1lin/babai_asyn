function [Ahat, P, t] = partition_H_2(A, m, n)

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


Ahat = A;
p = n;
q = 1;
P = eye(n);
t = zeros(2, n);
k = 0;

while p >= 1
    f = max(1, p-m+1);
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
    
    d = size(A_l, 2);
    
    %The case where A_l is rank deficient
    if r < d
        %Permute the columns of Ahat and the entries of        
        %Ahat(:, f:f+d-1 -r ) = A_P(:, r+1:d);
        %Ahat(:, f+d-r: p) = A_P(:, 1:r);
        Ahat(:, f:p) = [A_P(:, r+1:d) A_P(:, 1:r)];
        
        %Update the permutation matrix P_l
        I_d = eye(d);
        P_d = [I_d(:, r+1:d), I_d(:, 1:r)];
        P_hat = P_hat * P_d;
    else
        %Permute the columns of Ahat and the entries of
        Ahat(:, f:p) = A_P;        
    end
    
    P_l(f:p, f:p) = P_hat;
    P = P * P_l;
    
    f = p - r + 1;    
    k = k + 1;
    t(1, k) = f;
    t(2, k) = p;
    
    p = p - r;
end

%Remove the extra columns of the t
t = t(:, 1:k);

end