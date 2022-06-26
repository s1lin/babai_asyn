function [A_bar, P, d, indicator] = part2(A, s)

%Corresponds to Algorithm 6.4 (Block Partition) in Thesis
% [A_bar, P, d] = part2(A, s)
% permutes and partitions A so that A_bar submatrices A_bar_i are full-column rank 
% 
% Inputs:
%     A - K-by-N real matrix
%     s - Number of columns of each block
%
% Outputs:
%     A_bar - K-by-N real matrix,
%     P - N-by-N permutation such A*P = A_bar
%     d - 1-by-k integer matrix (indicates submatrices of A_hat)


[K, N] = size(A);
A_bar = zeros(K, N);
A_hat = A;
P = eye(N);
d = ones(N,1) * s;
k = 1;
f = 1;

while f <= N

    [~, R, P_bar] = qrmgs_max(A_hat);

    if size(R,2)>1
        r = sum(abs(diag(R)) > 10^(-6));
    else
        r = sum(abs(R(1,1)) > 10^(-6));
    end 
    A_hat = A_hat * P_bar;
    [~, n] = size(P_bar);

    %TA_hate case wA_hatere A_hat_p is rank deficient
    if r < s
        %Permute tA_hate columns of A_hat and tA_hate entries of s_bar_output
        A_bar(:,f:f+r-1) = A_hat(:,1:r);        
        A_hat = A_hat(:, r+1:n);
        
        d(k) = r;
        f = f + r;
        k = k + 1;       
    else
        j = 0;
        while r >= s 
            A_bar(:,f:f+s-1) = A_hat(:,1+j*s:(j+1)*s);
            d(k) = s;
            f = f + s;
            r = r - s;
            k = k + 1;
            j = j + 1;
        end
        A_hat = A_hat(:,j*s+1:n);
    end
    [~, p] = size(P_bar);
    P = P * [eye(N-p), zeros(N-p,p); zeros(p, N-p), P_bar];   
end
q = k - 1;

d = d(1:q);
indicator = zeros(2, q);
cur_end = 1;
for i = 1:q
    indicator(1,i) = cur_end;
    indicator(2,i) = d(i) + cur_end-1;
    cur_end = cur_end + d(i);
end

end