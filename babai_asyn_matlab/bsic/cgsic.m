function [x_hat, A_hat, P] = cgsic(A, y, lower, upper)

%Corresponds to Algorithm 2 (SIC) in Report 10

% [x_hat, rho, A_hat, P] = SIC_IP(A, y, n, bound) applies the SIC
% initial point method
% 
% Inputs:
%     A - m-by-n real matrix 
%     y - m-dimensional real vector
%     n - integer scalar
%     bound - integer scalar for the constraint
% 
% Outputs:
%     x_hat - n-dimensional integer vector for the initial point
%     rho - real scalar for the norm of the residual vector corresponding to x_hat
%     A_hat - m-by-n real matrix, permuted A for sub-optimal methods
%     P - n-by-n real matrix where A_hat*P'=A

[~, n] = size(A);
x_hat = zeros(n,1);
P = eye(n);
b= A'*y;
C = A'*A;
k = 0;
rho = norm(y)^2;
for i = n:-1:1
    rho_min = inf;
    for j = 1:i
        if i ~= n
            b(j) = b(j) - C(j,i+1)*x_hat(i+1);
        end
        xi = round(b(j)/C(j,j));
        xi = min(upper, max(xi, lower));
        rho_t = rho - 2 * b(j) * xi + C(j,j) * xi^2;
        if rho_t < rho_min
            k = j;
            x_hat(i)=xi;
            rho_min = rho_t;
        end
    end
    rho = rho_min;
    A(:,[k, i]) = A(:, [i,k]);
    P(:,[k, i]) = P(:, [i,k]);
    b([k,i]) = b([i,k]);
    C(:,[k, i]) = C(:, [i,k]);
    C([k, i], :) = C([i,k], :);
end
A_hat = A;
x_hat = P * x_hat;