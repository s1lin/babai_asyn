function [A, x_t, y, R0, permutation, size_perm] = gen_ublm_problem(k, m, n, SNR, c, max_iter)
% [A, y, v, x_t, sigma] = gen_problem(k, m, n, SNR)
% generates linear model y = A * x_t + v
%
% Inputs:
%     SNR - integer scalar
%     m   - integer scalar
%     n   - integer scalar
%     k   - integer scalar
%
% Outputs:
%     A   - m-by-n real matrix
%     x_t - n-dimensional real vector
%     v   - m-dimensional real vector
%     y   - m-dimensional real vector
%     sigma - real scalar
%
% Main References:
% [1] Z. Chen.Some Estimation Methods
% for Overdetermined Integer Linear Models. McGill theses.
% McGill UniversityLibraries, 2020.
%
% Authors: Shilei Lin
% Copyright (c) 2021. Scientific Computing Lab, McGill University.
% August 2022. Last revision: August 2022
 
rng('shuffle')
%Initialize Variables
sigma = sqrt((4^k-1)/(3*k*10^(SNR/10)));

if c==1
    Ar = randn(m/2, n/2);
    Ai = randn(m/2, n/2);
else 
    a = rand(1);
    b = rand(1);
    psi = zeros(m/2,n/2);
    phi = zeros(m/2,n/2);
    for i = 1:m/2
        for j = 1:n/2
            phi(i, j) = a^abs(i-j);
            psi(i, j) = b^abs(i-j);
        end
    end
    Ar = sqrtm(phi) * randn(n/2) * sqrtm(psi);
    Ai = sqrtm(phi) * randn(n/2) * sqrtm(psi);
end

Abar = [Ar -Ai; Ai, Ar];
A = 2 * Abar;

%True parameter x:
low = -2^(k-1);
upp = 2^(k-1) - 1;
xr = 1 + 2 * randi([low upp], n/2, 1);
xi = 1 + 2 * randi([low upp], n/2, 1);
xbar = [xr; xi];
x_t = (2^k - 1 + xbar)./2;

%Noise vector v:
v = sigma * randn(m, 1);

%Get Upper triangular matrix
y = A * x_t + v;

permutation = zeros(max_iter, n);
if factorial(n) > 1e7        
    for i = 1 : max_iter
        permutation(i,:) = randperm(n);
    end        
else
   permutation = perms(1:n);
   if max_iter > factorial(n)
      for i = factorial(n) : max_iter
          permutation(i,:) = randperm(n);
      end
  end        
end 

[size_perm, ~] = size(permutation);
permutation = permutation';    
permutation(:,1) = (1:n)';

% [x_hat, ~, ~] = cgsic(A, y, 0, 2^k-1);
% tic
% [s_bar_cur, v_norm_cur] = bsic(x_hat, inf, A, 0, max_iter-1, y, k, permutation, false);
% toc
% s_bar_cur'
% x_t'
% sum(s_bar_cur)

R0 = zeros(n);

end