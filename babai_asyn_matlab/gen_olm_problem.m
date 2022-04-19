function [A, x_t, y, R0] = gen_olm_problem(k, m, n, SNR, c)
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
% August 2021. Last revision: August 2021

 
rng('shuffle')
%Initialize Variables
sigma = sqrt((4^k-1)*n/(3*k*10^(SNR/10)));

if c==0
    A = randn(n, n);
elseif c==1
    Ar = randn(n/2);
    Ai = randn(n/2);    
else 
    phi = rand(n/2);
    PHI = rand(n/2);
    for i = 1:n/2
        for j = 1:n/2
            phi(i, j) = phi(i, j)^abs(i-j);
            PHI(i, j) = PHI(i, j)^abs(i-j);
        end
    end
    Ar = sqrt(phi) * randn(n/2) * sqrt(PHI);
    Ai = sqrt(phi) * randn(n/2) * sqrt(PHI);
end
if c >= 1
    Abar = [Ar -Ai; Ai, Ar];
    A = 2*Abar;
end

 %True parameter x:
low = -2^(k-1);
upp = 2^(k-1) - 1;
xr = 1 + 2 * randi([low upp], n/2, 1);
xi = 1 + 2 * randi([low upp], n/2, 1);
xbar = [xr; xi];
x_t = (2^k - 1 + xbar)./2;

%Noise vector v:
vr = normrnd(0, sigma, m/2, 1);
vi = normrnd(0, sigma, m/2, 1);
v = [vr; vi];

%Get Upper triangular matrix
y = A * x_t + v;

R0 = 0;
end






