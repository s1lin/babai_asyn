function [A, x_t, y, R0] = gen_olm_problem(k, m, n, SNR)
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
sigma = sqrt(((4^k-1))/(3*k*10^(SNR/10)));

Ar = normrnd(0, sqrt(1/2), m/2, n/2);
Ai = normrnd(0, sqrt(1/2), m/2, n/2);
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
vr = normrnd(0, sigma, m/2, 1);
vi = normrnd(0, sigma, m/2, 1);
v = [vr; vi];

%Get Upper triangular matrix
y = A * x_t + v;

R0 = 0;

end






