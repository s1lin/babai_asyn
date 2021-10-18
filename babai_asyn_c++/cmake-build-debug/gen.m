function [A, y, x_t] = gen(k, n, SNR)

rng('shuffle')
%Initialize Variables  
sigma = sqrt(((4^k-1)*n)/(6*10^(SNR/10)));            

Ar = normrnd(0, sqrt(1/2), n/2, n/2);
Ai = normrnd(0, sqrt(1/2), n/2, n/2);
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
vr = normrnd(0, sigma, n/2, 1);
vi = normrnd(0, sigma, n/2, 1);
v = [vr; vi];

%Get Upper triangular matrix
y = A * x_t + v;



