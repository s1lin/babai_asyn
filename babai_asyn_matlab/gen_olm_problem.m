function [A, x_t, y, R1] = gen_olm_problem(k, m, n, SNR, c)
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
    a = rand(1);
    b = rand(1);
    PHI = zeros(n/2,n/2);
    phi = zeros(n/2,n/2);
    for i = 1:n/2
        for j = 1:n/2
            phi(i, j) = a^abs(i-j);
            PHI(i, j) = b^abs(i-j);
        end
    end
    Ar = sqrtm(phi) * randn(n/2) * sqrtm(psi);
    Ai = sqrtm(phi) * randn(n/2) * sqrtm(psi);
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

%reduction
% [R1, Z, y1] = sils_reduction2(A, y);
% 
% upper = 2^k - 1;
% z_B1 = zeros(n, 1);
% z_B2 = zeros(n, 1);
% block=50;
% 
% ds = ones(m/block, 1) .* block;
% % 
% % for i = m/block-1:-1:1
% %     ds(i) = ds(i) + ds(i+1);
% % end
% % ds
% %BOB
% z_B1 = sils_block_search(R1, y1, z_B1, ds, 0, 2^k-1);
% 
% 
% %BNP
% for j = n:-1:1
%     z_B2(j) = (y1(j) - R1(j, j + 1:n) * z_B2(j + 1:n)) / R1(j, j);
%     z_B2(j) = round(z_B2(j));
% end
% 
% %z=Zz
% z_B1 = Z * z_B1;
% z_B2 = Z * z_B2;
% 
% for j = n:-1:1
%     z_B1(j) = max(min(z_B1(j), upper), 0);
%     z_B2(j) = max(min(z_B2(j), upper), 0);
% end
% % 
% res1 = norm(y - A*z_B1)
% res2 = norm(y - A*z_B2)
%y = y1;
%A = Z;
R1 = zeros(n);
end