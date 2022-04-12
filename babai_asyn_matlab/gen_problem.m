function [A, x_t, v, y, sigma, res, permutation, max_iter, R0] = gen_problem(k, m, n, SNR, max_iter)
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
%while 1
rng('shuffle')
%Initialize Variables
sigma = sqrt(((4^k-1)*n)/(6*10^(SNR/10)));

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

res = norm(y - A * x_t)
permutation = zeros(1,1);
R0=0;
%max_iter = 0;
% [R0,Z,yL] = aspl(A,y);
% upper = 2^k - 1;
% z_B = ones(n, 1) .* 4;
% tStart = tic;
% for j = n:-1:1
%     z_B(j) = (yL(j) - R0(j, j + 1:n) * z_B(j + 1:n)) / R0(j, j);
%     z_B(j) = round(z_B(j));
% end
% z_B = Z * z_B;
% for j = n:-1:1
%     z_B(j) = max(min(z_B(j), upper), 0);
% end
% tEnd = toc(tStart);         
% res_LLL = norm(y - A * z_B);
%         
% %sils_reduction(A, y);
% if m > n    
%     upper = 2^k - 1;
%     [R0,y0,l0,u0,p0] = obils_reduction(A,y,zeros(n,1), upper * ones(n,1));
%     
% end
if m <= n
    permutation = zeros(max_iter, n);
    %if factorial(n) > 1e7        
    for i = 1 : max_iter
        permutation(i,:) = randperm(n);
    end        
    %else
    %    permutation = perms(1:n);
    %   if max_iter > factorial(n)
    %       for i = factorial(n) : max_iter
    %           permutation(i,:) = randperm(n);
    %        end
    %   end        
    %end 
    
    permutation = permutation';    
    %permutation = permutation(: , randperm(max_iter));
    permutation(:,1) = (1:n)';
    
    if m <= 12
        val = 1;
        times=zeros(val,10);

        e1 = 0;
        %e2 = 0;
        for mm=1:val

            %Generate system
            %x_t = s'
            %Determine threshold
            tolerance = sqrt(m) * sigma;

            %init_res = norm(y - H * s)
            % Initial Point Method: QRP
            HH = A;
            Piv = eye(n);
            v_norm = 100;
            s_bar_IP = zeros(n, 1);
            %tic;
            [s_bar_IP, v_norm, HH, Piv] = SIC_IP(A, y, n, 2^k-1);            
            % gradproj(A, y, zeros(n, 1), ones(n, 1) * 2^k-1, s_bar_IP, 100);
            %s_bar_IP = round_int(s_bar_IP, 0, 2^k-1);
            %times(mm,1)=toc;
            s_sic = Piv * s_bar_IP;
            %R0 = partition_H_2(A, m, n);

            %tic;
            %[s_bar_cur, v_norm_cur, stopping] = SCP_Block_Babai_2(s_bar_IP, v_norm, HH, tolerance, factorial(n), 0, y, m, n, permutation', 3);
            %s_bar_babai = Piv*s_bar_cur;

            %tic;
            %max_iter
            [s_bar_cur, v_norm_cur, stopping] = SCP_Block_Optimal_3(s_bar_IP, v_norm, HH, tolerance, 3000, y, m, n, permutation, 3);
            s_bar_optim = Piv * s_bar_cur;


            %s = x_t'
            s_sic = s_sic'
            s_bar_optim = s_bar_optim'
            %e1 = e1 + norm(s - s_bar_babai);
            %e2 = e2 + norm(s - s_bar_optim);
            %[~, ~, ~, Q_tilde, R_tilde, ~] = partition_H(HH, s_sic, m, n)

        end
        %e1/100
        %e2/100
    end
    
end
%if abs(init_res - res) < tolerance
%    break
%end
[Q, R_, y_q] = qrmgs_row(A, y);
%end







