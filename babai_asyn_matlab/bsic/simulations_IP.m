function [x_hat] = simulations_IP(k, m, n, SNR, c)

% [error_bits_rate, times_ave] =  simulations_IP(m, n, loop_times, Htype, condnum) 
% applies all the initial point methods, evaluating their average BER and
% computation time
% 
% Inputs:
%     k - 4,16,64 QAM
%     m - integer scalar
%     n - integer scalar
%     SNR - SNR level
%     c - case for generated matrix
%
% Outputs:
%     error_bits_rate - 5-by-4 real matrix, gives average BER. Columns 
%     correspond to method, rows correspond to SNR level
%     times_ave - 5-by-4 real matrix, gives average computation time. Columns 
%     correspond to method, rows correspond to SNR level

%Generate system
[A, x_t, y, ~] = gen_ublm_problem(k, m, n, SNR, c);
lower = 0;
upper = 2^k - 1;

%% Initial Point Method 1: SIC
[x_hat, rho, A_hat1, P] = SIC_IP(A, y, n, lower, upper);
x_hat = P*x_hat;
rho
x_t'
x_hat'



%% Initial Point Method 2: QR
[x_hat, rho, A_hat2, P] = cgsic(A, y, lower, upper);
rho
x_hat'


%% Initial Point Method 3: GP
% tic;
% l = repelem(-1, length(s))';
% u = repelem(1, length(s))';
% x = zeros(length(s), 1);
% s_bar4_unrounded = gradproj(A, y, l, u, x, 100);
% s_bar4 = round_int(s_bar4_unrounded, -1, 1); 
% times = toc
% error_bits = length(nonzeros(s-s_bar4))



