function [s_bar1_for_next, y, H, HH_SIC, Piv_SIC, s_bar1, l, u] =  simulations_IP(m, SNR, n, Htype, condnum)

% [error_bits_rate, times_ave] =  simulations_IP(K, n, loop_times, Htype, condnum) 
% applies all the initial point methods, evaluating their average BER and
% computation time
% 
% Inputs:
%     K - integer scalar
%     n - integer scalar
%     loop_times - integer scalar, number of iterations for each SNR level
%     Htype - string, either 'random' or 'ill'
%     condnum - real scalar
%
% Outputs:
%     error_bits_rate - 5-by-4 real matrix, gives average BER. Columns 
%     correspond to method, rows correspond to SNR level
%     times_ave - 5-by-4 real matrix, gives average computation time. Columns 
%     correspond to method, rows correspond to SNR level


% SNR_loop  signal-to-noise ratio to be used to generate different noise vector v

error_bits_rate=0;
times_ave = 0;

%Generate system
[H, s, v, y, ~] = getSystem(SNR, Htype, m, n, condnum);

%% Initial Point Method 1: SIC
tic;
[s_bar1_for_next, v_norm1, HH_SIC, Piv_SIC] = SIC_IP(H, y, n, 1);
s_bar1=Piv_SIC*s_bar1_for_next;
v_norm1
toc
error_bits=length(nonzeros(s-s_bar1))


%% Initial Point Method 2: QR
tic;
[s_bar_IP, v_norm, HH, Piv] = QRP_IP(H, y, 1);
s_bar2 = Piv*s_bar_IP;
v_norm
toc
error_bits=length(nonzeros(s-s_bar2))
 
%% Initial Point Method 3: SIC + QR
tic;
[s_bar_IP, v_norm, HH, Piv] = QRP_IP(HH_SIC, y, 1);
s_bar_IP = Piv*s_bar_IP;
v_norm
s_bar3 = Piv_SIC * s_bar_IP;
toc
error_bits=length(nonzeros(s-s_bar3))
% 
% 
%% Initial Point Method 4: GP
tic;
l = repelem(-1, length(s))';
u = repelem(1, length(s))';
x = zeros(length(s), 1);
s_bar4_unrounded = gradproj(H, y, l, u, x, 100);
s_bar4 = round_int(s_bar4_unrounded, -1, 1); 
times = toc
error_bits = length(nonzeros(s-s_bar4))



