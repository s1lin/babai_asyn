function [s_bar4, y, H, HH, Piv, s_bar1, s, tolerance] =  simulations_Block_Optimal(k, SNR, K, N, loop_times, max_Babai, threshold)

% [error_bits_rate, times_ave] =  simulations_Block_Optimal(K, N, loop_times, max_Babai, max_Time, Htype, condNum, threshold)
% applies the QRP initial point method, the Block Babai methods, and the Block Optimal methods. The average BER and
% computation times are given.
% 
% Inputs:
%     K - integer scalar
%     N - integer scalar
%     loop_times - integer scalar, number of iterations for each SNR level
%     max_Babai - integer scalar, maximum number of calls to Babai_Gen.m
%     for Block Babai methods and maximum number of calls to block_opt.m
%     for the Block Optimal methods
%     max_Time - real scalar, maximum computation time of sub-optimal
%     algorithms
%     Htype - string, either 'random' or 'ill'
%     condNum - real scalar
%     threshold - boolean, indicates whether sub-optimal algorithms will
%     terminate when the norm of the residual vector is sufficiently small
%
% Outputs:
%     error_bits_rate - 5-by-5 real matrix, gives average BER. Columns 
%     correspond to method, rows correspond to SNR level
%     times_ave - 5-by-5 real matrix, gives average computation time. Columns 
%     correspond to method, rows correspond to SNR level

% SNR_loop  signal-to-noise ratio to be used to generate different noise vector v
%SNR_loop=[5:5:25];

%error_bits_rate=zeros(length(SNR_loop),5);
%times_ave = zeros(length(SNR_loop), 5);

s_bar4 = zeros(N, 1);
s_bar1 = zeros(N, 1);
HH = zeros(K, N);
Piv = eye(N);
% for kk=1:length(SNR_loop)
    %kk = 1;
    %SNR=35;
    %loop_times = 1;
    %error_bits=zeros(loop_times,5);
    times=zeros(loop_times,10);
    %mm = 1;
    e1 = 0;
    e2 = 0;
    e3 = 0;
    e4 = 0;
    for mm=1:loop_times
        
        %Generate system
        [H, s, v, y, sigma, res, permutation] = gen_problem(k, K, N, SNR, max_Babai);%getSystem(SNR, Htype, K, N, condNum);
        %x_t = s'
        %Determine threshold
        if threshold
            tolerance = sqrt(K) * sigma;
        else
            tolerance = 0;
        end
        
        %init_res = norm(y - H * s)
        % Initial Point Method: QRP
        tic;
        [s_bar_IP, v_norm1, HH, Piv] = QRP_IP(H, y, 2^k-1);
        times(mm,1)=toc;
        s_bar1 = Piv*s_bar_IP;
        %error = norm(s-s_bar1)
        %error_bits_qrp =length(nonzeros(s-s_bar1))
                
        
        
        % SCP-Block Babai
%         tic;
%         [s_bar_cur, v_norm_cur, stopping] = SCP_Block_Babai(s_bar_IP, v_norm1, HH, tolerance, max_Babai, max_Time, y, K, N);
%         s_bar2 = Piv*s_bar_cur;
%         times(mm,2)=toc + times(mm,1);
%         error_bits(mm,2)=length(nonzeros(s-s_bar2));
                
        
        %BCP-Block Babai
%         tic;
%         [s_bar_cur, v_norm_cur, stopping] = BCP_Block_Babai(s_bar_IP, v_norm1, HH, tolerance, max_Babai, max_Time, y, K, N);
%         s_bar3 = Piv*s_bar_cur;
%         times(mm,3)=toc + times(mm,1);
%         error_bits(mm,3)=length(nonzeros(s-s_bar3));
%         

%         %SCP-Block Optimal
        tic;
        [s_bar_cur, v_norm_cur, stopping] = SCP_Block_Babai(s_bar_IP, v_norm1, HH, tolerance, max_Babai, 1e6, y, K, N);      
        %v_norm_cur
        %stopping
        s_bar3 = Piv*s_bar_cur;
        %times(mm,3)=toc + times(mm,1)
        %error = norm(s-s_bar4)
        %error_bits_sbo1 = length(nonzeros(s-s_bar4))
        %res = norm(y - H * s_bar4)
        %SCP-Block Optimal 2
        
        tic;
        %[s_bar_cur, v_norm_cur, stopping] = SCP_Block_Babai(s_bar_IP, v_norm1, HH, tolerance, max_Babai, 1e6, y, K, N);
        [s_bar_cur, v_norm_cur, stopping] = SCP_Block_Babai_2(s_bar_IP, v_norm1, HH, tolerance, max_Babai, 0, y, K, N, permutation, 3);
        %v_norm_cur
        %stopping
        s_bar4 = Piv*s_bar_cur;
        %times(mm,3)=toc + times(mm,1)
        %error = norm(s-s_bar4)
        %error_bits_sbo1 = length(nonzeros(s-s_bar4))
        %res = norm(y - H * s_bar4)
        %SCP-Block Optimal 2

        [s_bar_cur, v_norm_cur, stopping] = SCP_Block_Babai_3(s_bar_IP, v_norm1, HH, tolerance, max_Babai, 0, y, K, N, permutation, 3);
        %v_norm_cur
        %stopping
        s_bar6 = Piv*s_bar_cur;
        %times(mm,3)=toc + times(mm,1)
        %error = norm(s-s_bar4)
        %error_bits_sbo1 = length(nonzeros(s-s_bar4))
        %res = norm(y - H * s_bar4)
        %SCP-Block Optimal 2
        
        tic;
        [s_bar_cur, v_norm_cur, stopping] = SCP_Block_Optimal_2(s_bar_IP, v_norm1, HH, tolerance, max_Babai, 0, y, K, N, permutation, 3);
        %v_norm_cur
        %stopping
        s_bar5 = Piv*s_bar_cur;
        %times(mm,4)=toc + times(mm,1)
        %error_bits_sbo2 = length(nonzeros(s-s_bar5))
        %error = norm(s-s_bar5)
        %norm(s_bar4 - s_bar5)
        %res = norm(y - H * s_bar5)
        %display(['s:' s'])
        s = s'
        %s_bar1 = s_bar1'
        s_bar_BBabai1 = s_bar3'
        s_bar_BBabai2 = s_bar4'
        s_bar_BBabai3 = s_bar6'
        s_bar_Optimal = s_bar5'
        e1 = e1 + norm(s - s_bar_BBabai1)
        e2 = e2 + norm(s - s_bar_BBabai2)
        e3 = e3 + norm(s - s_bar_BBabai3)
        e4 = e4 + norm(s - s_bar_Optimal)
%       
%         %BCP-Block Optimal
%         tic;
%         [s_bar_cur, v_norm_cur, stopping] = BCP_Block_Optimal(s_bar_IP, v_norm1, HH, tolerance, max_Babai, max_Time, y, K, N);
%         s_bar6 = Piv*s_bar_cur;
%         times(mm,6)=toc + times(mm,1);
%         error_bits_bbo1 = length(nonzeros(s-s_bar6))
%         
%         %BCP-Block Optimal2
%         tic;
%         [s_bar_cur, v_norm_cur, stopping] = BCP_Block_Optimal_2(s_bar_IP, v_norm1, HH, tolerance, max_Babai, y, K, N);
%         v_norm_cur
%         stopping
%         s_bar7 = Piv*s_bar_cur;
%         times(mm,7) = toc + times(mm,1)
%         error_bits_bbo2 = length(nonzeros(s-s_bar7))
%         norm(s_bar6 - s_bar7)
    %end
    
%     for i=1:5
%         error_bits_rate(kk, i)=sum(error_bits(:, i))/(loop_times*N);
%     end
%     
%     for i = 1:5
%         times_ave(kk, i) = sum(times(:, i))/(loop_times);
%     end
    end
    e1/100
    e2/100
    e3/100
    e4/100
end