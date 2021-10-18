function [error_bits_rate, times_ave, stopping_SCP_ave, stopping_BCP_ave] =  simulations_Block_Babai(K, N, loop_times, max_Babai, max_Time, Htype, condNum, threshold)

% [error_bits_rate, times_ave, stopping_SCP_ave, stopping_BCP_ave] =  simulations_Block_Babai(K, N, loop_times, max_Babai, max_Time, Htype, condNum, threshold)
% applies the QRP initial point method, the Block Babai methods, and the optimal method. The average BER and
% computation times are given, as well as the stopping criteria used for the Block Babai methods
% 
% Inputs:
%     K - integer scalar
%     N - integer scalar
%     loop_times - integer scalar, number of iterations for each SNR level
%     max_Babai - integer scalar, maximum number of calls to Babai_Gen.m
%     for Block Babai methods
%     max_Time - real scalar, maximum computation time of sub-optimal
%     algorithms
%     Htype - string, either 'random' or 'ill'
%     condNum - real scalar
%     threshold - boolean, indicates whether sub-optimal algorithms will
%     terminate when the norm of the residual vector is sufficiently small
%
% Outputs:
%     error_bits_rate - 5-by-4 real matrix, gives average BER. Columns 
%     correspond to method, rows correspond to SNR level
%     times_ave - 5-by-4 real matrix, gives average computation time. Columns 
%     correspond to method, rows correspond to SNR level
%     stopping_SCP_ave - 5-by-3 real matrix, gives frequency of stopping 
%     criteria used for SCP-Block Babai method. Columns correspond to 
%     stopping criteria, rows correspond to SNR level
%     stopping_BCP_ave - 5-by-4 real matrix, gives frequency of stopping 
%     criteria used for BCP-Block Babai method. Columns correspond to 
%     stopping criteria, rows correspond to SNR level


% SNR_loop  signal-to-noise ratio to be used to generate different noise vector v
SNR_loop=[5:5:25];

error_bits_rate=zeros(length(SNR_loop),4);
times_ave = zeros(length(SNR_loop), 4);
stopping_SCP_ave = zeros(length(SNR_loop),4);
stopping_BCP_ave = zeros(length(SNR_loop),4);

for kk=1:length(SNR_loop)
    SNR=SNR_loop(kk);
    
    error_bits=zeros(loop_times,5);
    times=zeros(loop_times,5);
    stopping_SCP=zeros(loop_times,4);
    stopping_BCP=zeros(loop_times,4);
    
    for mm=1:loop_times
        
        %Generate system
        [H, s, v, y, sigma_squared] = getSystem(SNR, Htype, K, N, condNum);
        
        %Determine threshold
        if threshold
            tolerance = sqrt(K*sigma_squared);
        else
            tolerance = 0;
        end
        
        % Initial Point Method: QRP
        tic;
        [s_bar_IP, v_norm1, HH, Piv] = QRP_IP(H, y, 1);
        times(mm,1)=toc;
        s_bar1 = Piv*s_bar_IP;
        error_bits(mm,1)=length(nonzeros(s-s_bar1));
                
        
        % SCP-Block Babai
        tic;
        [s_bar_cur, v_norm_cur, stopping] = SCP_Block_Babai(s_bar_IP, v_norm1, HH, tolerance, max_Babai, max_Time, y, K, N);
        s_bar2 = Piv*s_bar_cur;
        times(mm,2)=toc + times(mm,1);
        stopping_SCP(mm, 1:3) = stopping;
        error_bits(mm,2)=length(nonzeros(s-s_bar2));
                
        
        %BCP-Block Babai
        tic;
        [s_bar_cur, v_norm_cur, stopping] = BCP_Block_Babai(s_bar_IP, v_norm1, HH, tolerance, max_Babai, max_Time, y, K, N);
        s_bar3 = Piv*s_bar_cur;
        times(mm,3)=toc + times(mm,1);
        stopping_BCP(mm, 1:4) = stopping;
        error_bits(mm,3)=length(nonzeros(s-s_bar3));
        
        
        %Optimal method
        tic;
        s_bar4 = optimal(H, y, N);
        times(mm,4)=toc;
        error_bits(mm,4)=length(nonzeros(s-s_bar4));

    end
    
    for i=1:4
        error_bits_rate(kk, i)=sum(error_bits(:, i))/(loop_times*N);
    end
    
    for i = 1:4
        times_ave(kk, i) = sum(times(:, i))/(loop_times);
    end
    
    for i = 1:4
        stopping_SCP_ave(kk, i) = sum(stopping_SCP(:, i))/(loop_times);
        stopping_BCP_ave(kk, i) = sum(stopping_BCP(:, i))/(loop_times);
    end
end