function [s_bar_cur, s_bar1, s_bar2, y, H, HH, Piv] = simulations_SIC(K, N, loop_times, Htype, condNum, threshold)

% [error_bits_rate, times_ave, depths_ave, stopping_SIC1_ave, stopping_SIC2_ave] =  simulations_SIC(K, N, loop_times, Htype, condNum, threshold)
% applies the SIC initial point method, the SIC-based sub-optimal methods, and the optimal method. The average BER and
% computation times are given.
% 
% Inputs:
%     K - integer scalar
%     N - integer scalar
%     loop_times - integer scalar, number of iterations for each SNR level
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
%     depths_ave - 5-by-2 real matrix, gives average depth of the SIC-based 
%     sub-optimal methods (see SIC_subopt.m for details). Columns 
%     correspond to method, rows correspond to SNR level
%     stopping_SIC1_ave - 5-by-3 real matrix, gives frequency of stopping 
%     criteria used for SIC1 method. Columns correspond to 
%     stopping criteria, rows correspond to SNR level
%     stopping_SIC2_ave - 5-by-3 real matrix, gives frequency of stopping 
%     criteria used for SIC2 method. Columns correspond to 
%     stopping criteria, rows correspond to SNR level



% SNR_loop  signal-to-noise ratio to be used to generate different noise vector v
SNR_loop=[5:5:25];

% The rows correspond to each level of SNR. 
error_bits_rate=zeros(length(SNR_loop),4);
times_ave = zeros(length(SNR_loop), 4);
depths_ave = zeros(length(SNR_loop), 2);
stopping_SIC1_ave = zeros(length(SNR_loop), 3);
stopping_SIC2_ave = zeros(length(SNR_loop), 3);
SNR = 35;
mm = 1;
% for kk=1:length(SNR_loop)
%     SNR=SNR_loop(kk);
%     
%     % The rows correspond to the error for a given iteration. 
%     error_bits=zeros(loop_times,4);
%     times=zeros(loop_times,4);
%     depths=zeros(loop_times, 2);
%     stopping_SIC1 = zeros(loop_times, 3);
%     stopping_SIC2 = zeros(loop_times, 3);
%     
%     for mm=1:loop_times
        
        %% Generate system
        [H, s, v, y, sigma_squared] = getSystem(SNR, Htype, K, N, condNum);
        
        
        %Determine threshold
        if threshold
            tolerance = sqrt(K*sigma_squared);
        else
            tolerance = 0;
        end
        
        
        %% Initial Point Method: SIC
        tic;
        [s_bar_IP, v_norm1, HH, Piv] = SIC_IP(H, y, N, 1);
        times(mm,1)=toc;
        s_bar1 = Piv*s_bar_IP;
        error_bits(mm,1)=length(nonzeros(s-s_bar1));  
        v_IP = y - HH*s_bar_IP;
        

        %% SIC1 Method
%         tic;
%         [s_bar_cur, v_norm_cur, stopping, depth] = SIC_subopt(s_bar_IP, v_IP, v_norm1, HH, tolerance, N, 1)
%         s_bar2 = Piv*s_bar_cur;
%         times(mm,2)=toc + times(mm,1);
%         error_bits(mm,2)=length(nonzeros(s-s_bar2));
%         depths(mm,1) = depth;
%         stopping_SIC1(mm, 1:3) = stopping;
        
        
        %% SIC2 Method
        tic;
        [s_bar_cur, v_norm_cur, stopping, depth] = SIC_subopt(s_bar_IP, v_IP, v_norm1, HH, tolerance, N, 2)
        s_bar2 = Piv*s_bar_cur;
        times(mm,3)=toc + times(mm,1);
        error_bits(mm,3)=length(nonzeros(s-s_bar2));
        depths(mm,2) = depth;
        stopping_SIC2(mm, 1:3) = stopping;
        
        
        %% Optimal method
%         tic;
%         s_bar4 = optimal(H, y, N);
%         times(mm,4)=toc;
%         error_bits(mm,4)=length(nonzeros(s-s_bar4));
        
        
%     end
%     
%     for i=1:4
%         error_bits_rate(kk, i)=sum(error_bits(:, i))/(loop_times*N);
%     end
%     
%     for i = 1:4
%         times_ave(kk, i) = sum(times(:, i))/(loop_times);
%     end
%     
%     for i = 1:2
%         depths_ave(kk, i) = sum(depths(:, i))/(loop_times);
%     end
%     
%     
%     for i = 1:3
%         stopping_SIC1_ave(kk, i) = sum(stopping_SIC1(:, i))/(loop_times);
%         stopping_SIC2_ave(kk, i) = sum(stopping_SIC2(:, i))/(loop_times);
%     end
    
end