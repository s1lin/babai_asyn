function [error_bits_rate, times_ave] =  simulations_SCPBB_Dif_IP(K, N, loop_times, max_Babai, max_Time, Htype, condNum, threshold)

% [error_bits_rate, times_ave] =  simulations_SCPBB_Dif_IP(K, N, loop_times, max_Babai, max_Time, Htype, condNum, threshold)
% applies the QRP initial point method, SIC initial point method, the SCP-Block Babai method (to both initial points), 
% and the optimal method. The average BER and computation times are given
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


% SNR_loop  signal-to-noise ratio to be used to generate different noise vector v
SNR_loop=[5:5:25];

error_bits_rate=zeros(length(SNR_loop),5);
times_ave = zeros(length(SNR_loop), 5);

for kk=1:length(SNR_loop)
    SNR=SNR_loop(kk);
    error_bits=zeros(loop_times,5);
    times=zeros(loop_times,5);
    
    for mm=1:loop_times
        
        %Generate system
        [H, s, v, y, sigma_squared] = getSystem(SNR, Htype, K, N, condNum);
        
        %Determine threshold
        if threshold
            tolerance = sqrt(K*sigma_squared);
        else
            tolerance = 0;
        end
        
        % Initial Point 1: QRP
        tic;
        [s_bar_IP_QRP, v_norm_QRP, HH_QRP, Piv_QRP] = QRP_IP(H, y, 1);
        times(mm,1)=toc;
        s_bar1 = Piv_QRP*s_bar_IP_QRP;
        error_bits(mm,1)=length(nonzeros(s-s_bar1));
        
        
        % Initial Point 2: SIC
        tic;
        [s_bar_IP_SIC, v_norm_SIC, HH_SIC, Piv_SIC] = SIC_IP(H, y, N, 1);
        s_bar2=Piv_SIC*s_bar_IP_SIC;
        times(mm,2) = toc;
        error_bits(mm,2)=length(nonzeros(s-s_bar2));
                
        
        % SCP-Block Babai with QRP IP
        tic;
        [s_bar_cur, v_norm_cur, stopping] = SCP_Block_Babai(s_bar_IP_QRP, v_norm_QRP, HH_QRP, tolerance, max_Babai, max_Time, y, K, N);
        s_bar3 = Piv_QRP*s_bar_cur;
        times(mm,3)=toc + times(mm,1);
        error_bits(mm,3)=length(nonzeros(s-s_bar3));
                
        
        % SCP-Block Babai with SIC IP
        tic;
        [s_bar_cur, v_norm_cur, stopping] = SCP_Block_Babai(s_bar_IP_SIC, v_norm_SIC, HH_SIC, tolerance, max_Babai, max_Time, y, K, N);
        s_bar4 = Piv_SIC*s_bar_cur;
        times(mm,4)=toc + times(mm,2);
        error_bits(mm,4)=length(nonzeros(s-s_bar4));
        
        
        %Optimal method
        tic;
        s_bar5 = optimal(H, y, N);
        times(mm,5)=toc;
        error_bits(mm,5)=length(nonzeros(s-s_bar5));

    end
    
    for i=1:5
        error_bits_rate(kk, i)=sum(error_bits(:, i))/(loop_times*N);
    end
    
    for i = 1:5
        times_ave(kk, i) = sum(times(:, i))/(loop_times);
    end
    
end