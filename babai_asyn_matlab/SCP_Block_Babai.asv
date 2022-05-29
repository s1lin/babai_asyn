function [s_bar_cur, v_norm_cur, stopping] =  SCP_Block_Babai(s_bar_IP, v_norm_IP, H_cur, tolerance, max_Babai, max_Time, y, K, N)

% [s_bar_cur, v_norm_cur, stopping] =  SCP_Block_Babai(s_bar_IP, v_norm_IP, H_cur, tolerance, max_Babai, max_Time, y, K, N)
% applies the SCP-Block Babai method to obtain a sub-optimal solution
% 
% Inputs:
%     s_bar_IP - N-dimensional real vector, initial point
%     v_norm_IP - real scalar, norm of residual vector corresponding to s_bar_IP
%     H_cur - K-by-N real matrix
%     tolerance - real scalar, tolerance for norm of residual vector
%     max_Babai - integer scalar, maximum number of calls to Babai_Gen.m
%     max_Time - real scalar, maximum computation time of algorithm
%     y - K-dimensional real vector
%     K - integer scalar
%     N - integer scalar
%
% Outputs:
%     s_bar_cur - N-dimensional integer vector for the sub-optimal solution
%     v_norm_cur - real scalar for the norm of the residual vector corresponding to s_bar_cur
%     stopping - 1-by-3 boolean vector, indicates stopping criterion used

% Subfunctions: SCP


timerVal = tic;

stopping=zeros(1,3);
b_count=0;

if v_norm_IP <= tolerance
    stopping(1)=1;
    s_bar_cur=s_bar_IP;
    v_norm_cur=v_norm_IP;
    return;
end

v_norm_cur=v_norm_IP;
[H_cur, P, s_bar_cur, Q_tilde, R_tilde, indicator] = partition_H(H_cur, s_bar_IP, K, N);
P_cum = P;

while b_count < max_Babai && v_norm_cur > tolerance && toc(timerVal) < max_Time
    
    %Apply permutation strategy to update s_bar_cur and v_norm_cur
    [s_bar_cur, v_norm_cur, H_cur, b_toadd, P_cum, Q_tilde, R_tilde, indicator] = SCP(H_cur, y, s_bar_cur, v_norm_cur, (max_Babai-b_count), P_cum, N, K, Q_tilde, R_tilde, indicator, max_Time-toc(timerVal));
    b_count = b_count + b_toadd;
    
    if v_norm_cur <= tolerance
        stopping(2)=1;
        break;
    end
    
    if b_count >= max_Babai
        stopping(3)=1;
        break;
    end
    
end
s_bar_cur = P_cum * s_bar_cur; %Permutes back to H_cur

end









function [s_bar_cur, v_norm_cur, H_cur, b_count, P_cum, Q_tilde, R_tilde, indicator] = SCP(H_prev, y, s_bar_prev, v_norm_prev, b_rem, P_cum, N, K, Q_tilde, R_tilde, indicator, max_Time)

%Corresponds to Algorithm 7 (Single Column Permutation) in Report 10

% [s_bar_cur, v_norm_cur, H_cur, b_count, P_cum, Q_tilde, R_tilde, indicator] = SCP(H_prev, y, s_bar_prev, v_norm_prev, b_rem, P_cum, N, K, Q_tilde, R_tilde, indicator, max_Time)
% applies the single column permutation strategy and fits the parameter vector 
% 
% Inputs:
%     H_prev - K-by-N real matrix
%     y - K-dimensional real vector
%     s_bar_prev - N-dimensional real vector
%     v_norm_prev - real scalar, norm residual vector corresponding to s_bar_prev
%     b_rem - integer scalar, remaining number of calls to Babai_Gen.m
%     P_cum - N-by-N real matrix, permutation where H_prev*P_cum'=H
%     N - integer scalar
%     K - integer scalar
%     Q_tilde - K-by-N real matrix
%     R_tilde - K-by-N real matrix
%     indicator - 2-by-q integer matrix
%     max_Time - real scalar, remaining computation time of algorithm

%
% Outputs:
%     s_bar_cur - N-dimensional integer vector for the sub-optimal solution
%     v_norm_cur - real scalar for the norm of the residual vector
%     corresponding to s_bar_cur
%     H_cur - K-by-N real matrix, (permuted) H_prev
%     b_count - integer scalar, number of calls to Babai_Gen.m
%     P_cum - N-by-N real matrix, permutation where H_cur*P_cum'=H
%     Q_tilde - K-by-N real matrix
%     R_tilde - K-by-N real matrix
%     indicator - 2-by-q integer matrix


timerVal = tic;

H_cur = H_prev;
s_bar_cur = s_bar_prev;
v_norm_cur = v_norm_prev;
b_count = 0;

%Try computing Babai point with current H_prev
[s_bar_temp, v_norm_temp] = Babai_Gen(H_prev, y, s_bar_prev, N, Q_tilde, R_tilde, indicator, 1);
b_count = b_count + 1;

if v_norm_temp < v_norm_prev
    s_bar_cur = s_bar_temp;
    v_norm_cur = v_norm_temp;
end

%If we don't decrease the residual, keep trying permutations
while b_count < b_rem && v_norm_prev <= v_norm_cur && toc(timerVal) < max_Time
    %Compute permutation matrix and compute permutations
    permutation = randperm(N);
    H_permuted = H_prev(:,permutation);
    s_bar_prev_permuted = s_bar_prev(permutation);

    %Calculate Babai point
    [H_temp, P, s_bar_output, Q_tilde, R_tilde, indicator] = partition_H(H_permuted, s_bar_prev_permuted, K, N);
    [s_bar_temp, v_norm_temp] = Babai_Gen(H_temp, y, s_bar_output, N, Q_tilde, R_tilde, indicator, 1);
    b_count = b_count + 1;
    
    if v_norm_temp < v_norm_prev
        s_bar_cur = s_bar_temp; 
        v_norm_cur = v_norm_temp;
        H_cur = H_temp;
        I_n = eye(N);
        P_inv = I_n(:, permutation);
        P_cum = P_cum * P_inv * P;
    end
end
end