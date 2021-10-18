function [s_bar_cur, v_norm_cur, stopping] =  BCP_Block_Babai(s_bar_IP, v_norm_IP, H_cur, tolerance, max_Babai, max_Time, y, K, N)

% [s_bar_cur, v_norm_cur, stopping] =  BCP_Block_Babai(s_bar_IP, v_norm_IP, H_cur, tolerance, max_Babai, max_Time, y, K, N)
% applies the BCP-Block Babai method to obtain a sub-optimal solution
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
%     v_norm_cur - real scalar for the norm of the residual vector
%     corresponding to s_bar_cur
%     stopping - 1-by-4 boolean vector, indicates stopping criterion used

% Subfunctions: BCP


timerVal = tic;

stopping=zeros(1,4);
b_count=0;

if v_norm_IP <= tolerance
    stopping(1)=1;
    s_bar_cur=s_bar_IP;
    v_norm_cur=v_norm_IP;
    return;
end

v_norm_cur=v_norm_IP;
[H_cur, P, s_bar_cur, Q_tilde, R_tilde, indicator] = partition_H(H_cur, s_bar_IP, K, N);

while b_count < max_Babai && v_norm_cur > tolerance && toc(timerVal) < max_Time
    
    %Apply permutation strategy to update s_bar_cur and v_norm_cur
    [s_bar_cur, v_norm_cur, b_toadd, indicator, term_boolean] = BCP(H_cur, y, s_bar_cur, v_norm_cur, (max_Babai-b_count), N, Q_tilde, R_tilde, indicator, max_Time-toc(timerVal));
    b_count = b_count + b_toadd;
    
    if v_norm_cur <= tolerance
        stopping(2)=1;
        break;
    end
    
    if b_count >= max_Babai
        stopping(3)=1;
        break;
    end
    
    if term_boolean
        stopping(4)=1;
        break;
    end
end
s_bar_cur = P * s_bar_cur; %Permutes back to H_cur

end





function [s_bar_cur, v_norm_cur, b_count, indicator, term] = BCP(H_cur, y, s_bar_prev, v_norm_prev, b_rem, N, Q_tilde, R_tilde, indicator, max_Time)

%Corresponds to Algorithm 8 (Block Column Permutation) in Report 10

% [s_bar_cur, v_norm_cur, b_count, indicator, term] = BCP(H_cur, y, s_bar_prev, v_norm_prev, b_rem, N, Q_tilde, R_tilde, indicator, max_Time)
% applies the block column permutation strategy and fits the parameter vector 
% 
% Inputs:
%     H_cur - K-by-N real matrix
%     y - K-dimensional real vector
%     s_bar_prev - N-dimensional real vector
%     v_norm_prev - real scalar, norm residual vector corresponding to s_bar_prev
%     b_rem - integer scalar, remaining number of calls to Babai_Gen.m
%     N - integer scalar
%     Q_tilde - K-by-N real matrix (Q factors)
%     R_tilde - K-by-N real matrix (R factors)
%     indicator - 2-by-q integer matrix (indicates submatrices of H)
%     max_Time - real scalar, remaining maximum computation time of algorithm

%
% Outputs:
%     s_bar_cur - N-dimensional integer vector for the sub-optimal solution
%     v_norm_cur - real scalar for the norm of the residual vector
%     corresponding to s_bar_cur
%     H_cur - K-by-N real matrix, (permuted) H_prev
%     b_count - integer scalar, number of calls to Babai_Gen.m
%     indicator - 2-by-q integer matrix (indicates submatrices of H)
%     term - boolean, indicating whether all block permutations do not
%     reduce norm of residual vector


timerVal = tic;

s_bar_cur = s_bar_prev;
v_norm_cur = v_norm_prev;
b_count = 0;
n_blocks = size(indicator,2);
term=false;

%Try computing Babai point 
[s_bar_temp, v_norm_temp] = Babai_Gen(H_cur, y, s_bar_prev, N, Q_tilde, R_tilde, indicator, 1);
b_count = b_count + 1;

if v_norm_temp < v_norm_prev
    s_bar_cur = s_bar_temp;
    v_norm_cur = v_norm_temp;
    return;
end

%If we didn't decrease the residual, keep trying permutations
all_perms = perms(1:n_blocks);
n_perms = size(all_perms, 1);

for i=1:n_perms
    if b_count >= b_rem || toc(timerVal) >= max_Time
        return;
    end
    
    permutation = all_perms(i,:);
    indicator_permuted = indicator(:, permutation);

    %Calculate Babai point
    [s_bar_temp, v_norm_temp] = Babai_Gen(H_cur, y, s_bar_prev, N, Q_tilde, R_tilde, indicator_permuted, 1);
    b_count = b_count + 1;
    
    if v_norm_temp < v_norm_prev
        s_bar_cur = s_bar_temp; 
        v_norm_cur = v_norm_temp;
        indicator=indicator_permuted;
        return;
    end
end

%If all possible permutations don't reduce the residual, terminate
%algorithm
term = true; 

end