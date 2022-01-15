function [x_cur, v_norm_cur, stopping] = SCP_Block_Babai_3(x_cur, v_norm_cur, H, tolerance, max_Babai, ~, y, K, N, ~, k)

% [x_cur, v_norm_cur, stopping] = SCP_Block_Optimal(s_bar_IP, v_norm_IP, H, tolerance, max_Babai, max_Time, y, K, N)
% applies the SCP-Block Optimal method to obtain a sub-optimal solution
%
% Inputs:
%     s_bar_IP - N-dimensional real vector, initial point
%     v_norm_IP - real scalar, norm of residual vector corresponding to s_bar_IP
%     H - K-by-N real matrix
%     tolerance - real scalar, tolerance for norm of residual vector
%     max_Babai - integer scalar, maximum number of calls to block_opt.m
%     max_Time - real scalar, maximum computation time of algorithm
%     y - K-dimensional real vector
%     K - integer scalar
%     N - integer scalar
%
% Outputs:
%     x_cur - N-dimensional integer vector for the sub-optimal solution
%     v_norm_cur - real scalar for the norm of the residual vector
%     corresponding to x_cur
%     stopping - 1-by-3 boolean vector, indicates stopping criterion used

% Subfunctions: SCP_opt
stopping=zeros(1,4);
b_count = 0;

if v_norm_cur <= tolerance
    stopping(1)=1;
    return;
end

q = ceil(N/K);
indicator = zeros(2, q);
cur_end = N;
i = 1;
while cur_end > 0
    cur_1st = max(1, cur_end-K+1);
    indicator(1,i) = cur_1st;
    indicator(2,i) = cur_end;
    cur_end = cur_1st - 1;
    i = i + 1;
end
[H_cur, P, s_bar_cur, Q_tilde, R_tilde, indicator] = partition_H(H, x_cur, K, N);
I = eye(N);
Piv = eye(N);
permutation = 1:N;
H_P = H_cur;
x_tmp = s_bar_cur;
v_norm = v_norm_cur;
per = false;
%all_perms = perms(1:N);
%n_perms = size(all_perms, 1);
for i = 1:max_Babai 
    
    %Apply permutation strategy to update x_cur and v_norm_cur
    %[x_tmp, v_norm_temp] = block_opt(H_P, y, x_tmp, N, indicator);
    %Corresponds to Algorithm 12 (Block Optimal) in Report 10
    for j = 1:q
        %Get the QR factorization corresponding to block j
        R = R_tilde(:, indicator(1, j):indicator(2, j));
        Q = Q_tilde(:, indicator(1, j):indicator(2, j));
        
        %cur_1st refers to the column of H where the current block starts
        %cur_end refers to the column of H where the current block ends
        cur_1st = indicator(1, j);
        cur_end = indicator(2, j);
        t = cur_end - cur_1st + 1;
        
        % Compute y_bar in the psuedocode of the report
        if cur_end == N
            %H_P(:,1:cur_1st-1)
            y_bar = y - H_P(:,1:cur_1st-1) * x_tmp(1:cur_1st-1);
        elseif cur_1st == 1
            y_bar = y - H_P(:,cur_end+1:N) * x_tmp(cur_end+1:N);
        else
            y_bar = y - H_P(:,1:cur_1st-1) * x_tmp(1:cur_1st-1) - H_P(:, cur_end+1:N) * x_tmp(cur_end+1:N);
        end
        y_bar = Q' * y_bar;
        
        for h=t:-1:1
            if h==t
                s_temp=y_bar(h)/(R(h,h));
            else
                s_temp=(y_bar(h)- R(h,h+1:t)*x_tmp(h+cur_1st:cur_end))/(R(h,h));
            end
            x_tmp(h+cur_end-t)=round_int(s_temp, 0, 2^k-1);
        end

    end
    
    v_norm_cur = norm(y - H_P * x_tmp);
    
    if v_norm_cur < v_norm
        x_cur = x_tmp;
        %v_norm_cur = v_norm_temp;
        if per         
            P = P * I(:, permutation) * Piv;
            per = false; 
            H_cur = H_P;
        end        
        if v_norm_cur <= tolerance
            stopping(2)=1;
            break;
        end        
       
        v_norm = v_norm_cur;
    end
    
    if stopping(2) ~= 1
        %permutation = all_perms(i,:);
        %permutation = [1 3 4 5 6 2]%
        permutation = randperm(N);
        H_P = H_cur(:,permutation);
        x_tmp = x_cur(permutation);
        [H_P, Piv, x_tmp, Q_tilde, R_tilde, indicator] = partition_H(H_P, x_tmp, K, N);
        per = true;
    end
    
    if b_count >= max_Babai
        stopping(3)=1;
        break;
    end
    b_count = b_count + 1;
    %If we don't decrease the residual, keep trying permutations
   
end
% if i == n_perms
%     stopping(4) = 1;
% end
x_cur = P * x_cur;
end