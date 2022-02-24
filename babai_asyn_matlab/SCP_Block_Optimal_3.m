function [x_cur, v_norm_cur, stopping] = SCP_Block_Optimal_3(x_cur, v_norm_cur, H, tol, search_iter, y, K, N, permutation, k)

% [x_cur, v_norm_cur, stopping] = SCP_Block_Optimal_3(x_cur, v_norm_cur, H, tol, search_iter, y, K, N, permutation, k)
% applies the SCP-Block Optimal method to obtain a sub-optimal solution
%
% Inputs:
%     x_cur - N-dimensional real vector, initial point
%     v_norm_cur - real scalar, norm of residual vector corresponding to x_cur
%     HW - K-by-N real matrix
%     tol - real scalar, tol for norm of residual vector
%     search_iter - integer scalar, maximum number of calls to block_opt.m
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
stopping = zeros(1, 2);

if v_norm_cur <= tol
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

I = eye(N);
v_norm = v_norm_cur;
best_per = -1;
x_per = x_cur;
P_par = I;

for i = 1:search_iter

    H_P = H(:,permutation(:, i));
    x_tmp = x_per(permutation(:, i));
    
    [H_P, P_hat, indicator] = partition_H_2(H_P, K, N);
    x_tmp = P_hat * x_tmp;
    
    per = i;

    %Optimal solution for each block
    for j = 1:size(indicator, 2)
        
        %cur_1st refers to the column of H where the current block starts
        cur_1st = indicator(1, j);
        %cur_end refers to the column of H where the current block ends
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
        
        % Compute optimal solution
        H_adj = H_P(:, cur_1st:cur_end);
        l = repelem(0, t)';
        u = repelem(2^k-1, t)';
        
        %todo: cils_search:
        
        z = obils(H_adj, y_bar, l, u);
        %x_tmp(cur_1st:cur_end) = 2 * z + e_vec;
        x_tmp(cur_1st:cur_end) = z;
        %z
    end
    
    v_norm_cur = norm(y - H_P * x_tmp);
    
    if v_norm_cur < v_norm
        x_per = x_tmp;
        best_per = per;
        P_par = P_hat;
        
        if v_norm_cur <= tol
            stopping(2) = 1;
            break;
        end
        
        v_norm = v_norm_cur;
    end    
end

if best_per ~= -1
    %x_per'
    x_cur = I(:, permutation(:, best_per)) * P_par * x_per;
end

end