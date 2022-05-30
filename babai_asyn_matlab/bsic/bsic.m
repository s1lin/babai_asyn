function [x_cur, rho] = bsic(x_cur, rho, A, tolerance, max_iter, y, k, permutation, optimal)

% [x_cur, rho, stopping] = SCP_Block_Optimal(x_cur, rho, A, tolerance, max_iter, max_Time, y, m, n)
% applies the SCP-Block Optimal method to obtain a sub-optimal solution
%
% Inputs:
%     x_cur - n-dimensional real vector, initial point
%     rho - real scalar, norm of residual vector corresponding to x_cur
%     A - m-by-n real matrix
%     tolerance - real scalar, tolerance for norm of residual vector
%     max_iter - integer scalar, maximum number of calls to block_opt.m
%     y - m-dimensional real vector
%
% Outputs:
%     x_cur - n-dimensional integer vector for the sub-optimal solution
%     rho - real scalar for the norm of the residual vector corresponding to x_cur

[m, n] = size(A);
I = eye(n);
v_norm = rho;
for i = 1:max_iter 
    %permutation = randperm(n);
    H_P = A(:,permutation(:, i));
    x_tmp = x_cur(permutation(:, i));
    %H_P = A(:,permutation);
    %x_tmp = x_cur(permutation);
    [H_t, Piv_cum, x_t, indicator] = part(H_P, x_tmp, m, n);
    y_hat = y - H_t * x_t;
    
    for j = 1:size(indicator, 2)
        
        cur_1st = indicator(1, j);
        cur_end = indicator(2, j);
        t = cur_end - cur_1st + 1;        
  
        H_adj = H_t(:, cur_1st:cur_end);
        y_hat = y_hat + H_adj * x_t(cur_1st:cur_end);
        
        l = repelem(0, t)';
        u = repelem(2^k-1, t)';       
        if optimal
            z = zeros(t, 0);%obils(H_adj, y_hat, l, u);
        else
            [~, R, y_bar,~ ,~ , p] = obils_reduction(H_adj,y_hat,l,u);
            z = zeros(t, 1);
            
            for h=t:-1:1
                if h==t
                    s_temp=y_bar(h)/(R(h,h));
                else
                    s_temp=(y_bar(h)- R(h,h+1:t)*z(h+1:t))/(R(h,h));
                end
                z(h) = max(min(round(s_temp),u(h)),l(h));
            end
            x = zeros(t, 1);
            for h = 1 : t
                x(p(h)) = z(h); 
            end
            z = x;
        end

        x_t(cur_1st:cur_end) = z;
        y_hat = y_hat - H_adj * z;

    end
    
    rho = norm(y - H_t * x_t);
    
    if rho < v_norm        
        P = I(:, permutation(:, i)) * Piv_cum; % 
        x_cur = P * x_t;        
        if rho <= tolerance
            break;
        end               
        v_norm = rho;
    end   
end
end