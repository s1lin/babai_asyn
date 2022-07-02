function [x_cur, rhos] = bsic(x_cur, rho, A, tolerance, max_iter, y, k, permutation, optimal)

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
q = ceil(n/m);
indicator = zeros(2, q);
cur_end = n;
i = 1;
while cur_end > 0
    cur_1st = max(1, cur_end-m+1);
    indicator(1,i) = cur_1st;
    indicator(2,i) = cur_end;
    cur_end = cur_1st - 1;
    i = i + 1;
end

I = eye(n);
Piv_cum = eye(n);
rhos = zeros(max_iter, 1);
v_norm = rho;
%permutation = 1:n;
for i = 1:max_iter 
    %permutation = randperm(n);
    H_P = A(:,permutation(:, i));
    x_tmp = x_cur(permutation(:, i));
    
    %H_P = A(:,permutation);
    %x_tmp = x_cur(permutation);
    [H_t, Piv_cum, indicator] = part(H_P, m);
    %H_t = H_P;
    x_t = Piv_cum' * x_tmp;
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
            z = obils_4_block_search(H_adj, y_hat, l, u);
        else
            %z = obils_2_block_search(H_adj, y_hat, l, u);
            [~, R, y_bar,~ ,~ , p] = obils_reduction(H_adj,y_hat,l,u);
            %R
            z = random_babai(R,y_bar,l,u,10);
%             z = zeros(t, 1);
%             for h=t:-1:1   
%                 if h==t                                                        
%                    s_temp=y_bar(h)/(R(h,h));                                    
%                 else                                                             
%                    s_temp=(y_bar(h)- R(h,h+1:t)*z(h+1:t))/(R(h,h));            
%                 end                                  
%                 z(h) = max(min(round(s_temp),u(h)),l(h));                       
%             end                     
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
    rhos(i) = rho;
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