function [s_bar_cur, v_norm_cur, stopping, depth] = SIC_subopt(s_bar_cur, v_cur, v_norm_cur, H_cur, tolerance, N, SIC_method)

%Corresponds to Algorithm 9 (Outline of SIC-Based Sub-Optimal Methods) in Report 10

% [s_bar_cur, v_norm_cur, stopping, depth] = SIC_subopt(s_bar_cur, v_cur, v_norm_cur, H_cur, tolerance, N, SIC_method)
% applies the SIC-based sub-optimal methods
% 
% Inputs:
%     s_bar_cur - N-dimensional real vector, initial point
%     v_cur - K-dimensional real vector, residual vector corresponding to
%     the initial point
%     v_norm_cur - real scalar, norm v_cur
%     H_cur - K-by-N real matrix
%     tolerance - real scalar
%     N - integer scalar
%     SIC_method - integer scalar (1 or 2)
%
% Outputs:
%     s_bar_cur - N-dimensional integer vector for the sub-optimal solution
%     v_norm_cur - real scalar for the norm of the residual vector
%     corresponding to s_bar_cur
%     stopping - 1-by-3 boolean vector, indicates stopping criterion used
%     depth - integer scalar, indicates number of iterations in while loop

% Subfunctions: SIC1_update, SIC2_update



stopping=zeros(1,3);
depth = 0;
if v_norm_cur <= tolerance
    stopping(1)=1;
    return;
end
v_cur
v_norm_temp = 0;
s_bar_temp = 0;
v_temp = 0;
while v_norm_cur > tolerance
    if SIC_method == 1
        [s_bar_temp, v_norm_temp, v_temp] = SIC1_update(H_cur, v_cur, N, 1, s_bar_cur);
        v_norm_temp
    end
    if SIC_method == 2
        [s_bar_temp, v_norm_temp, v_temp] = SIC2_update(H_cur, v_cur, N, 1, s_bar_cur);
    end
    depth = depth+1;
    
    if v_norm_temp < 0.99999 * v_norm_cur
        s_bar_cur = s_bar_temp;
        v_cur = v_temp;
        v_norm_cur = v_norm_temp;
    else
        stopping(3) = 1;
        return;
    end
    
    if v_norm_cur < tolerance
        stopping(2) = 1;
        return;
    end
end

end







function [s_bar, v_norm, v] = SIC1_update(H, v, N, bound, s_bar)

%Corresponds to Algorithm 10 (SIC1) in Report 10

% [s_bar, v_norm, v] = SIC1_update(H, v, N, bound, s_bar) applies the 
% SIC1 sub-optimal method
% 
% Inputs:
%     H - K-by-N real matrix
%     v_cur - K-dimensional real vector, residual vector corresponding to s_bar
%     N - integer scalar
%     bound - integer scalar for the constraint
%     s_bar_cur - N-dimensional real vector, initial point
%
% Outputs:
%     s_bar - N-dimensional integer vector for the sub-optimal solution
%     v_norm - real scalar for the norm of the residual vector
%     v - K-dimensional residual vector

for j=N:-1:1
    v = v+s_bar(j)*H(:,j); %Removes the \hat{x}_j from the residual
    s_temp_unrounded = H(:,j)'*v / (H(:,j)' * H(:,j));
    s_bar_temp = round_int(s_temp_unrounded, -bound, bound);
    
    v =v- s_bar_temp *H(:,j); %Updates the term for \hat{x}_j in the residual
    s_bar(j)=s_bar_temp; 
end
v_norm = norm(v);
end




function [s_bar, v_norm, v] = SIC2_update(H, v, N, bound, s_bar)

%Corresponds to Algorithm 11 (SIC2) in Report 10

% [s_bar, v_norm, v] = SIC1_update(H, v, N, bound, s_bar) applies the 
% SIC1 sub-optimal method
% 
% Inputs:
%     H - K-by-N real matrix
%     v_cur - K-dimensional real vector, residual vector corresponding to s_bar
%     N - integer scalar
%     bound - integer scalar for the constraint
%     s_bar_cur - N-dimensional real vector, initial point
%
% Outputs:
%     s_bar - N-dimensional integer vector for the sub-optimal solution
%     v_norm - real scalar for the norm of the residual vector
%     v - K-dimensional residual vector
k = 0;
s_est = 0;
v_best = -inf;
max_res = inf;
for j=N:-1:1
    v_temp = v+s_bar(j)*H(:,j); %Removes the \hat{x}_j
    s_temp_unrounded = H(:,j)'*v_temp / (H(:,j)' * H(:,j));
    s_bar_temp = round_int(s_temp_unrounded, -bound, bound);
    
    v_temp =v_temp- s_bar_temp *H(:,j);
    res = norm(v_temp);
    if res < max_res
        k=j;
        s_est=s_bar_temp;
        v_best = v_temp;
        max_res = res;
    end
end
s_bar(k) = s_est;
v = v_best;
v_norm = max_res;
end