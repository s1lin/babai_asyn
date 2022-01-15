function s_bar = optimal(H, y, N)

% s_bar = optimal(H, y, N) applies the optimal solution (in MATLAB)
% 
% Inputs:
%     H - K-by-N real matrix
%     y - K-dimensional real vector
%     N - integer scalar
%
% Outputs:
%     s_bar_cur - N-dimensional integer vector for the optimal solution


e_vec = repelem(1, N)';
y_adj = y - H*e_vec;
H_adj = 2*H;
l = repelem(-1, N)';
u = repelem(0, N)';
z = ubils(H_adj,y_adj,l,u);
s_bar = 2 * z + e_vec;

end