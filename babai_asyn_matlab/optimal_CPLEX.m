function [s_bar, non_convex] = optimal_CPLEX(H, y, N)

% s_bar = optimal_CPLEX(H, y, N) applies the optimal solution (in CPLEX)
% 
% Inputs:
%     H - K-by-N real matrix
%     y - K-dimensional real vector
%     N - integer scalar
%
% Outputs:
%     s_bar_cur - N-dimensional integer vector for the optimal solution
%     non_convex - boolean, indicating error message from CPLEX

e_vec = repelem(1, N)';
y_adj = y - H*e_vec;
H_adj = 2*H;
l = repelem(-1, N)';
u = repelem(0, N)';
int.ind = repelem('I', N);

problem.C = H_adj;
problem.d = y_adj;
problem.Aineq = [];
problem.bineq=[];
problem.lb=l;
problem.ub=u;
problem.ctype=int.ind;


try
    z = cplexlsqmilp(problem);
    s_bar = 2 * z + e_vec;
    non_convex=0;
catch
    s_bar=repelem(0, N)';
    non_convex=1;
end


end