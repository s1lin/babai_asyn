function x_hat = gradproj(B,y,l,u,x,max_iter)
% Gradient projection method to find a real solutiont to the
% box-constrained real LS problem min_{l<=x<=u}||y-Bx||_2
% The input x is an initial point, we may take x=(l+u)/2.
% max_iter is the maximum number of iterations, say 50.

n = length(x);
x_hat = zeros(n, 1);
c = B'*y;
t = 0;
for iter = 1:max_iter
    
    g = B'*(B*x-y);
    
    % Check KKT conditions
    if (x==l) == 0
        k1 = 1;
    elseif (g(x==l) > -1.e-5) == 1
        k1 = 1;
    else
        k1 = 0;
    end
    if (x==u) == 0
        k2 = 1;
    elseif (g(x==u) < 1.e-5) == 1
        k2 = 1;
    else
        k2 = 0;
    end
    if (l<x & x<u) == 0
        k3 = 1;
    elseif (g(l<x & x<u) < 1.e-5) == 1
        k3 = 1;
    else
        k3 = 0;
    end
    if (k1 && k2 && k3)
        x_hat = x;
        break
    end
    
    % Find the Cauchy point
    t_bar = 1.e5*ones(n,1);
    t_bar(g<0) = (x(g<0)-u(g<0))./g(g<0);
    t_bar(g>0) = (x(g>0)-l(g>0))./g(g>0);
    
    % Generate the ordered and non-repeated sequence of t_bar
    %t_bar
    t_seq = unique([0;t_bar]);   % Add 0 to make the implementation easier
    
    % Search
    for j = 2:length(t_seq)
        tj_1 = t_seq(j-1);
        tj = t_seq(j);
        % Compute x(t_{j-1})
        xt_j_1 = x - min(tj_1,t_bar).*g;
        % Compute teh search direction p_{j-1}
        pj_1 = zeros(n,1);
        pj_1(tj_1<t_bar) = -g(tj_1<t_bar);
        % Compute coefficients
        q = B*pj_1;
        fj_1d = (B*xt_j_1)'*q - c'*pj_1;
        fj_1dd = q'*q;
        t = tj;
        % Find a local minimizer
        delta_t = -fj_1d/fj_1dd;
        if fj_1d >= 0
            t = tj_1;
            break;
        elseif delta_t < (tj-tj_1)
            t = tj_1+delta_t;
            break;
        end
    end
    
    x = x - min(t,t_bar).*g;
    
end