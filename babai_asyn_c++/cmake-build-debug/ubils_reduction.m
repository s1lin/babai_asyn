function [R,y,l,u,p,z0,beta] = ubils_reduction(B,y,l,u)
%
% [R,y,l,u,p,beta] = ubils_reduction(B,y,l,u) reduces an underdetremined 
% box-constrained integer least squares problem or rank-deficient 
% overdetermiend box-constrained integer least squares problem
% to an upper trapezoid one by a reduction strategy involving 
% the QR factorization with column permutations Q'*B*P = [R; 0]. 
% The orthogonal matrix Q is not produced.  
%
% Inputs:
%    B - m by n real matrix with m<n
%    y - m-dimensional real vector
%    l - n-dimensional integer vector, lower bound 
%    u - n-dimensional integer vector, upper bound    
%
% Outputs:
%    R - r by n real upper trapezoidal matrix, Q'*B*P = [R; 0]  
%        Q: m by m column orthogonal, P: n by n permutation matrix 
%    y - r-dimensional real vector, y := (Q'*y)(1:r) 
%    l - n-dimensional integer vector, lower bound 
%    u - n-dimensional integer vector, upper bound
%    p - n-dimensional integer vector to store the information of P
%    z0 - an initial integer point which determins the seach radius
%    beta - initial search radius

% Subfunctions: radius, reorder (included in this file)
% 
% Main References: 
% [1] X.-W. Chang, and X. Yang. An efficient tree search
%     decoder with column reordering for underdetermined MIMO systems. 
%     Global Telecommunications Conference, 2007. 5 pages
% [2] S. Breen and X.-W. Chang. Column Reordering for 
%     Box-Constrained Integer Least Squares Problems, 
%     Proceedings of IEEE GLOBECOM 2011, 6 pages.

% Authors: Xiao-Wen Chang, www.cs.mcgill.ca/~chang 
%          Xiangyu Ren
% Copyright (c) 2015-2018. Scientific Computing Lab, McGill University.
% April 2015. Last revision: April 2018


[m,n] = size(B);

% Reduce the problem to a full-row-rank one
[Q,R,p] = qr(B,'vector');
r = sum(abs(diag(R)) > n*eps(abs(R(1,1)))); % Determine the rank of B
if r == n
    error('The problem is overdetermined, use obils rather than ubils!')
else
    R = R(1:r,:);
    y = Q' * y; y = y(1:r);
    l = l(p); u = u(p);
    m = r;
end

% Compute an initial search radius
[beta,x0] = radius(B,y,l,u);

index = 0;
prod = inf;
k = m;
R_temp = 0;
y_temp = 0;
piv_temp = 0;
% Determine column m of R
for j = 1:m
    R1 = R; 
    y1 = y; 
    l1 = l; 
    u1 = u;   
    p1 = p; 
        
    if j ~= m
    % Interchanger columns m and j of R1
       R1(:,[j,m]) = R1(:,[m,j]);
       l1([j,m]) = l1([m,j]);
       u1([j,m]) = u1([m,j]);
       p1([j,m]) = p1([m,j]);
    end
    
    % Transform R1 to an upper trapezoidal matrix by plane rotations
    % and apply same rotations to y1
    for i = m:-1:j+1
        [W,~] = planerot(R1(i-1:i,j));
        R1(i-1:i,:) = W * R1(i-1:i,:);
        y1(i-1:i) = W * y1(i-1:i,:);
    end
    
    for i = j+1:m-1
        [W,~] = planerot(R1(i:i+1,i));
        R1(i:i+1,:) = W * R1(i:i+1,:);
        y1(i:i+1) = W * y1(i:i+1);
    end
    
    if y1(m) > 0
        R1(m,:) = -R1(m,:);
        y1(m) = -y1(m);
    end
    
    % Column reordering for (part of) R(:,m:n)    
    [piv,ind_tmp,prod_tmp] = reorder(R1,y1(m),l1,u1,beta);
      
    if ind_tmp > index
        index = ind_tmp;
        k = j;
        R_temp = R1(:,piv);
        y_temp = y1;
        piv_temp = piv;
    elseif ind_tmp == index && prod_tmp < prod
        prod = prod_tmp;
        k = j;
        R_temp = R1(:,piv);
        y_temp = y1;
        piv_temp = piv; 
    end  
end

R = R_temp;
y = y_temp;
l([k,m]) = l([m,k]);  l= l(piv_temp);  
u([k,m]) = u([m,k]);  u = u(piv_temp);
p([k,m]) = p([m,k]);  p = p(piv_temp);

% Column reordering for R(:,m:index-1)
if index > m+1
    [~,id] = sort(abs((R(m,m:index-1))));
    R(:,m:index-1) = R(:,id+m-1);
    p(m:index-1) = p(id+m-1);
    l(m:index-1) = l(id+m-1);
    u(m:index-1) = u(id+m-1);
end
    
% Apply the reduction strategy to reorder the first m-1 columns of R
x = zeros(n,1);
y_m = y(m);
for i = n:-1:m
    x(i) = median([l(i),u(i),round(y_m/R(m,i))]);
    y_m = y_m - R(m,i)*x(i);
end
    
y1 = y(1:m-1) - R(1:m-1,m:n) * x(m:n);
R1 = R(1:m-1,1:m-1);
[~,~,l(1:m-1),u(1:m-1),piv]= obils_reduction(R1,y1,l(1:m-1),u(1:m-1));   
R(:,1:m-1) = R(:,piv);
[Q,R] = qr(R);
y = Q'*y;
p(1:m-1) = p(piv);

% Permute x0
z0 = x(p);

end


% ------------------------------------------------------------------
% --------  Subfunctions  ------------------------------------------
% ------------------------------------------------------------------

function [beta,x0] = radius(B,y,l,u)
%
% Find an initial search radius determined  by the rounded value of 
% a real solution of the least squares problem. 
%
% Inputs:
%    B - m by n real matrix
%    y - m-dimensional real vector
%    l - n-dimensional integer vector, lower bound 
%    u - n-dimensional integer vector, upper bound    
%
% Outputs:
%    beta - residual at x0 to be used as a asearch radius
%    x0 - the integer point determining beta

% Subfunction:  gradproj

% Compute two initial points
[Q,R] = qr(B',0);
x1 = median([l,round(Q*(R'\y)),u],2);
x2 = round((l+u)/2);

% Compute two box-constrained real LS solutions by the gradient projection
% method
x1_real = gradproj(x1,B,y,l,u,100);
x2_real = gradproj(x2,B,y,l,u,100);

% Round the box-constrained real solution to an integer vector
x1 = round(x1_real);
x2 = round(x2_real);

% Initial 
beta1 = norm(y-B*x1);
beta2 = norm(y-B*x2);

if beta1 <= beta2
    x0 = x1;
    beta = beta1;
else
    x0 = x2;
    beta = beta2;
end

end



function x = gradproj(x,B,y,l,u,max_iter)
%
% Find a solution to the box-constrained real least squares problem 
% min_{l<=x<=u}||y-Bx|| by the gradient projection method
%
% Inputs:
%    x - n-dimensional real vector as an initial point
%    B - m by n real matrix
%    y - m-dimensional real vector
%    l - n-dimensional integer vector, lower bound 
%    u - n-dimensional integer vector, upper bound    
%
% Output:
%    x - n-dimensional real vector, a solution

n = length(x);

c = B'*y;

for iter = 1:max_iter
    
    g = B'*(B*x-y);
    
    % Check KKT conditions
    if (x==l) == 0 
        k1 = 1;
    elseif (g(x==l) > -1.e-5) == 1
        k1 = 1;
    else k1 = 0;
    end
    if (x==u) == 0 
        k2 = 1;
    elseif (g(x==u) < 1.e-5) == 1
        k2 = 1;
    else k2 = 0;
    end
    if (l<x & x<u) == 0 
        k3 = 1;
    elseif (g(l<x & x<u) < 1.e-5) == 1
        k3 = 1;
    else k3 = 0;
    end
    if (k1 & k2 & k3)
        break
    end
    
    % Find the Cauchy point
    t_bar = 1.e5*ones(n,1);;
    t_bar(g<0) = (x(g<0)-u(g<0))./g(g<0);
    t_bar(g>0) = (x(g>0)-l(g>0))./g(g>0);
    
    % Generate the ordered and non-repeated sequence of t_bar
    t_seq = unique([0;t_bar]);   % Add 0 to make the implementation easier 
    t = 0;
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

end


function [piv,index,prod] = reorder(R,y_m,l,u,beta)
%
% Find a new ordering for the last part of columns of R 
%
% Inputs:
%    R - m by n real matrix
%    y_m - a real scalar, the last elemement of m-dimensional vector y
%    l - n-dimensional integer vector, lower bound 
%    u - n-dimensional integer vector, upper bound  
%    beta - search radius
%
% Output:
%    piv - n-dimensional x - n-dimensional real vector, a solution

[m,n] = size(R);
r = R(m,:);
index = m-1;
prod = 1;
j = n;
piv = 1:n;

r(m:n) = abs(r(m:n));
alpha = y_m + dot(u(m:n)-l(m:n),r(m:n));
x = zeros(n,1);
s1 = 0;
s2 = dot(r(m:n-1),u(m:n-1)-l(m:n-1));
p = 0;
minlambd = 0;
minmu = 0;
while j >= m
    L_j = inf;
    
    for i = m:j-1
        mu = (alpha+beta-s1) / r(i);
        lambda = (alpha-beta-s1-s2) / r(i);
        L_i = min(u(i)-l(i),floor(mu)+sign(mu-floor(mu))-1) ...
            - max(0,ceil(lambda)-sign(ceil(lambda)-lambda)+1) + 1;
        if L_i < L_j
            L_j = L_i;
            p = i;
            minlambd = lambda;
            minmu = mu;
        end
    end
    
    if p ~= j
        R(:,[p,j]) = R(:,[j,p]);
        piv([p,j]) = piv([j,p]);
        r([p,j]) = r([j,p]);
        l([p,j]) = l([j,p]);
        u([p,j]) = u([j,p]);
    end
    
    if L_j <= 0
        index = j;
        break;
    end
      
    s2 = s2 - r(j)*(u(j)-l(j));
    c = round((alpha-s1)/r(j));
    left = max(0,ceil(minlambd));
    right = min(u(j)-l(j),floor(minmu));
    x(j) = max(min(c,right),left);   
    s1 = s1 + r(j)*x(j);
    prod = prod * L_j;
    j = j-1;
end

end