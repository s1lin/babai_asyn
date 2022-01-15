function x = ubils(B,y,l,u)
% 
% x = ubils(B,y,l,u) produces the solution to the underdetermined 
% box-constrained integer least square problem min_{l<=x<=u}||y-Bx|| 
%
% Inputs:
%    B - m-by-n real matrix with m<n or n<=m but rank deficient
%    y - m-dimensional real vector
%    l - n-dimensional integer vector, lower bound 
%    u - n-dimensional integer vector, upper bound, l < u
%
% Output:
%    x - n-dimensional integer vector (in double precision) for
%        the optimal solution

% Subfunctions: ubils_reduction, ubils_search

% Main References:
% [1] X.-W. Chang and X. Yang. An efficient tree search
%     decoder with column reordering for underdetermined MIMO systems. 
%     Proceedings of IEEE GLOBECOM 2007, 5 pages.
% [2] S. Breen and X.-W. Chang. Column Reordering for 
%     Box-Constrained Integer Least Squares Problems.
%     Proceedings of IEEE GLOBECOM 2011, 6 pages.

% Authors: Xiao-Wen Chang, www.cs.mcgill.ca/~chang 
%          Xiangyu Ren
% Copyright (c) 2015-2018. Scientific Computing Lab, McGill University.
% April 2015. Last revision: April 2018


% Check the input arguments
if nargin > 4
    error('Too many input arguments');
end

if nargin < 4
    error('Not enought input arguments');
end

[m,n] = size(B);

if m ~= size(y,1) || size(y,2) ~= 1 || ... 
       n ~= size(l,1) || size(l,2) ~= 1 || ...
       n ~= size(u,1) || size(u,2) ~= 1     % Input error
    error('Input arguments have a matrix dimension error!')
end

l = ceil(l); u = floor(u);  % To make it work with real bounds
for i = 1 : n
    if l(i) >= u(i)
        error('Invalid upper bound or lower bound');
    end
end

% Reduction - reduce the problem to the trapezoid form
[R,y,l,u,p,z0,beta] = ubils_reduction(B,y,l,u);

% Search - find the optimal solution to the reduced problem
z = ubils_search(R,y,l,u,z0,beta);

% Reorder z to obtain the optimal solution to the original problem
x = zeros(n,1);
for i = 1 : n
    x(p(i)) = z(i);
end
