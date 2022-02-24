function x = obils(B,y,l,u)
% x = B * y;
% 
% x = obils(B,y,l,u) produces the solution to the overdetermined box
% constrained integer least squares problem min_{l<=x<=u}||y-Bx||
%
% Inputs:
%    B - m-by-n real matrix with full column rank
%    y - m-dimensional real vector
%    l - n-dimensional integer vector, lower bound 
%    u - n-dimensional integer vector, upper bound, l < u    
%
% Output:
%    x - n-dimensional integer vector (in double precision) for
%        the optimal solution
%

% Subfunctions: obils_reduction, obils_search

% Main References: 
% [1] S. Breen and X.-W. Chang. Column Reordering for 
%     Box-Constrained Integer Least Squares Problems, 
%     Proceedings of IEEE GLOBECOM 2011, 6 pages.
% [2] X.-W. Chang and Q. Han. Solving Box-Constrained Integer Least 
%     Squares Problems, IEEE Transactions on Wireless Communications,  
%     7 (2008), pp. 277-287.

% Authors: Xiao-Wen Chang, www.cs.mcgill.ca/~chang 
%          Xiangyu Ren
% Copyright (c) 2015. Scientific Computing Lab, McGill University.
% April 2015. Last revision: December 2015


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

if rank(B) < n
    error('Matrix does not have full column rank, use ubils')
end

l = ceil(l); u = floor(u);  % To make it work with real bounds
for i = 1 : n
    if l(i) >= u(i)
        error('Invalid upper bound or lower bound');
    end
end

% Reduction
[~, R,y,l,u,p] = obils_reduction(B,y,l,u);

% Search 
z = obils_search(R,y,l,u);

% Reorder z to obtain the optimal solution
x = zeros(n,1);
for i = 1 : n
    x(p(i)) = z(i); 
end
