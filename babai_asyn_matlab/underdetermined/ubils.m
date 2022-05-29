function [x,flag] = ubils(B,y,l,u,cp,maxtime)
% 
% x = ubils(B,y,l,u) produces the optimal solution to the underdetermined 
%     box-constrained integer least square problem min_{l<=x<=u}||y-Bx||.
%     B - m-by-n real matrix with rank(B) < n
%     y - m-dimensional real vector
%     l - n-dimensional integer vector, lower bound 
%     u - n-dimensional integer vector, upper bound, l < u
%     x - n-dimensional integer vector (in double precision), the optimal
%         solution 
%     
% x = ubils(B,y,l,u,cp) provides a column permutation option to choose
%     cp = 0 / [] (default), performs simple column permutations;
%     cp = 1, performs sophasitated column permutations, preferable for 
%             large resdiual cases. 
%
% x = ubils(B,y,l,u,cp,maxtime) specifies the maximum running time in seconds. 
%     This option is to avoid too long running time. 
%     The default value of maxtime is Inf. 
%     x is the optimal solution or a suboptimal solution.
%     
% [x,flag] = ubils(B,y,l,u,cp,maxtime) also returns a flag
%     flag = 0, x is the optimal solution
%     flag = 1, x is a suboptimal solution 

% Subfunctions: ubils_reduction, ubils_search

% Main References:
% [1] X.-W. Chang and X. Yang. An efficient tree search
%     decoder with column reordering for underdetermined MIMO systems. 
%     Proceedings of IEEE GLOBECOM 2007, 5 pages.
% [2] Jing Zhu. Numerical methods for underdetermined box-constrained 
%     integer least squares problems, M.Sc. Thesis, School of Computer 
%     Science, McGill University, 2016.

% Authors: Xiao-Wen Chang, www.cs.mcgill.ca/~chang 
%          Zhilong Chen
% Copyright (c) 2019. Scientific Computing Lab, McGill University.
% Last revision: June 2019



% Check the input arguments
if nargin > 6
    error('Too many input arguments');
end

if nargin < 4
    error('Not enought input arguments');
end

switch nargin
    case 4 
        cp = 0;
        maxtime = Inf;
    case 5 
        if isempty(cp)
           cp = 0;
        end
        if cp ~= 0 &  cp ~= 1
            error('cp should be [], 0 or 1');
        end
        maxtime = Inf;
    case 6
        if isempty(cp)
           cp = 0;
        end
        if cp ~= 0 &  cp ~= 1
            error('cp should be [], 0 or 1');
        end
        if maxtime <= 0;
           error('Maxtime has to be positive');
        end
end

% Time limit (instance)
duetime = cputime + maxtime;


[m,n] = size(B);

if m ~= size(y,1) || size(y,2) ~= 1 || ... 
       n ~= size(l,1) || size(l,2) ~= 1 || ...
       n ~= size(u,1) || size(u,2) ~= 1     % Input error
    error('Input arguments have a matrix dimension error')
end

l = ceil(l); u = floor(u);  % To make it work with real bounds
for i = 1 : n
    if l(i) >= u(i) || l(i) == -Inf || u(i) == Inf
        error('Invalid lower bound or upper bound');
    end
end


% If B is a zero matrix, one solution is obtained
if B == 0
    x = l;
    flag = 0;
    warning('The input matrix is zero and here one solution is produced')
    return
end

% Reduction - reduce the problem to the trapezoid form
[R,y,l,u,p,z0,beta] = ubils_reduction(B,y,l,u,cp);

% Search 
if beta > n*eps*norm(abs(y)+abs(R)*abs(z0))
    [z,flag] = ubils_search(R,y,l,u,z0,beta,duetime);   
else
    z = z0; % z0 is the solution of the reduced problem
    flag = 0;
end

% Reorder z to obtain the solution to the original problem
x = zeros(n,1);
for i = 1 : n
    x(p(i)) = z(i);
end 

