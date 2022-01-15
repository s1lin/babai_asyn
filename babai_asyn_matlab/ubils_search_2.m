function z = ubils_search_2(R,y,l,u,z0,beta)
%
% z = ubils_search(R,y,l,u) produces the optimal solution to the
% underdetermined box constrained integer least squares problem
% min_{z}||y-Rz|| subject to z in [l, u] by a search algorithm.
%
% Inputs:
%     R ---- m by n real upper trapezoidal matrix with deficient column
%            rank
%     y ---- m-dimensional real vector
%     l ---- n-dimensional integer vector, lower bound 
%     u ---- n-dimensional integer vector, upper bound
%     beta ---- the initial search bound
%
% Output:
%     z ---- n-dimensional integer vector (in double precision). 

% Local functions:  bound, init, shift, update

% Main References: 
% [1] Xiao-Wen Chang, and Xiaohua Yang. An efficient tree search
%     decoder with column reordering for underdetermined MIMO systems. 
%     Global Telecommunications Conference, 2007. 5 pages
% [2] A. Ghasemmehdi and E. Agrell, Faster Recursions in Sphere Decoding,
%     IEEE Transactions on Information Theory, 57 (2011), pp. 3530-3536. 

% Authors: Xiao-Wen Chang, www.cs.mcgill.ca/~chang 
%          Xiangyu Ren
% Copyright (c) 2015. Scientific Computing Lab, McGill University.
% April 2015. Last revision: April 2018

% ------------------------------------------------------------------
% --------  Initialization  ----------------------------------------
% ------------------------------------------------------------------
[m,n] = size(R);

% Point which determins the initial search radius beta
zhat = z0;

% Current point
z = zeros(n,1);

% c(k) = (y(k)-R(k,k+1:n)*z(k+1:n))/R(k,k)
c = zeros(n,1);

% d(k): left or right search direction at level k   
d = zeros(n,1);

% lflag(k) = 1 if the lower bound is reached at level k
lflag = zeros(size(l));
% uflag(k) = 1 if the upper bound is reached at level k
uflag = lflag;

% Store some quantities for efficiently calculating c
S = zeros(n,n);

% Partial squared residual norm for z
% prsd(k) = (norm(y(k+1:n)-R(k+1:n,k+1:n)*z(k+1:n)))^2
prsd = zeros(m+1,1);

% path(k): record information for updating S(k,k:path(k)-1) 
path = n*ones(n,1); 

% The level at which search starts to move up to a higher level
ulevel = 0;  

% ------------------------------------------------------------------
% --------  Search process  ----------------------------------------
% ------------------------------------------------------------------

% A transformation for z and bounds
[R,y,u,l,Z,s] = shift(R,y,l,u);

utemp = u;

k = n;

% S(k,n) = y(k), k=1:m-1, S(k,n) = y(m), k=m:n
S(m:n,n) = y(m);
S(1:m-1,n) = y(1:m-1);

c(n) = S(n,n)/R(m,n);

% Compute new bound at level k
[l(k),u(k)] = bound(c(k),R,utemp,beta,prsd,k);

% Find the initial integer in [l(k), u(k)]
[z(k),d(k),lflag(k),uflag(k)] = init(c(k),l(k),u(k));

% dflag for down or up search direction 
dflag = 1; % Intend to move down to a lower level

while 1
    if dflag == 1
        if lflag(k) ~= 1 || uflag(k) ~= 1
           if k ~= 1 % Move to level k-1
              k1 = k - 1;
              % Update path  
              if ulevel ~= 0
                  path(ulevel:k-1) = k;
                  for j = ulevel-1:-1:1
                      if path(j) < k
                          path(j) = k;
                      else
                          break; % Note path(1:j-1) >= path(j)
                      end
                  end
              end
              
              % Update S
              if k <= m
                  for j = path(k1):-1:k1+1
                      S(k1,j-1) = S(k1,j) - R(k1,j)*z(j);
                  end
                  c(k1) = S(k1,k1) / R(k1,k1);
              else
                  S(m:k1,k1) = S(k1,k1+1) - R(m,k1+1)*z(k1+1);
                  c(k1) = S(k1,k1) / R(m,k1);
              end
              
              % Compute new bound at level k1
              [l(k1),u(k1)] = bound(c(k1),R,utemp,beta,prsd,k1);
              
              % Find the initial integer in [l(k1), u(k1)]
              if l(k1) > u(k1)
                  lflag(k1) = 1;
                  uflag(k1) = 1;
              else    
                  [z(k1),d(k1),lflag(k1),uflag(k1)] = init(c(k1),l(k1),u(k1));
            
                  if k1 <= m
                      gamma = R(k1,k1)*(c(k1)-z(k1));
                      prsd(k1)= prsd(k1+1) + gamma * gamma;
                  end
              end

              k = k1;           
              ulevel = 0;
           else % A valid point is found
              zhat = z;
              beta = sqrt(prsd(1));
              for j = 1:n
                  [l(j),u(j)] = bound(c(j),R,utemp,beta,prsd,j);
              end
              dflag = 0;
           end
        else % Will move back to a higher level          
            dflag = 0;
        end
    else
        if k == n % The optimal solution has been found, terminate
            break;
        else 
            % Move back to level k+1
            if ulevel == 0
               ulevel = k;
            end
            
            k = k + 1;
            
            if lflag(k) ~= 1 || uflag(k) ~= 1
                % Find a new integer at level k 
                [z(k),d(k),lflag(k),uflag(k)] = ...
                      update(z(k),d(k),lflag(k),uflag(k),l(k),u(k));
                if k <= m
                   gamma = R(k,k)*(c(k)-z(k));
                   prsd(k)= prsd(k+1) + gamma * gamma;
                end
                dflag = 1;
            end
        end
    end   
end

if zhat == z0  % z0 is the optimal solution
    z = zhat;
else % Shift the optimal solution back
    z = Z*zhat + s; 
end


% ------------------------------------------------------------------
% --------  Local functions  ---------------------------------------
% ------------------------------------------------------------------

function [l_k,u_k] = bound(c_k,R,u,beta,prsd,k)
%
% Compute new lower bound and upper bound for z_k
%
m = size(R,1);

if k > m-1
    lambda_k = c_k - (beta+R(m,m:k-1)*u(m:k-1))/R(m,k);
    mu_k = c_k + beta/R(m,k);
else
    lambda_k = c_k - sqrt(beta^2-prsd(k+1))/R(k,k);
    mu_k = c_k + sqrt(beta^2-prsd(k+1))/R(k,k);
end

if lambda_k - floor(lambda_k) < 1e-12 && lambda_k ~= 0
    lambda_k = floor(lambda_k);
end

if ceil(mu_k) - mu_k < 1e-12 && mu_k ~= u(k)
    mu_k = ceil(mu_k);
end

l_k = ceil(max(0,lambda_k));
u_k = floor(min(u(k),mu_k));



function [z_k,d_k,lflag_k,uflag_k] = init(c_k,l_k,u_k)
%
% Find the initial integer and the search direction at level _k
%
z_k = round(c_k);
if z_k <= l_k
    z_k = l_k;
    lflag_k = 1;  % The lower bound is reached
    uflag_k = 0;
    d_k = 1;
elseif z_k >= u_k
    z_k = u_k;
    uflag_k = 1;  % The upper bound is reached
    lflag_k = 0;
    d_k = -1;
else
    lflag_k = 0; 
    uflag_k = 0;
    if c_k > z_k
        d_k = 1;
    else
        d_k = -1;
    end
end



function [R,y,u,l,Z,s] = shift(R,y,l,u)
%
% Perform transformations: 
%    z(j) := -l(j) + z(j) if r_{tj}>=0  for t=m or t=j
%    z(j) := u(j) - z(j)  if r_{tj}<0   for t=m or t=j
% so that z(j) is in {0,1, . . . ,u_j-l_j} and R(t,j)>=0
%
[m,n] = size(R);
R_temp = R;
Z = zeros(n,n);
s = zeros(n,1);

for j = 1:n
    if j > m
        t = m;
    else 
        t = j;
    end
    
    if R(t,j) >= 0
        Z(j,j) = 1;
        s(j) = l(j);
    else
        Z(j,j) = -1;
        s(j) = u(j);
    end
end

R = R_temp * Z;
y = y - R_temp*s;
u = u - l;
l = zeros(size(l));



function [z_k,d_k,lflag_k,uflag_k] = update(z_k,d_k,lflag_k,uflag_k,l_k,u_k)
%
% Find a new integer at level k and record it if it hits a boundary.
%
if lflag_k == 0 && uflag_k == 0
    zk = z_k + d_k;
    if zk > u_k
        uflag_k = 1;
    elseif zk < l_k
        lflag_k = 1;
    else
        z_k = zk;
        if d_k > 0
            d_k = -d_k - 1;
        else
            d_k = -d_k + 1;
        end
    end
end

if lflag_k == 1 && uflag_k == 0
    z_k = z_k + 1;
    if z_k > u_k
       uflag_k = 1;
    end
elseif lflag_k == 0 && uflag_k == 1
    z_k = z_k - 1;
    if z_k < l_k
        lflag_k = 1;
    end
end