function ZBhat = sils_babai_asyn(R,y)
%
% Zhat = sils_babai_asyn(R,y,p) produces p babai point solutions to 
% the upper triangular integer least squares problem min_{z}||y-Rz|| 
% by a chaotic relaxation algorithm.
%
% Inputs:
%    R - n-by-n real nonsingular upper triangular matrix
%    y - n-dimensional real vector
%
% Output:
%    ZBhat - n-by-1 integer matrix (in double precision), whose j-th column 
%           is the j-th optimal solution, i.e., its residual is the j-th 
%           smallest, so ||y-R*Zhat(:,1)|| <= ...<= ||y-R*Zhat(:,p)||

%TESTTESTESTTESTTESTTEST

% ------------------------------------------------------------------
% --------  Initialization  ----------------------------------------
% ------------------------------------------------------------------

n = size(R,1);

% Current point
z = zeros(n,1);

% c(k)=(y(k)-R(k,k+1:n)*z(k+1:n))/R(k,k)
c = zeros(n,1);

% d(k): left or right search direction at level k   
d = zeros(n,1); 

% Partial squared residual norm for z
% prsd(k) = (norm(y(k+1:n) - R(k+1:n,k+1:n)*z(k+1:n)))^2
prsd = zeros(n,1); 

% Store some quantities for efficiently calculating c
% S(k,n) = y(k),
% S(k,j-1) = y(k) - R(k,j:n)*z(j:n) = S(k,j) - R(k,j)*z(j), j=k+1:n
S = zeros(n,n);
S(:,n) = y;

% path(k): record information for updating S(k,k:path(k)-1) 
path = n*ones(n,1); 

% The level at which search starts to move up to a higher level
ulevel = 0; 

% The p candidate solutions (or points) 
ZBhat = zeros(n,1); 

% Initial squared search radius
beta = inf; 


% ------------------------------------------------------------------
% --------  Search process  ----------------------------------------
% ------------------------------------------------------------------

c(n) = y(n) / R(n,n);
z(n) = round(c(n));
gamma = R(n,n) * (c(n) - z(n));
% Determine enumeration direction at level n
if c(n) > z(n)
    d(n) = 1;
else
    d(n) = -1;
end

k = n;

while 1
    % Temporary partial squared residual norm at level k
    newprsd = prsd(k) + gamma * gamma;

    if newprsd < beta
        if k ~= 1 % move to level k-1
            % Update path  
            if ulevel ~= 0 
                path(ulevel:k-1) = k;
                for j = ulevel-1 : -1 : 1
                     if path(j) < k
                           path(j) = k;
                     else
                         break;  % Note path(1:j-1) >= path(j)
                     end
                end
            end
            
            % Update S
            k = k - 1;
            for j = path(k) : -1 : k+1
                S(k,j-1) = S(k,j) - R(k,j) * z(j);
            end
            
            % Update the partial squared residual norm
            prsd(k) = newprsd;
            
            % Find the initial integer
            c(k) = S(k,k) / R(k,k);
            z(k) = round(c(k));
            gamma = R(k,k) * (c(k) - z(k));
            if c(k) > z(k) 
                d(k) = 1;
            else
                d(k) = -1;
            end
            
            ulevel = 0; 
            
        else % The babai point is found, update the set of candidate solutions
         
            ZBhat = z;
            beta = newprsd;
            z(1) = z(1) + d(1);
            gamma = R(1,1)*(c(1)-z(1));
            if d(1) > 0
                d(1) = -d(1) - 1;
            else
                d(1) = -d(1) + 1;
            end
        end
    else  
        if k == n % The Babai point has been found
            break
        else  % Move back to level k+1
            if ulevel == 0
               ulevel = k;
            end
            k = k + 1; 
            % Find a new integer at level k  
            z(k) = z(k) + d(k);
            gamma = R(k,k) * (c(k) - z(k));
            if d(k) > 0
                d(k) = -d(k) - 1;
            else
                d(k) = -d(k) + 1;
            end
        end 
    end
end


