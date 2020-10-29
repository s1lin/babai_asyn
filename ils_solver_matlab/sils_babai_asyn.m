function [Zhat, z] = sils_babai_asyn(R,y)


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

% The level at which search starts to move up to a higher level
Zhat = zeros(n,1); 
% Initial squared search radius
beta = inf; 

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
            k = k - 1;
            s = y(k) - R(k, k + 1:n) * z(k + 1:n);
            
            % Update the partial squared residual norm
            prsd(k) = newprsd;
            
            % Find the initial integer
            %c(k) = S(k,k) / R(k,k);
            c(k) = s / R(k,k);
            z(k) = round(c(k));
            gamma = R(k,k) * (c(k) - z(k));
            if c(k) > z(k) 
                d(k) = 1;
            else
                d(k) = -1;
            end
            
        else % A new point is found, update the set of candidate solutions
            beta = newprsd;
            Zhat = z;
            z(1) = z(1) + d(1);
            %display(z)
            gamma = R(1,1)*(c(1)-z(1));
            if d(1) > 0
                d(1) = -d(1) - 1;
            else
                d(1) = -d(1) + 1;
            end
        end
    else  
        if k == n % The p optimal solutions have been found
            break
        else  % Move back to level k+1
            %if ulevel == 0
            %   ulevel = k;
            %end
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


