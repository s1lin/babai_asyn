function [A, R, Z, y, y_LLL, x_t, d] = eo_sils_reduction(k, n, SNR)

rng('shuffle')
%Initialize Variables  
sigma = sqrt(((4^k-1)*n)/(6*10^(SNR/10)));            

Ar = normrnd(0, sqrt(1/2), n/2, n/2);
Ai = normrnd(0, sqrt(1/2), n/2, n/2);
Abar = [Ar -Ai; Ai, Ar];
A = 2 * Abar;

%True parameter x:
low = -2^(k-1);
upp = 2^(k-1) - 1;
xr = 1 + 2 * randi([low upp], n/2, 1);
xi = 1 + 2 * randi([low upp], n/2, 1);
xbar = [xr; xi];
x_t = (2^k - 1 + xbar)./2;                    

%Noise vector v:
vr = normrnd(0, sigma, n/2, 1);
vi = normrnd(0, sigma, n/2, 1);
v = [vr; vi];

%Get Upper triangular matrix
y_LLL = A * x_t + v;

% QR factorization with minimum-column pivoting
[~, R] = qr([A y_LLL]);

y = R(:, n+1);
R = R(:,1:n);
R_ = R;
% Obtain the permutation matrix Z
Z = eye(n);

% for j = 1 : n
%     Z(piv(j),j) = 1;
% end

% ------------------------------------------------------------------
% --------  Perfome the partial LLL reduction  ---------------------
% ------------------------------------------------------------------

f = true;
swap = zeros(n,1);
tic
while f
    f = false;
    for i = 2:2:n
        i1 = i-1;
        zeta = round(R(i1,i) / R(i1,i1));  
        alpha = R(i1,i) - zeta * R(i1,i1);     
        if R(i1,i1)^2 > (1 + 1.e-0) * (alpha^2 + R(i,i)^2)                        
            if zeta ~= 0
                % Perform a size reduction on R(k-1,k) 
                f = true;
                swap(i) = 1;
                R(i1,i) = alpha;
                R(1:i-2,i) = R(1:i-2,i) - zeta * R(1:i-2,i-1);
                Z(:,i) = Z(:,i) - zeta * Z(:,i-1);        
                
                % Permute columns k-1 and k of R and Z
                R(1:i,[i1,i]) = R(1:i,[i,i1]);
                Z(:,[i1,i]) = Z(:,[i,i1]);
            end
        end
    end  
    for i = 2:2:n
        i1 = i-1;
        if swap(i) == 1 
            [G,R([i1,i],i1)] = planerot(R([i1,i],i1));
            R([i1,i],i:n) = G * R([i1,i],i:n);  
            y([i1,i]) = G * y([i1,i]); 
            swap(i) = 0;
        end
    end
    for i = 3:2:n
        i1 = i-1;
        zeta = round(R(i1,i) / R(i1,i1));  
        alpha = R(i1,i) - zeta * R(i1,i1); 
        if R(i1,i1)^2 > (1 + 1.e-0) * (alpha^2 + R(i,i)^2)           
            if zeta ~= 0
                f = true;
                swap(i) = 1;
                % Perform a size reduction on R(k-1,k)
                R(i1,i) = alpha;
                R(1:i-2,i) = R(1:i-2,i) - zeta * R(1:i-2,i-1);
                Z(:,i) = Z(:,i) - zeta * Z(:,i-1);         
                
                % Permute columns k-1 and k of R and Z
                R(1:i,[i1,i]) = R(1:i,[i,i1]);
                Z(:,[i1,i]) = Z(:,[i,i1]);
            end
        end
    end    
    for i = 3:2:n
        i1 = i-1;
        if swap(i) == 1             
            % Bring R baci to an upper triangular matrix by a Givens rotation
            [G,R([i1,i],i1)] = planerot(R([i1,i],i1));
            R([i1,i],i:n) = G * R([i1,i],i:n);  
            
            % Apply the Givens rotation to y
            y([i1,i]) = G * y([i1,i]); 
            swap(i) = 0;
        end
    end
end
toc
Q1 = R_*Z/R;
% Q1
d = det(Q1*Q1');

