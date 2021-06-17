function [R,Z,y] = eo_sils_reduction(B,y)
%
% [R,Z,y] = sils_reduction(B,y) reduces the general standard integer 
% least squares problem to an upper triangular one by the LLL-QRZ 
% factorization Q'*B*Z = [R; 0]. The orthogonal matrix Q 
% is not produced. 
%
% Inputs:
%    B - m-by-n real matrix with full column rank
%    y - m-dimensional real vector to be transformed to Q'*y
%
% Outputs:
%    R - n-by-n LLL-reduced upper triangular matrix
%    Z - n-by-n unimodular matrix, i.e., an integer matrix with
%    |det(Z)|=1pro
%    y - m-vector transformed from the input y by Q', i.e., y := Q'*y
%

% Subfunction: qrmcp

% Main Reference: 
% X. Xie, X.-W. Chang, and M. Al Borno. Partial LLL Reduction, 
% Proceedings of IEEE GLOBECOM 2011, 5 pages.

% Authors: Xiao-Wen Chang, www.cs.mcgill.ca/~chang
%          Xiaohu Xie, Tianyang Zhou 
% Copyright (c) 2006-2016. Scientific Computing Lab, McGill University.
% October 2006. Last revision: June 2016


[m,n] = size(B);

% QR factorization with minimum-column pivoting
[Q, R] = qr(B);

R_ = R(:,1:n);
y = Q'*y;

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

                % Bring R baci to an upper triangular matrix by a Givens rotation
%                 [G,R([i1,i],i1)] = planerot(R([i1,i],i1));
%                 R([i1,i],i:n) = G * R([i1,i],i:n);   

                % Apply the Givens rotation to y
%                 y([i1,i]) = G * y([i1,i]);
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

                % Bring R baci to an upper triangular matrix by a Givens rotation
%                 [G,R([i1,i],i1)] = planerot(R([i1,i],i1));
%                 R([i1,i],i:n) = G * R([i1,i],i:n);   

                % Apply the Givens rotation to y
%                 y([i1,i]) = G * y([i1,i]);
            end
        end
    end    
    for i = 3:2:n
        i1 = i-1;
        if swap(i) == 1 
            [G,R([i1,i],i1)] = planerot(R([i1,i],i1));
            R([i1,i],i:n) = G * R([i1,i],i:n);  
            y([i1,i]) = G * y([i1,i]); 
            swap(i) = 0;
        end
%        for i = k-2:-1:1
%             zeta = round(R(i,k)/R(i,i));  
%             if zeta ~= 0
%                 R(1:i,k) = R(1:i,k) - zeta * R(1:i,i);  
%                 Z(:,k) = Z(:,k) - zeta * Z(:,i);  
%             end
%        end
    end
end
Q1 = R_*Z/R;
det(Q1*Q1')