function [R,Z,y] = block_LLL(B,y,k)
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
%    Z - n-by-n unimodular matrix, i.e., an integer matrix with |det(Z)|=1
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
%[R,piv,y] = qrmcp(B,y);
[Q, R, ~] = qrmgs_row(B, y);

y_ = y;
y = Q'*y;
R_ = R;
% Obtain the permutation matrix Z
% Z = zeros(n,n);
Z = eye(n);
% for j = 1 : n
%     Z(piv(j),j) = 1;
% end

% ------------------------------------------------------------------
% --------  Perfome the partial LLL reduction  ---------------------
% ------------------------------------------------------------------
d = n/k; 
f = 0;
change = ones(1,d);
nextChange = ones(1,d);

tic
while 1
    
    for i = 1:d
        if change(i) ~= 1
            continue
        end
        R_i = R(((i-1)*d + 1):i*k,((i-1)*k + 1):i*k);
        [Q,R_i,Z_i,r] = Local_PLLL(R_i,f);
        R(((i-1)*d + 1):i*k,((i-1)*k + 1):i*k) = R_i;
        if Z_i == eye(d,d)
            continue
        end
        nextChange(max(1,i-1))=1;
        nextChange(i) = 1;
        Z(1:d,i)=Z(1:d,i)*Z_i;
        R(1:i-1,i) = R(1:i-1,i)*Z_i;
        R(i,i+1:d) = Q'*R(i,i+1:d);
        
              
    end
    for i = 2:d
        [R(1:i-1,i),Z_i] = BPSR(R(1:i-1,i),R(1:i-1,1:i-1),r);
        Z(:,i)=Z(:,i)+Z(:,1:i-1)*Z_i; 
    end
    if nextChange(:)==0
        break
    end
    f = 1;
    for i = 1:d
        change(i) = nextChange(i);
        nextChange(i) = 0;
    end  
end
[R,Z_i] = BSR(R);
Z = Z*Z_i;
toc
if n <= 16
    Z
    R
    y
end
end
% Q = B*Z*inv(R)
% Q'*Q
% diff = Q'*y_ - y
% norm(Q'*y_ - y, 1)
% sils_lll_eval(R);
function [R,Z] = BPSR(R1,R2,c)
    k = size(c);
    i = size(R1);
    i = i - 1;
    for t = i-1:-1:1
        for j = 1:k
            if c(j)==1
                
            end
        end
        R1(1:t-1,i) = R1(1:t-1,i)-R2(1:t-1,t)*Z(t,i);
    end
end
function [Q,R,Z,c] = Local_PLLL(R, f)
    [~, n] = size(R);
    if f == 0
        k = 2;
    else
        k = n/2+1;
    end
    Q = eye(n);
    Z = eye(n);
    c = zeros(1,n);
    while k <= n
    
        k1 = k-1;
        zeta = round(R(k1,k) / R(k1,k1));  
        alpha = R(k1,k) - zeta * R(k1,k1);  

        if R(k1,k1)^2 < (alpha^2 + R(k,k)^2)   
            if zeta ~= 0
                % Perform a size reduction on R(k-1,k)
                c(k) = 1;
                % Perform size reductions on R(1:k-2,k)
                for i = k-1:-1:1
                    zeta = round(R(i,k)/R(i,i));  
                    if zeta ~= 0
                        R(1:i,k) = R(1:i,k) - zeta * R(1:i,i);  
                        Z(:,k) = Z(:,k) - zeta * Z(:,i);  
                    end
                end
            end
        
            % Permute columns k-1 and k of R and Z
            R(1:k,[k1,k]) = R(1:k,[k,k1]);       
            Z(:,[k1,k]) = Z(:,[k,k1]);

            % Bring R back to an upper triangular matrix by a Givens rotation
            [G,R([k1,k],k1)] = planerot(R([k1,k],k1));
            R([k1,k],k:n) = G * R([k1,k],k:n);   
            Q(:,[k1,k]) = Q(:,[k,k1])*G';

            if k > 2
                k = k - 1;
            end
        
        else    
            k = k + 1;
        end
    end
end