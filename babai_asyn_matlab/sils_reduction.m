% function [R,Z,y] = sils_reduction(B,y)
% %
% % [R,Z,y] = sils_reduction(B,y) reduces the general standard integer 
% % least squares problem to an upper triangular one by the LLL-QRZ 
% % factorization Q'*B*Z = [R; 0]. The orthogonal matrix Q 
% % is not produced. 
% %
% % Inputs:
% %    B - m-by-n real matrix with full column rank
% %    y - m-dimensional real vector to be transformed to Q'*y
% %
% % Outputs:
% %    R - n-by-n LLL-reduced upper triangular matrix
% %    Z - n-by-n unimodular matrix, i.e., an integer matrix with
% %    |det(Z)|=1
% %    y - m-vector transformed from the input y by Q', i.e., y := Q'*y
% %
% 
% % Subfunction: qrmcp
% 
% % Main Reference: 
% % X. Xie, X.-W. Chang, and M. Al Borno. Partial LLL Reduction, 
% % Proceedings of IEEE GLOBECOM 2011, 5 pages.
% 
% % Authors: Xiao-Wen Chang, www.cs.mcgill.ca/~chang
% %          Xiaohu Xie, Tianyang Zhou 
% % Copyright (c) 2006-2016. Scientific Computing Lab, McGill University.
% % October 2006. Last revision: June 2016
% 
% 
% [m,n] = size(B);
% 
% % QR factorization with minimum-column pivoting Q'*B*P =R
% %[R,piv,y] = qrmcp(B,y);
% [Q, R] = qr(B);
% %B_ = B;
% 
% %Q_ = Q;
% % B-Q*R
% y = Q'*y;
% % Obtain the permutation matrix Z
% Z = eye(n);
% % Z = zeros(n,n);
% % for j = 1 : n
% %   Z(piv(j),j) = 1;
% % end
% R_ = R;% / Z;
% % ------------------------------------------------------------------
% % --------  Perfome the partial LLL reduction  ---------------------
% % ------------------------------------------------------------------
% 
% k = 2;
% tic
% while k <= n
%     
%     k1 = k-1;
%     zeta = round(R(k1,k) / R(k1,k1));  
%     alpha = R(k1,k) - zeta * R(k1,k1);  
% 
%     if R(k1,k1)^2 > (1 + 1.e-10) * (alpha^2 + R(k,k)^2)   
%         if zeta ~= 0
%             % Perform a size reduction on R(k-1,k)
%             R(k1,k) = alpha;
%             R(1:k-2,k) = R(1:k-2,k) - zeta * R(1:k-2,k-1);
%             Z(:,k) = Z(:,k) - zeta * Z(:,k-1);  
%             
%             % Perform size reductions on R(1:k-2,k)
%             for i = k-2:-1:1
%                 zeta = round(R(i,k)/R(i,i));  
%                 if zeta ~= 0
%                     R(1:i,k) = R(1:i,k) - zeta * R(1:i,i);  
%                     Z(:,k) = Z(:,k) - zeta * Z(:,i);  
%                 end
%             end
%         end
%         
%         % Permute columns k-1 and k of R and Z
%         R(1:k,[k1,k]) = R(1:k,[k,k1]);
%         Z(:,[k1,k]) = Z(:,[k,k1]);
%         
%         % Bring R back to an upper triangular matrix by a Givens rotation
%         [G,R([k1,k],k1)] = planerot(R([k1,k],k1));
%         R([k1,k],k:n) = G * R([k1,k],k:n);   
%         
%         % Apply the Givens rotation to y
%         y([k1,k]) = G * y([k1,k]);
%         
%         if k > 2
%             k = k - 1;
%         end
%         
%     else    
%         k = k + 1;
%     end
% end
% toc
% Q1 = R_*Z/R;
% det(Q1*Q1')
% diff = Q'*B*Z-R
% norm(Q'*B*Z-R, 1)
% sils_lll_eval(R);


function [R,Z,y] = sils_reduction(B,y)
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
[Q, R] = qr(B);

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

k = 2;

while k <= n
    
    k1 = k-1;
    zeta = round(R(k1,k) / R(k1,k1));  
    alpha = R(k1,k) - zeta * R(k1,k1);  

    if R(k1,k1)^2 > (1 + 1.e-10) * (alpha^2 + R(k,k)^2)   
        if zeta ~= 0
            % Perform a size reduction on R(k-1,k)
            R(k1,k) = alpha;
            R(1:k-2,k) = R(1:k-2,k) - zeta * R(1:k-2,k-1);
            Z(:,k) = Z(:,k) - zeta * Z(:,k-1);  
            
            % Perform size reductions on R(1:k-2,k)
            for i = k-2:-1:1
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
                
        % Apply the Givens rotation to y
        y([k1,k]) = G * y([k1,k]);
        
        if k > 2
            k = k - 1;
        end
        
    else    
        k = k + 1;
    end
end
y
% Q = B*Z*inv(R)
% Q'*Q
% diff = Q'*y_ - y
% norm(Q'*y_ - y, 1)
% sils_lll_eval(R);
