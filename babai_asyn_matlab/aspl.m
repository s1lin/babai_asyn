function [R,Z,y] = aspl(A,y)
% 
% rng('shuffle')
% %Initialize Variables
% sigma = sqrt(((4^qam-1)*n)/(6*10^(SNR/10)));
% 
% Ar = normrnd(0, sqrt(1/2), n/2, n/2);
% Ai = normrnd(0, sqrt(1/2), n/2, n/2);
% Abar = [Ar -Ai; Ai, Ar];
% A = 2 * Abar;
% 
% %True parameter x:
% low = -2^(qam-1);
% upp = 2^(qam-1) - 1;
% xr = 1 + 2 * randi([low upp], n/2, 1);
% xi = 1 + 2 * randi([low upp], n/2, 1);
% xbar = [xr; xi];
% x_t = (2^qam - 1 + xbar)./2;
% 
% %Noise vector v:
% vr = normrnd(0, sigma, n/2, 1);
% vi = normrnd(0, sigma, n/2, 1);
% v = [vr; vi];
% 
% %Get Upper triangular matrix
% y_LLL = A * x_t + v;
% % 
% % % QR factorization with minimum-column pivoting
[m, n] = size(A);
[Q, R] = qr(A);
y_LLL = y;
y = Q' * y;
% %y = R(:, n+1);
% %R = R(:,1:n);
% % Obtain the permutation matrix Z
Z = eye(n);
% ------------------------------------------------------------------
% --------  Perfome the partial LLL reduction  ---------------------
% ------------------------------------------------------------------


f = true;
even = true;
start = 2;
iter = 0;
tic
while f
    iter = iter + 1;
    f = false;
    for e = 1:1:n-1
        for k= n:-1:n-e+1
%             k1 = k - 1;
%             zeta = round(R(k1,k) / R(k1,k1)); 
%             alpha = R(k1,k) - zeta * R(k1,k1);
            %if R(k1,k1)^2 > (1 + 1.e-10) * (alpha^2 + R(k,k)^2) 
%                 if zeta ~= 0
                    % Perform a size reduction on R(k-1,k)
    %                 R(k1,k) = R(k1,k) - zeta * R(k1,k1);
    %                 R(1:k-2,k) = R(1:k-2,k) - zeta * R(1:k-2,k-1);
    %                 Z(:,k) = Z(:,k) - zeta * Z(:,k-1);  

                    % Perform size reductions on R(1:k-2,k)
                    for i = k-1:-1:1
                        zeta = round(R(i,k)/R(i,i));  
                        if zeta ~= 0
                            R(1:i,k) = R(1:i,k) - zeta * R(1:i,i);  
                            Z(:,k) = Z(:,k) - zeta * Z(:,i);  
                        end
                    end
%                 end
            %end
        end
    end
    for k = start:2:n        
        k1 = k - 1;    
        zeta = round(R(k1,k) / R(k1,k1));  
        alpha = R(k1,k) - zeta * R(k1,k1);
        if R(k1,k1)^2 > (1 + 1.e-10) * (alpha^2 + R(k,k)^2)
            f = true;
            % Permute columns k-1 and k of R and Z
            R(1:k,[k1,k]) = R(1:k,[k,k1]);       
            Z(:,[k1,k]) = Z(:,[k,k1]);

            % Bring R back to an upper triangular matrix by a Givens rotation
            [G,R([k1,k],k1)] = planerot(R([k1,k],k1));
            R([k1,k],k:n) = G * R([k1,k],k:n);   

            % Apply the Givens rotation to y
            y([k1,k]) = G * y([k1,k]);
         
        end
    end
    if even
        even = false;
        start = 3;
    else
        even = true;
        start = 2;
    end
%     if ~f
%         for k = start:2:n        
%             k1 = k - 1;    
%             zeta = round(R(k1,k) / R(k1,k1));  
%             alpha = R(k1,k) - zeta * R(k1,k1);
%             if R(k1,k1)^2 > (1 + 1.e-10) * (alpha^2 + R(k,k)^2)
%                 f = true;
%                 % Permute columns k-1 and k of R and Z
%                 R(1:k,[k1,k]) = R(1:k,[k,k1]);       
%                 Z(:,[k1,k]) = Z(:,[k,k1]);
% 
%                 % Bring R back to an upper triangular matrix by a Givens rotation
%                 [G,R([k1,k],k1)] = planerot(R([k1,k],k1));
%                 R([k1,k],k:n) = G * R([k1,k],k:n);   
% 
%                 % Apply the Givens rotation to y
%                 y([k1,k]) = G * y([k1,k]);
%             end
%         end
%     end
    
end
toc
Q = A*Z*inv(R);
iter
if n <= 16
    Z
    R
    Q
    y
end
diff = norm(Q'*y_LLL - y, 2)


k = 2;

while k <= n
    
    k1 = k-1;
    zeta = round(R(k1,k) / R(k1,k1));  
    alpha = R(k1,k) - zeta * R(k1,k1);  

    if R(k1,k1)^2 > (1 + 1.e-10) * (alpha^2 + R(k,k)^2)   
        disp(["failed",k])      
    end     
    k = k + 1;
end
sils_reduction(A,y_LLL);

end


function [R, y, f] = swap_GR(R, Z, y, start, f)
    [~, n] = size(R);
    for k = start:2:n        
        k1 = k - 1;    
        zeta = round(R(k1,k) / R(k1,k1));  
        alpha = R(k1,k) - zeta * R(k1,k1);
        if R(k1,k1)^2 > (1 + 1.e-10) * (alpha^2 + R(k,k)^2)
            f = true;
            % Permute columns k-1 and k of R and Z
            R(1:k,[k1,k]) = R(1:k,[k,k1]);       
            Z(:,[k1,k]) = Z(:,[k,k1]);

            % Bring R back to an upper triangular matrix by a Givens rotation
            [G,R([k1,k],k1)] = planerot(R([k1,k],k1));
            R([k1,k],k:n) = G * R([k1,k],k:n);   

            % Apply the Givens rotation to y
            y([k1,k]) = G * y([k1,k]);
         
        end
    end
end
