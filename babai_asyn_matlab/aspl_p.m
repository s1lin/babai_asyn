function [R, Z, y] = aspl_p(A, y)
[m, n] = size(A);
[Q, R, ~] = qrmgs_row(A, y);
y_LLL = y;
y = Q'*y;

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
tic
while f
    f = false;
    for k = start:2:n
        k1 = k - 1;
        zeta = round(R(k1,k) / R(k1,k1));
        alpha = R(k1,k) - zeta * R(k1,k1);
        if R(k1,k1)^2 > (1 + 1.e-10) * (R(k1,k)^2 + R(k,k)^2)
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
    if ~f
        for k = start:2:n
            k1 = k - 1;
            zeta = round(R(k1,k) / R(k1,k1));
            alpha = R(k1,k) - zeta * R(k1,k1);
            if R(k1,k1)^2 > (1 + 1.e-10) * (R(k1,k)^2 + R(k,k)^2)
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
end

toc
Q = A*Z*inv(R);
if n <= 16
    G
    R
    Q
    Q'*Q
end
diff = norm(Q'*y_LLL - y, 2)

% sils_lll_eval(R);
% sils_reduction(A,y_LLL);


