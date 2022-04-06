function [A, R, P, y] = aspl_p(A, y)

% ------------------------------------------------------------------
% --------  Perfome the partial LLL reduction  ---------------------
% ------------------------------------------------------------------

f = true;
swap = zeros(n,1);
even = true;
start = 2;
G = cell(n, 1);
tic
while f || ~even
    f = false;
    for k = start:2:n
        k1 = k-1;
        zeta = round(R(k1,k) / R(k1,k1));
        alpha = R(k1,k) - zeta * R(k1,k1);
        if R(k1,k1)^2 > 2 * (alpha^2 + R(k,k)^2)
            swap(k) = 1;
            f = true;           
        end
    end
    for k = start:2:n
        if swap(k) == 1
         k1 = k - 1;         
         % Permute columns k-1 and k of R and Z
         R(1:k,[k1,k]) = R(1:k,[k,k1]);       
         Z(:,[k1,k]) = Z(:,[k,k1]);
         [G{k},R([k1,k],k1)] = planerot(R([k1,k],k1));
        end
    end
    
    for k = start:2:n    
        if swap(k) == 1
            k1 = k-1;
            R([k1,k],k:n) = G{k} * R([k1,k],k:n);   
            y([k1,k]) = G{k} * y([k1,k]);
            swap(k) = 0;
        end
    end

    if even
        even = false;
        start = 3;
    else
        even = true;
        start = 2;           
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

sils_lll_eval(R);
sils_reduction(A,y_LLL);


