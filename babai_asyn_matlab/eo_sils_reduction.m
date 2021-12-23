function [A, R, Z, y, y_LLL, x_t, d] = eo_sils_reduction(qam, n, SNR)

rng('shuffle')
%Initialize Variables
sigma = sqrt(((4^qam-1)*n)/(6*10^(SNR/10)));

Ar = normrnd(0, sqrt(1/2), n/2, n/2);
Ai = normrnd(0, sqrt(1/2), n/2, n/2);
Abar = [Ar -Ai; Ai, Ar];
A = 2 * Abar;

%True parameter x:
low = -2^(qam-1);
upp = 2^(qam-1) - 1;
xr = 1 + 2 * randi([low upp], n/2, 1);
xi = 1 + 2 * randi([low upp], n/2, 1);
xbar = [xr; xi];
x_t = (2^qam - 1 + xbar)./2;

%Noise vector v:
vr = normrnd(0, sigma, n/2, 1);
vi = normrnd(0, sigma, n/2, 1);
v = [vr; vi];

%Get Upper triangular matrix
y_LLL = A * x_t + v;

% QR factorization with minimum-column pivoting
[Q, R] = qr(A);
y = Q' * y_LLL;
%y = R(:, n+1);
%R = R(:,1:n);
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
            if zeta ~= 0
                % Perform a size reduction on R(k-1,k)
                R(k1,k) = alpha;
                R(1:k-2,k) = R(1:k-2,k) - zeta * R(1:k-2,k-1);
                Z(:,k) = Z(:,k) - zeta * Z(:,k-1);  

                %Perform size reductions on R(1:k-2,k)
                for i = k-2:-1:1
                    zeta = round(R(i,k)/R(i,i));  
                    if zeta ~= 0
                        R(1:i,k) = R(1:i,k) - zeta * R(1:i,i);  
                        Z(:,k) = Z(:,k) - zeta * Z(:,i);  
                    end
                end
            end
           
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


