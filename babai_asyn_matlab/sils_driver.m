function [A, R, Z, y, y_LLL, x_t, init_res, babai_norm] = sils_driver(k, m, SNR, is_qr)
    rng('shuffle')
    %Initialize Variables
    n = 2^m; %The real size       
    x_t = zeros(n, 1);
    sigma = sqrt(((4^k-1)*2^m)/(6*10^(SNR/10)));            
    Z = zeros(n, n);
    
    while (true) 
        %Initialize A:
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
        if is_qr == 1
            [Q, R] = qr(A);
            y = Q' * y_LLL;
            Z = eye(n, n);
        else            
            [R, Z, y] = sils_reduction(A, y_LLL);
        end
        
        init_res = norm(y_LLL - A * x_t);
        if all(Z(:) >= 0) && all(Z(:) <= 1)
            break
        end
    end

    %%%TEST BABAI:
    upper = 2^k - 1;
    z_B = zeros(n, 1);
    
    for j = n:-1:1
        z_B(j) = (y(j) - R(j, j + 1:n) * z_B(j + 1:n)) / R(j, j);
        if(round(z_B(j)) > upper)
            z_B(j) = upper;
        elseif (round(z_B(j)) < 0)
            z_B(j) = 0;
        else
            z_B(j) = round(z_B(j));
        end
    end
    
    babai_norm = norm(y_LLL - A * (Z * z_B));
    A = A';
    R = R';
