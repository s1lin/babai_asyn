 function [B,R,x,y] = auto_gen(n)

    B = randn(n,n);
    R = triu(qr(B)); 
    x = randi([-10,10],n,1); 
    x0 = zero(n,1);
    y = R * x + 0.5*randn(n,1);
    
    for iswp = 1:n
        parfor i = 1:n
            sum = 0;
            for j = 1:i-1
                sum = sum + R(i,j) * x(j);
            end
            x0(i) = (y(i) - sum) / R(i,i);
            x(i) = round(x(i));
        end
    end
    
    parfor k = 1:n     
        for kp = 1:n
            sum = 0;
            for j = 1:kp-1
                sum = sum + R(kp,j) * x(j)
            end
            x(k) = (y(k) - sum) / R(k,k);
            x(k) = round(x(k));
        end        
    end
    %x(k) = (y(k) - R(k,k+1:n) * x(k+1:n)) / R(k,k);