function d= sizesel(R,sigma,tr)
%block size selection for BOB, tr is the threshold
%basic verision
[n,m] = size(R);
if n==1
   d=1;
else
    if tr<10^-3
        d = ones(n,1);
        return
    end
    n1 = round(n/2);
    R1=R(1:n1,1:n1);
    R2=R(n1+1:n,n1+1:n);
    PB1 = success_prob_babai(R1,sigma);
    PB2 = success_prob_babai(R2,sigma);
    if PB1*PB2>=tr
        d = ones(n,1);
    else 
        P1 = max(pest(R1,sigma),PB1);
        P2 = max(pest(R2,sigma),PB2);
        if P1*P2 <tr
            d=n;
        else
            w1 = sqrt(tr*P1/P2);
            w2 = sqrt(tr*P2/P1);
            d = [sizesel(R1,sigma,w1);sizesel(R2,sigma,w2) ];
        end
    end
end