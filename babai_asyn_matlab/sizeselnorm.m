function d= sizeselnorm(sigma,tr,s,e,total)
n = e-s+1
if n==1
   d=1;
else
    n1 = round((s+e-1)/2);
    P1 = pestnorm(sigma,s,n1,total);
    P2 = pestnorm(sigma,n1+1,e,total);
    if P1*P2 <tr
        d=n;
    else
        w1 = sqrt(tr*P1/P2);
        w2 = sqrt(tr*P2/P1);
        d = [sizeselnorm(sigma,w1,s,n1,total);sizeselnorm(sigma,w2,n1+1,e,total) ];
    end
end