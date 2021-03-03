function p = pestnorm(noise,s,e,total)
%get the estimated success probability of the ILS by lower bound
% for normally distribued lattice
n = e-s+1;
m=n
if n==1
     j = total-e+1
     ff=@(x) erf(1*x/2/sqrt(2)/noise).*(x.^(j-1).*exp(-x.^2/2)./2.^(j/2-1)./gamma(j/2));
     p=integral(ff,0,inf);
else

    mu=(exp(0.5*log(factorial(total-s)/factorial(total-e)))/(pi^(m/2)/factorial(m/2)))^(1/m);
    p= chi2cdf(mu^2/noise^2,m);
end