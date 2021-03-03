function [sp dg]=precomputation(R,sigma)
%preproduct of the r_ii and SP of Babai
[m,n]=size(R);
sp=ones(1,n+1);
dg=ones(1,n+1);
for i=1:n
    sp(i+1)=sp(i)*erf(abs(R(i,i))/2/sigma/sqrt(2));
    dg(i+1) = dg(i)*abs(R(i,i));
end
end