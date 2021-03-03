function y=success_prob_babai(R,sigma)
%SP of the Babai point
%sigma noise Standard variation
% upper triangular R
[m,n]=size(R);
y=1;
for i=1:n
    y=y*erf(abs(R(i,i))/2/sigma/sqrt(2));
end
end