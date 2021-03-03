function y=success_prob_double_n(R,sigma)
%low bound by double layer SP
y=1;
[m,n]=size(R);
for i=1:n/2
    y=y*success_pro_double(R(2*i-1:2*i,2*i-1:2*i),sigma);
end
end