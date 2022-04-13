function [A, y, R0] = gen_lll_problem(k, m, n)
rng('shuffle')
if k==0
    A = randn(m,n);
    y = randn(m,1);
else
    [U, ~] = qr(randn(m,n));
    [V, ~] = qr(randn(m,n));
    D = zeros(n,n);
    for i = 1:n
        D(i,i) = 10^(3*(n/2-i)/(n-1));
    end
    A = U*D*V';
    y = randn(m,1);
end

[R0,~,~] = aspl(A,y);
[R0,~,~] = asplk1(A,y);
end



