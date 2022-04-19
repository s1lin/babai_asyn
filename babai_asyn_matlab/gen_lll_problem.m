function [A, y, R0] = gen_lll_problem(k, m, n)
rng('shuffle')
if k==0
    Ar = randn(n/2);
    Ai = randn(n/2);    
else
    a = rand(1);
    b = rand(1);
    PHI = zeros(n/2,n/2);
    phi = zeros(n/2,n/2);
    for i = 1:n/2
        for j = 1:n/2
            phi(i, j) = a^abs(i-j);
            PHI(i, j) = b^abs(i-j);
        end
    end

    Ar = sqrtm(phi) * randn(n/2) * sqrtm(PHI);
    Ai = sqrtm(phi) * randn(n/2) * sqrtm(PHI);
end

Abar = [Ar -Ai; Ai, Ar];

A = Abar;
y = randn(m,1);
R0 = 0;
% [R0,~,~] = aspl(A,y);
% [R0,~,~] = asplk1(A,y);
end



