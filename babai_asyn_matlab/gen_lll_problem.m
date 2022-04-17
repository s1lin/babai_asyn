function [A, y, R0] = gen_lll_problem(k, m, n)
rng('shuffle')
if k==0
    Ar = randn(n/2);
    Ai = randn(n/2);    
else
    phi = rand(n/2);
    PHI = rand(n/2);
    for i = 1:n/2
        for j = 1:n/2
            phi(i, j) = phi(i, j)^abs(i-j);
            PHI(i, j) = PHI(i, j)^abs(i-j);
        end
    end
    Ar = sqrt(phi) * randn(n/2) * sqrt(PHI);
    Ai = sqrt(phi) * randn(n/2) * sqrt(PHI);
end
Abar = [Ar -Ai; Ai, Ar];
A = Abar;
y = randn(m,1);
R0 = 0;
% [R0,~,~] = aspl(A,y);
% [R0,~,~] = asplk1(A,y);
end



