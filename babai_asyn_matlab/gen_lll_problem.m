function [A, y, R0] = gen_lll_problem(k, m, n)
rng('shuffle')
% t1_avg = 0;
% t2_avg = 0;
% c1 = 0;
% c2 = 0;

if k==0
    Ar = randn(n/2);
    Ai = randn(n/2);    
else
    d = 0;
    while abs(d-1)>1e-5
        a = rand(1);
        b = rand(1);
        psi = zeros(n/2,n/2);
        phi = zeros(n/2,n/2);
        for i = 1:n/2
            for j = 1:n/2
                phi(i, j) = a^abs(i-j);
                psi(i, j) = b^abs(i-j);
            end
        end

        Ar = sqrtm(phi) * randn(n/2) * sqrtm(psi);
        Ai = sqrtm(phi) * randn(n/2) * sqrtm(psi);
        [~, ~, ~, ~, d] = qrmgs_cp([Ar -Ai; Ai, Ar], zeros(n,1));

    end
end
Abar = [Ar -Ai; Ai, Ar];
A = Abar;
y = randn(m,1);
%[~, R0, ~, ~, ~] = qrmgs_cp(A, y);
%     c2 = c2 + cond(A);
%     [R0,~,~,t2,diff] = sils_reduction(A,y);
%     t2_avg = t2_avg + t2;
% 
%     Ar = randn(n/2);
%     Ai = randn(n/2);  
%     Abar = [Ar -Ai; Ai, Ar];
% 
%     A = Abar;
%     c1 = c1 + cond(A);
%     [R0,~,~,t1,diff] = sils_reduction(A,y);
%     t1_avg = t1_avg + t1;
    
    
end




