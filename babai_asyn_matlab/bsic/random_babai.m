function x = random_babai(R,y,l,u,k)
%Input:  R is upper triangular
%        y is the target REAL vector in the lattice.
%        l is the lower bound of the box
%        u is the upper bound of the box
%        k is the number of runs to obtain the optimal one
%Output: x is an estimate 

G = 10/min(abs(diag(R)))^2; % This is a parameter used in the algorithm
n = size(R,2);
x = l;
r = norm(y-R*x);
x_temp = zeros(n,1);

for j = 1:k
    for i = n:-1:1
        c = (y(i)-R(i,i+1:n)*x_temp(i+1:n))/R(i,i);        
        [domain,range] = klein_dist(G*R(i,i)^2,c,l(i),u(i));
        x_temp(i) = randsample(domain,1,true,range);
    end
    r_temp = norm(y-R*x_temp);
    if r_temp < r
        x = x_temp;
        r = r_temp;
    end
end

end