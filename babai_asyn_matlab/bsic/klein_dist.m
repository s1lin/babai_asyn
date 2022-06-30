function [domain,range] = klein_dist(g,c,bl,bu)
%Input: g is the parameter controlling the distribution. As g->0 the
%distribution tends to a discrete uniform.
%       c is the real-valued input to be rounded.
%       bl and bu are the lower and upper ends of the constraint interval
%
%Output: domain - candidates with probability > 0 (in MATLAB)
%        range - probabilities of each element in the domain


domain = zeros(1,bu-bl+1);
range = zeros(1,bu-bl+1);
index=0;
for i=bl:bu
    index = index+1;
    s = 0;
    for j = bl:bu
        s = s+exp(-g*(2*c-j-i)*(i-j));
    end
    prob = 1/s;
    domain(index) = i;
    range(index) = prob;
end

end