function p = pest(R, noise)
    %get the estimated success probability of the ILS by lower bound
    [n, m] = size(R);

    if n == 1
        p = erf(abs(R) / 2 / sqrt(2) / noise);
    else
        mu = (abs(det(R)) / (pi^(m / 2) / factorial(round(m / 2))))^(1 / m);
        p = chi2cdf(mu^2 / noise^2/4, m);

    end
end