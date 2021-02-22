function p = pest3(left, right, predignol, noise)
    %get the estimated success probability of the ILS by lower bound
    %improved version
    m = right - left + 1;
    detR = predignol(right + 1) / predignol(left);
    mu = (abs(detR) / (pi^(m / 2) / factorial(m / 2)))^(1 / m);
    p = chi2cdf(mu^2 / noise^2/4, m);
end
