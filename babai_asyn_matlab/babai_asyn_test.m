function [raw_x0, res, tEnd] = find_raw_x0(bsa)
    raw_x0 = bsa.x0;
    tStart = tic;

    for k = bsa.n:-1:1
        raw_x0(k) = (bsa.y(k) - bsa.R(k, k + 1:bsa.n) * raw_x0(k + 1:bsa.n)) / bsa.R(k, k);
        raw_x0(k) = round(raw_x0(k));
    end

    res = norm(bsa.y - bsa.R * raw_x0);
    tEnd = toc(tStart);
end