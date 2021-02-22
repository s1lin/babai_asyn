function x = sils_block_search(Rb, yb, x, d, u)
    %BOB with input partition vector d
    [~, n] = size(Rb);
    [ds, ~] = size(d);
    %1 dimension
    if ds == 1
        if d == 1
            x = round(yb / Rb);
            return
        else
            x = obils_search(Rb, yb, 0, u);
            
            return
        end

    else
        %Babai
        if ds == n
            raw_x0 = zeros(n, 1);
            for k = n:-1:1
                raw_x0(k) = (yb(k) - Rb(k, k + 1:n) * x(k + 1:n)) / Rb(k, k);
                raw_x0(k) = round(raw_x0(k));
            end
            x = raw_x0;
            return
        else
            q = d(1);
            xx1 = sils_block_search(Rb(q + 1:n, q + 1:n), yb(q + 1:n), x, d(2:ds), u);
            yb2 = yb(1:q) - Rb(1:q, q + 1:n) * xx1;
            if q == 1 %Babai
                xx2 = round(yb2 / Rb(1, 1));
            else
                xx2 = obils_search(Rb(1:q, 1:q), yb2, 0, u);
            end
            x = [xx2; xx1];
        end
    end   
end
