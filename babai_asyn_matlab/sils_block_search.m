function x = sils_block_search(Rb, yb, x, d, l, u)
    %BOB with input partition vector d
    [~, nn] = size(Rb);
    [ds, ~] = size(d);
    %1 dimension
    if ds == 1
        if d == 1
            x = round(yb / Rb);
            return
        else
            tic
            x = sils_search(Rb, yb, 1);
            toc
            return
        end

    else
        %Babai
        if ds == nn
            raw_x0 = zeros(nn, 1);
            for h = nn:-1:1
                raw_x0(h) = (yb(h) - Rb(h, h + 1:nn) * x(h + 1:nn)) / Rb(h, h);
                raw_x0(h) = round(raw_x0(h));
            end
            x = raw_x0;
            return
        else
            q = d(1);
            xx1 = sils_block_search(Rb(q + 1:nn, q + 1:nn), yb(q + 1:nn), x, d(2:ds), l, u);
            yb2 = yb(1:q) - Rb(1:q, q + 1:nn) * xx1;
            if q == 1 %Babai
                xx2 = round(yb2 / Rb(1, 1));
            else
                tic
                xx2 = sils_search(Rb(1:q, 1:q), yb2, 1);
                toc
            end
            x = [xx2; xx1];
        end
    end   
end
