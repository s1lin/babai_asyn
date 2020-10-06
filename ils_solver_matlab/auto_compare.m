function [diff, res] = auto_compare(n, auto_gen, sils_babai_asyn, sils_search)
    [B,R,x0,y] = auto_gen(n);
    xB = sils_babai_asyn(R,y);
    %xS = sils_search(R,y,n/2);

    diff(1) = norm(x0 - xB);
    %diff(2) = norm(x0 - xS(:,n/2));
    %diff(3) = norm(xB - xS(:,n/2));
    
    res(1) = norm(R*x0- y);
    res(3) = norm(R*xB - y);
    %res(2) = norm(R*xS(:,n/2)- y);
    
    %d = [x0, xS(:, 1), xB];
    d = [x0, xB];
    display(d);