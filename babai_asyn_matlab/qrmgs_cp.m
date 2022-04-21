function [Q, R, P, y, d] = qrmgs_cp(A, y)
[m, n] = size(A);
R = zeros(n, n);
s = zeros(2, n);
piv = 1:n;
Q = A;

for k = 1:n
    s(1, k) = norm(Q(:,k))^2;
end

for j = 1:n
    [~, l] = min(s(1,j:n)-s(2,j:n));
    l = l + j - 1;
    if l > j   
        piv([j,l]) = piv([l,j]);
        s(:,[j,l]) = s(:,[l,j]);
        Q(:,[j,l]) = Q(:,[l,j]);
        R(:,[j,l]) = R(:,[l,j]);
    end
    
    R(j,j) = norm(Q(:,j));
    Q(:,j) = Q(:,j)/R(j,j);
    for k = j+1:n
        R(j,k) = Q(:,j)'* Q(:,k);
        s(2, k) = s(2, k) + R(j,k)^2;
        Q(:,k) = Q(:,k) - Q(:,j)*R(j,k);
    end
end
d = 1;
y = Q' * y;
P = zeros(n, n);
for j = 1 : n
    P(piv(j),j) = 1;
end
% d = det(Q'*Q);
% if abs(d-1)>1e-3
%     d
%     R
%     [R_,piv,y_] = qrmcp(A,y);
%     R_
% end
end
