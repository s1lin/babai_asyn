function [Q, R, P] = qrmgs_max(A)
[~, n] = size(A);
R = zeros(n, n);
s = zeros(2, n);
P = eye(n);
Q = A;

for k = 1:n
    s(1, k) = norm(Q(:,k))^2;
end

for j = 1:n
    [~, l] = max(s(1,j:n)-s(2,j:n));
    l = l + j - 1;
    if l > j   
        P(:,[j,l]) = P(:,[l,j]);
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
end
