function [Q, R, y] = qrmgs(A, y)
    [m, n] = size(A);
    Q = zeros(m, n);
    R = zeros(n);
    for k = 1:n
        for j = 1:k-1
            R(j,k) = Q(:,j)'*A(:,k);
            A(:,k) = A(:,k) - Q(:,j)*R(j,k);
        end
        R(k,k) = norm(A(:,k));
        Q(:,k) = A(:,k)/R(k,k);
    end
    y = Q' * y;

end
