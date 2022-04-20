function [Q, R_, y_q] = qrmgs_row(A, y)
    A = [A y];
    [m, n] = size(A);
    Q = zeros(m, m);
    R = zeros(m, n);

    for j = 1:m
        for k = j:n
            R(j,k) = Q(:,j)'*A(:,k);
            A(:,k) = A(:,k) - Q(:,j)*R(j,k);
            if k == j
                R(k,k) = norm(A(:,k));
                Q(:,k) = A(:,k)/R(k,k);
            end
        end
    end
    R_  = R(:,1:n-1);
    y_q = R(:, n);
end
