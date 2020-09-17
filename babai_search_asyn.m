classdef babai_search_asyn
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here

    properties
        B, R, x0, y, n
    end

    methods

        function obj = babai_search_asyn(n)
            obj.n = n;
            obj.B = randn(n, n);
            obj.R = triu(qr(obj.B));
            obj.x0 = randi([-10, 10], n, 1);
            obj.y = obj.R * obj.x0 + randn(n, 1);
        end

        %Test parfor
        function new_x0 = gen_x0(obj, nswp)
            x = obj.x0;
            new_x0 = zeros(obj.n, 1);

            for iswp = 1:nswp
                m = obj.n - 1;

                parfor i = 1:m
                    k = obj.n - i;
                    x(i) = (obj.y(k) - summation(obj, k, new_x0)) / obj.R(k, k);
                    x(i) = round(x(i));
                end

                new_x0 = round(x);
            end

        end

        function real_x0 = find_x0(obj)
            real_x0 = obj.x0;

            for k = obj.n:-1:1
                real_x0(k) = (obj.y(k) - obj.R(k, k + 1:obj.n) * real_x0(k + 1:obj.n)) / obj.R(k, k);
                real_x0(k) = round(real_x0(k));
            end

        end

        function ZBhat = sils_babai_asyn(obj)

            z = zeros(obj.n, 1);

            % c(k)=(y(k)-R(k,k+1:n)*z(k+1:n))/R(k,k)
            c = zeros(obj.n, 1);

            % d(k): left or right search direction at level k
            d = zeros(obj.n, 1);

            % Partial squared residual norm for z
            % prsd(k) = (norm(y(k+1:n) - R(k+1:n,k+1:n)*z(k+1:n)))^2
            prsd = zeros(obj.n, 1);

            % Store some quantities for efficiently calculating c
            % S(k,n) = y(k),
            % S(k,j-1) = y(k) - R(k,j:n)*z(j:n) = S(k,j) - R(k,j)*z(j), j=k+1:n
            S = zeros(obj.n, obj.n);
            S(:, obj.n) = obj.y;

            % path(k): record information for updating S(k,k:path(k)-1)
            path = obj.n * ones(obj.n, 1);

            % The level at which search starts to move up to a higher level
            ulevel = 0;

            % The p candidate solutions (or points)
            ZBhat = zeros(obj.n, 1);

            % Initial squared search radius
            beta = inf;
            c(obj.n) = obj.y(obj.n) / obj.R(obj.n, obj.n);
            z(obj.n) = round(c(obj.n));
            gamma = obj.R(obj.n, obj.n) * (c(obj.n) - z(obj.n));
            % Determine enumeration direction at level n
            d(obj.n) = -1;

            if c(obj.n) > z(obj.n)
                d(obj.n) = -1;
            end

            k = obj.n;

            while 1
                % Temporary partial squared residual norm at level k
                newprsd = prsd(k) + gamma * gamma;

                if newprsd < beta

                    if k ~= 1% move to level k-1
                        % Update path
                        if ulevel ~= 0
                            path(ulevel:k - 1) = k;

                            for j = ulevel - 1:-1:1

                                if path(j) < k
                                    path(j) = k;
                                else
                                    break; % Note path(1:j-1) >= path(j)
                                end

                            end

                        end

                        % Update S
                        k = k - 1;

                        for j = path(k):-1:k + 1
                            S(k, j - 1) = S(k, j) - obj.R(k, j) * z(j);
                        end

                        % Update the partial squared residual norm
                        prsd(k) = newprsd;

                        % Find the initial integer
                        c(k) = S(k, k) / obj.R(k, k);
                        z(k) = round(c(k));
                        gamma = obj.R(k, k) * (c(k) - z(k));

                        if c(k) > z(k)
                            d(k) = 1;
                        else
                            d(k) = -1;
                        end

                        ulevel = 0;

                    else % The babai point is found, update the set of candidate solutions

                        ZBhat = z;
                        beta = newprsd;
                        z(1) = z(1) + d(1);
                        gamma = obj.R(1, 1) * (c(1) - z(1));

                        if d(1) > 0
                            d(1) = -d(1) - 1;
                        else
                            d(1) = -d(1) + 1;
                        end

                    end

                else

                    if k == obj.n% The Babai point has been found
                        break
                    else % Move back to level k+1

                        if ulevel == 0
                            ulevel = k;
                        end

                        k = k + 1;
                        % Find a new integer at level k
                        z(k) = z(k) + d(k);
                        gamma = obj.R(k, k) * (c(k) - z(k));

                        if d(k) > 0
                            d(k) = -d(k) - 1;
                        else
                            d(k) = -d(k) + 1;
                        end

                    end

                end

            end

        end

        function s = summation(obj, i, new_x0)
            s = 0;

            for j = obj.n:-1:(obj.n - i + 1)
                s = obj.R(i, j) * new_x0(j);
            end

        end

    end

end
