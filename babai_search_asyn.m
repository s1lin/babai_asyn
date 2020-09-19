classdef babai_search_asyn
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here

    properties
        B, R, x0, y, n, z
    end

    methods (Static)

        function result = iif(statement1, output1, output2)

            if statement1
                result = output1;
            else
                result = output2;
            end

        end

    end

    methods

        function obj = babai_search_asyn(n)
            obj.n = n;
            %obj.z = zeros(n, 1);
            obj.z = ones(n, 1);
            obj.B = randn(n, n);
            obj.R = triu(qr(obj.B));
            obj.x0 = randi([-10, 10], n, 1);
            obj.y = obj.R * obj.x0 + 0.5 * randn(n, 1);
        end

        %Find raw x0 with parallel pooling.
        function [raw_par_x0, tEnd] = find_raw_x0_par(obj, nswp)
            x = obj.x0;
            new_x0 = zeros(obj.n, 1);
            tStart = tic;

            for iswp = 1:nswp
                m = obj.n - 1;

                parfor i = 1:m
                    k = obj.n - i;
                    x(i) = (obj.y(k) - summation(obj, k, new_x0)) / obj.R(k, k);
                    x(i) = round(x(i));
                end

                raw_par_x0 = round(x);
            end

            tEnd = toc(tStart);
        end

        %Find raw x0 in serial for loop.
        function [raw_x0, tEnd] = find_raw_x0(obj)
            raw_x0 = obj.x0;

            tStart = tic;

            for k = obj.n:-1:1
                raw_x0(k) = (obj.y(k) - obj.R(k, k + 1:obj.n) * raw_x0(k + 1:obj.n)) / obj.R(k, k);
                raw_x0(k) = round(raw_x0(k));
            end

            tEnd = toc(tStart);
        end

        %Find babai point z_b in serial time.
        function [z_b, tEnd] = babai_search_serial(obj)

            b = inf;
            k = obj.n;

            % d(k): left or right search direction at level k
            d = zeros(obj.n, 1);

            % Partial squared residual norm for z
            % prsd(k) = (norm(y(k+1:n) - R(k+1:n,k+1:n)*z(k+1:n)))^2
            prsd = zeros(obj.n, 1);

            % c(k)=(y(k)-R(k,k+1:n)*z(k+1:n))/R(k,k)
            c = zeros(k, 1);
            c(k) = obj.y(k) / obj.R(k, k);
            obj.z(k) = round(c(k));
            gamma = obj.R(k, k) * (c(k) - obj.z(k));

            % Determine enumeration direction at level n
            % iif(S, output1:S is true, output2:S is false)
            d(k) = babai_search_asyn.iif(c(k) > obj.z(k), 1, -1);

            tStart = tic;

            while 1
                % Temporary partial squared residual norm at level k
                newprsd = prsd(k) + gamma * gamma;

                if newprsd >= b

                    if k == obj.n
                        break
                    else
                        k = k + 1;
                        obj.z(k) = obj.z(k) + d(k);
                        % Update the partial squared residual norm
                        gamma = obj.R(k, k) * (c(k) - obj.z(k));
                        % Determine enumeration direction at level k+1
                        d(k) = babai_search_asyn.iif(d(k) > 0, -d(k) - 1, -d(k) + 1);
                    end

                else

                    if k ~= 1
                        k = k - 1;
                        s = summation_helper(obj, k);
                        % Find the initial integer
                        c(k) = s / obj.R(k, k);
                        obj.z(k) = round(c(k));

                        % Update the partial squared residual norm
                        prsd(k) = newprsd;
                        gamma = obj.R(k, k) * (c(k) - obj.z(k));

                        % Determine enumeration direction at level k - 1
                        d(k) = babai_search_asyn.iif(c(k) > obj.z(k), 1, -1);
                    else
                        b = newprsd;
                        z_b = obj.z;
                        obj.z(1) = obj.z(1) + d(1);
                        gamma = obj.R(1, 1) * (c(1) - obj.z(1));
                        d(k) = babai_search_asyn.iif(d(k) > 0, -d(k) - 1, -d(k) + 1);
                    end

                end

            end

            tEnd = toc(tStart);
        end

        %Find babai point z_b in parallel time.
        function [z_b, tEnd] = babai_search_par(obj)

            b = inf;
            k = obj.n;

            % d(k): left or right search direction at level k
            d = zeros(obj.n, 1);

            % Partial squared residual norm for z
            % prsd(k) = (norm(y(k+1:n) - R(k+1:n,k+1:n)*z(k+1:n)))^2
            prsd = zeros(obj.n, 1);

            % c(k)=(y(k)-R(k,k+1:n)*z(k+1:n))/R(k,k)
            c = zeros(k, 1);
            c(k) = obj.y(k) / obj.R(k, k);
            obj.z(k) = round(c(k));
            gamma = obj.R(k, k) * (c(k) - obj.z(k));

            % Determine enumeration direction at level n
            % iif(S, output1:S is true, output2:S is false)
            d(k) = babai_search_asyn.iif(c(k) > obj.z(k), 1, -1);

            tStart = tic;

            while 1
                % Temporary partial squared residual norm at level k
                newprsd = prsd(k) + gamma * gamma;

                if newprsd >= b

                    if k == obj.n
                        break
                    else
                        k = k + 1;
                        obj.z(k) = obj.z(k) + d(k);
                        % Update the partial squared residual norm
                        gamma = obj.R(k, k) * (c(k) - obj.z(k));
                        % Determine enumeration direction at level k+1
                        d(k) = babai_search_asyn.iif(d(k) > 0, -d(k) - 1, -d(k) + 1);
                    end

                else

                    if k ~= 1
                        k = k - 1;
                        s = summation_helper(obj, k);
                        % Find the initial integer
                        c(k) = s / obj.R(k, k);
                        obj.z(k) = round(c(k));

                        % Update the partial squared residual norm
                        prsd(k) = newprsd;
                        gamma = obj.R(k, k) * (c(k) - obj.z(k));

                        % Determine enumeration direction at level k - 1
                        d(k) = babai_search_asyn.iif(c(k) > obj.z(k), 1, -1);
                    else
                        b = newprsd;
                        z_b = obj.z;
                        obj.z(1) = obj.z(1) + d(1);
                        gamma = obj.R(1, 1) * (c(1) - obj.z(1));
                        d(k) = babai_search_asyn.iif(d(k) > 0, -d(k) - 1, -d(k) + 1);
                    end

                end

            end

            tEnd = toc(tStart);
        end

        %Helpers:

        % s = y(k)-R(k,k+1:n)*z(k+1:n)
        function s = summation_helper(obj, k)
            s = obj.y(k) - obj.R(k, k + 1:obj.n) * obj.z(k + 1:obj.n);
        end

        function s = summation(obj, i, new_x0)
            s = 0;

            for j = obj.n:-1:(obj.n - i + 1)
                s = obj.R(i, j) * new_x0(j);
            end

        end

    end

end
