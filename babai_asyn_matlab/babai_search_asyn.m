classdef babai_search_asyn

    properties
        A, R, x0, y, n, z, init_res, CompThreads
    end

    methods (Static)

        function result = iif(statement1, output1, output2)

            if statement1
                result = output1;
            else
                result = output2;
            end

        end

        function [residual, time, raw_x0, raw_x0p, j] = auto_compare(n, np, nswp)
            %Init babai_search_asyn object, and results.
            bsa = babai_search_asyn(n);
            residual = zeros(3, 1);

            %Run all the functions
            residual(1) = bsa.init_res;
            [raw_x0, residual(2), t1] = find_raw_x0(bsa);
            [raw_x0p, j, residual(3), t2] = find_raw_x0_par(bsa, np, nswp);

            %Compute the results
            time = t1 - t2;
        end

    end

    methods
        %Constructor
        function bsa = babai_search_asyn(n)
            bsa.n = n;
            bsa.z = ones(n, 1);
            bsa.A = randn(n, n);
            bsa.R = triu(qr(bsa.A));
            bsa.x0 = randi([-n, n], n, 1);
            bsa.y = bsa.R * bsa.x0 + 0.1 * randn(n, 1);
            bsa.init_res = norm(bsa.y - bsa.R * bsa.x0);
            bsa.CompThreads = maxNumCompThreads;
        end
        
        function bsa = init_from_files(bsa)
            bsa.R = table2array(readtable('/home/shilei/CLionProjects/babai_asyn/babai_asyn_cuda/cmake-build-debug/R_2048.csv'));
            bsa.x0 =  table2array(readtable('/home/shilei/CLionProjects/babai_asyn/babai_asyn_cuda/cmake-build-debug/x_2048.csv'));
            bsa.y = table2array(readtable('/home/shilei/CLionProjects/babai_asyn/babai_asyn_cuda/cmake-build-debug/y_2048.csv'));
            bsa.init_res = norm(bsa.y - bsa.R * bsa.x0);
        end
        
        function bsa = write_to_files(bsa)
            writematrix(bsa.R, '/home/shilei/CLionProjects/babai_asyn/babai_asyn_cuda/cmake-build-debug/R_2048.csv');
            writematrix(bsa.x0, '/home/shilei/CLionProjects/babai_asyn/babai_asyn_cuda/cmake-build-debug/x_2048.csv');
            writematrix(bsa.y, '/home/shilei/CLionProjects/babai_asyn/babai_asyn_cuda/cmake-build-debug/y_2048.csv');
        end

        %Find raw x0 with parallel pooling CPU ONLY for now.
        function [x, j, res, tEnd] = find_raw_x0_par(bsa, np, nswp)
            p = parpool(np);
            [~, tol, ~] = find_raw_x0(bsa);
            D = 1 ./ diag(bsa.R, 0);
            B = eye(bsa.n) - bsa.R ./ diag(bsa.R, 0);
            C = D .* bsa.y;
            raw_x0 = bsa.x0;
            tStart = tic;            
            spmd(np)
                vet = codistributor1d(1, codistributor1d.unsetPartition, [bsa.n, 1]);
                mat = codistributor1d(1, codistributor1d.unsetPartition, [bsa.n, bsa.n]);

                Rm = codistributed(bsa.R, mat);
                yv = codistributed(bsa.y, vet);
                x = codistributed(Rm * raw_x0 - yv, vet);

                j = 1;
                tol_x = tol;
                %norm(x - raw_x0, Inf) > norm(raw_x0, Inf) * TOLX && 
                while (j < nswp)

                    if (tol * norm(x, Inf) > realmin)
                        tol_x = norm(x, Inf) * tol;
                    else
                        tol_x = realmin;
                    end
                    disp(tol_x)
                    
                    %raw_x0 = x;
                    for k = bsa.n:-1:1
                        x(k) = (yv(k) - Rm(k, k + 1:bsa.n) * x(k + 1:bsa.n)) / Rm(k, k);
                        x(k) = round(x(k));
                    end
%                     raw_x0 = x;
%                     x = round(yv - Rm*x.*D);
                    %x = round(B * raw_x0 + C);
                    j = j + 1;
                end
            end
            tEnd = toc(tStart);
            j = j{1};
            x = gather(x);
            %disp([t1, tEnd - tStart]);
            %disp(x)
            res = norm(bsa.y - bsa.R * x);
            disp([res tol]);
            delete(p);
        end

        function x = deploy(bsa, x)
        %s = (b(i) - bsa.R(i, i + 1:bsa.bsa.n) * x(i + 1:bsa.n, j)) / bsa.R(i, i);

        %             disp('   task  i  j  x_prev   x_next');
        %             %task = get(getCurrentTask(), 'ID');
        %             %i = m - i + 1;
        %             disp([i, j])
        %             %if i == u(j)
        %             s = bsa.R(i, i + 1:bsa.bsa.n) * x(i + 1:bsa.bsa.n, j);
        %             x(i, j + 1) = round((b(i) - s) / bsa.R(i, i));
        %             %k_next(i) = k(i) - 1;
        %             %disp([task, i, j, x(i, j), x(i, j+1)])
        %             %else
        %             %x(i, j + 1) = x(i, j);
        %             %end
        %             %end
            for k = bsa.n:-1:1
                    x(k) = (bsa.y(k) - bsa.R(k, k + 1:bsa.bsa.n) * x(k + 1:bsa.bsa.n)) / bsa.R(k, k);
                    x(k) = round(x(k));
            end

        end

            %Find raw x0 in serial for loop.
        function [raw_x0, res, tEnd] = find_raw_x0(bsa)
            raw_x0 = zeros(bsa.n,1);

            tStart = tic;
            for k = bsa.n:-1:1
                raw_x0(k) = (bsa.y(k) - bsa.R(k, k + 1:bsa.n) * raw_x0(k + 1:bsa.n)) / bsa.R(k, k);
                raw_x0(k) = round(raw_x0(k));
            end
            tEnd = toc(tStart);

            res = norm(bsa.x0 - raw_x0);   
            
        end

    end

end
