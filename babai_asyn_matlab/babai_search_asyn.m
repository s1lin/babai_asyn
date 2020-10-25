classdef babai_search_asyn

    properties
        A, R, x0, x0_R, y, n, z, init_res, CompThreads
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
            bsa.x0_R = zeros(n, 1);
            bsa.x0 = randi([-n, n], n, 1);
            bsa.y = bsa.R * bsa.x0 + 0.05 * randn(n, 1);
            bsa.init_res = norm(bsa.y - bsa.R * bsa.x0);
            bsa.CompThreads = maxNumCompThreads;
        end

        function bsa = init_from_files(bsa)
            bsa.R = table2array(readtable(append('/home/shilei/CLionProjects/babai_asyn/data/R_', int2str(bsa.n), '.csv')));
            bsa.x0 = table2array(readtable(append('/home/shilei/CLionProjects/babai_asyn/data/x_', int2str(bsa.n), '.csv')));            
            bsa.y = table2array(readtable(append('/home/shilei/CLionProjects/babai_asyn/data/y_', int2str(bsa.n), '.csv')));
            bsa.init_res = norm(bsa.y - bsa.R * bsa.x0);
        end
        
        function bsa = write_to_nc(bsa)
            nccreate(append('/home/shilei/CLionProjects/babai_asyn/data/', int2str(bsa.n), '.nc'), 'R', 'Dimensions', {'x',bsa.n,'y',bsa.n});
            nccreate(append('/home/shilei/CLionProjects/babai_asyn/data/', int2str(bsa.n), '.nc'), 'x', 'Dimensions', {'x',bsa.n});
            nccreate(append('/home/shilei/CLionProjects/babai_asyn/data/', int2str(bsa.n), '.nc'), 'x0_R', 'Dimensions', {'x',bsa.n});
            nccreate(append('/home/shilei/CLionProjects/babai_asyn/data/', int2str(bsa.n), '.nc'), 'y', 'Dimensions', {'x',bsa.n});
            ncwrite(append('/home/shilei/CLionProjects/babai_asyn/data/', int2str(bsa.n), '.nc'),'R',bsa.R);
            ncwrite(append('/home/shilei/CLionProjects/babai_asyn/data/', int2str(bsa.n), '.nc'),'x',bsa.x0);
            ncwrite(append('/home/shilei/CLionProjects/babai_asyn/data/', int2str(bsa.n), '.nc'),'x0_R',bsa.x0_R);
            ncwrite(append('/home/shilei/CLionProjects/babai_asyn/data/', int2str(bsa.n), '.nc'),'y',bsa.y);
        end
        
        function bsa = write_to_hdf5(bsa)
            h5create(append('/home/shilei/CLionProjects/babai_asyn/data/', int2str(bsa.n), '.h5'), '/R',[bsa.n bsa.n])         
            h5write(append('/home/shilei/CLionProjects/babai_asyn/data/', int2str(bsa.n), '.h5'),'/R',bsa.R);
        end
        
        function bsa = write_to_files(bsa)
            writematrix(bsa.R, append('/home/shilei/CLionProjects/babai_asyn/data/R_', int2str(bsa.n), '.csv'));
            writematrix(bsa.x0, append('/home/shilei/CLionProjects/babai_asyn/data/x_', int2str(bsa.n), '.csv'));
            writematrix(bsa.x0_R, append('/home/shilei/CLionProjects/babai_asyn/data/x_R_', int2str(bsa.n), '.csv'));
            writematrix(bsa.y, append('/home/shilei/CLionProjects/babai_asyn/data/y_', int2str(bsa.n), '.csv'));
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

        function x = deploy(bsa)
            res_tim = zeros(20, 1);
            res_res = zeros(20, 1);

            for i = 1:20
                [~, res_res(i), res_tim(i)] = find_raw_x0(bsa);
            end

            writematrix(res_tim, append('/home/shilei/CLionProjects/babai_asyn/data/Tim_', int2str(bsa.n), '.csv'));
            writematrix(res_res, append('/home/shilei/CLionProjects/babai_asyn/data/Res_', int2str(bsa.n), '.csv'));
        end

            %Find raw x0 in serial for loop.
        function [raw_x0, res, avg] = find_raw_x0(bsa, x)
            
            for init = -1:1
                avg = 0;
                res = 0;
                for i = 1:20
                    if init ~= -1
                        raw_x0 = zeros(bsa.n, 1) + init;
                    else
                        raw_x0 = x;
                    end
                    
                    tStart = tic;
                    for k = bsa.n:-1:1
                        raw_x0(k) = (bsa.y(k) - bsa.R(k, k + 1:bsa.n) * raw_x0(k + 1:bsa.n)) / bsa.R(k, k);
                        raw_x0(k) = round(raw_x0(k));
                    end
                    tEnd = toc(tStart);
                    
                    avg = avg + tEnd;
                    res = res + norm(bsa.y - bsa.R * raw_x0);
                end
                disp([init, avg / 20, res / 20]);
                
            end
        end
        
        function [x_R, res, avg] = find_real_x0(bsa)

                tStart = tic;

                for k = bsa.n:-1:1
                    bsa.x0_R(k) = (bsa.y(k) - bsa.R(k, k + 1:bsa.n) * bsa.x0_R(k + 1:bsa.n)) / bsa.R(k, k);
                end
                tEnd = toc(tStart);
                x_R = round(bsa.x0_R);
                avg = tEnd;
                res = norm(bsa.y - bsa.R *  x_R);            
        end
    end

end
