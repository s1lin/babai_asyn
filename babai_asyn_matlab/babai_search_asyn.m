classdef babai_search_asyn

    properties
        A, R, x0, x0_R, y, n, k, SNR, z, init_res, sigma;
    end

    methods (Static)

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
        function bsa = babai_search_asyn(k, m, SNR)
            bsa.k = k;
            bsa.n = 2^m;
            bsa.SNR = SNR;
            bsa.z = ones(bsa.n, 1);
            bsa.A = randn(bsa.n, bsa.n);
            bsa.R = triu(qr(bsa.A));
            bsa.x0_R = zeros(bsa.n, 1);
            bsa.x0 = randi([0, 2^k - 1],bsa.n,1);
            bsa.sigma = sqrt(((4^k-1)*m)/(6*10^(SNR/10)));
            bsa.y = bsa.R * bsa.x0 + normrnd(0, bsa.sigma, bsa.n, 1);
            bsa.init_res = norm(bsa.y - bsa.R * bsa.x0);
            
        end

        function bsa = init_from_files(bsa)
            bsa.R = table2array(readtable(append('../data/R_', int2str(bsa.n), '.csv')));
            bsa.x0 = table2array(readtable(append('../data/x_', int2str(bsa.n), '.csv')));            
            bsa.y = table2array(readtable(append('../data/y_', int2str(bsa.n), '.csv')));
            bsa.init_res = norm(bsa.y - bsa.R * bsa.x0);
        end
        
        function bsa = write_to_nc(bsa)
            nccreate(append('../data/', int2str(bsa.n), '.nc'), 'R', 'Dimensions', {'x',bsa.n,'y',bsa.n});
            nccreate(append('../data/', int2str(bsa.n), '.nc'), 'x', 'Dimensions', {'x',bsa.n});
            nccreate(append('../data/', int2str(bsa.n), '.nc'), 'x0_R', 'Dimensions', {'x',bsa.n});
            nccreate(append('../data/', int2str(bsa.n), '.nc'), 'y', 'Dimensions', {'x',bsa.n});
            ncwrite(append('../data/', int2str(bsa.n), '.nc'),'R',bsa.R);
            ncwrite(append('../data/', int2str(bsa.n), '.nc'),'x',bsa.x0);
            ncwrite(append('../data/', int2str(bsa.n), '.nc'),'x0_R',bsa.x0_R);
            ncwrite(append('../data/', int2str(bsa.n), '.nc'),'y',bsa.y);
        end
        
        function bsa = write_to_hdf5(bsa)
            h5create(append('../data/', int2str(bsa.n), '.h5'), '/R',[bsa.n bsa.n])         
            h5write(append('../data/', int2str(bsa.n), '.h5'),'/R',bsa.R);
        end
        
        function bsa = write_to_files(bsa)
            [x_R, res, avg] = find_real_x0(bsa);
            disp([res, avg]);
            R_A = zeros(bsa.n * (bsa.n + 1)/2,1);
            index = 1;
            for i=1:bsa.n
                for j=i:bsa.n
                    if bsa.R(i,j)~=0
                        R_A(index) = bsa.R(i,j);
                        index = index + 1;
                    end
                end
            end
            writematrix(bsa.R, append('../data/R_', int2str(bsa.n), '.csv'));
            writematrix(R_A, append('../data/R_A_', int2str(bsa.n), '.csv'));
            writematrix(bsa.x0, append('../data/x_', int2str(bsa.n), '.csv'));
            writematrix(x_R, append('../data/x_R_', int2str(bsa.n), '.csv'));
            writematrix(bsa.y, append('../data/y_', int2str(bsa.n), '.csv'));
        end

        %Find raw x0 in serial for loop.
        function [raw_x0, res, avg] = find_raw_x0(bsa)
            
            for init = -1:1
                avg = 0;
                res = 0;
                for i = 1:20
                    if init ~= -1
                        raw_x0 = zeros(bsa.n, 1) + init;
                    else
                        [raw_x0, ~, ~] = find_real_x0(bsa);
                    end
                    
                    tStart = tic;
                    for j = bsa.n:-1:1
                        raw_x0(j) = (bsa.y(j) - bsa.R(j, j + 1:bsa.n) * raw_x0(j + 1:bsa.n)) / bsa.R(j, j);
                        raw_x0(j) = round(raw_x0(j));
                    end
                    tEnd = toc(tStart);
                    
                    avg = avg + tEnd;
                    res = res + norm(bsa.y - bsa.R * raw_x0);%norm(bsa.y - bsa.R * raw_x0);
                end
                disp([init, avg / 20, res / 20]);
            end
            res = res / 20;
            
        end
        
        function [x_R, res, avg] = find_real_x0(bsa)
            tStart = tic;
            for j = bsa.n:-1:1
                bsa.x0_R(j) = (bsa.y(j) - bsa.R(j, j + 1:bsa.n) * bsa.x0_R(j + 1:bsa.n)) / bsa.R(j, j);
            end
            tEnd = toc(tStart);
            x_R = round(bsa.x0_R);
            avg = tEnd;
            res = norm(bsa.x0 - x_R);            
        end
    end

end
