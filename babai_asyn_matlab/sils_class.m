classdef sils_class

    properties
        A, Q, R, R_LLL, x0, x0_R, x0_R_LLL, y, y_LLL, n, k, SNR, z, init_res, init_res_LLL, sigma;
    end

    methods(Static)
        function auto_gen()
            for k = 1:3
                for SNR = 15:20:45
                    for m = 4:5
                        s = sils_class(k, m, SNR);
                        s.write_to_nc();
                    end
                end
            end
        end
    end
    methods
        %Constructor
        function sils = sils_class(k, m, SNR)
            sils.k = k;
            sils.n = 2^m;
            sils.SNR = SNR;
            sils.z = ones(sils.n, 1);
            sils.A = normrnd(0, sqrt(.5), sils.n, sils.n);
            [sils.Q, sils.R] = qr(sils.A);
            sils.x0_R = zeros(sils.n, 1);
            sils.x0_R_LLL = zeros(sils.n, 1);
            sils.x0 = randi([0, 2^k - 1],sils.n,1);
            sils.sigma = sqrt(((4^k-1)*m)/(6*10^(SNR/20)));
            v = normrnd(0, sils.sigma, sils.n, 1);
            sils.y = sils.A * sils.x0 + v;
            [sils.R_LLL,~,sils.y_LLL] = sils_reduction(sils.A, sils.y);
            sils.y = sils.Q' * sils.y;
            sils.init_res = norm(sils.y - sils.R * sils.x0);
            sils.init_res_LLL = norm(sils.y_LLL - sils.R_LLL * sils.x0);
            disp([sils.init_res, sils.init_res_LLL]);
        end

        function sils = init_from_files(sils)
            sils.R = table2array(readtable(append('../data/R_', int2str(sils.n), '.csv')));
            sils.x0 = table2array(readtable(append('../data/x_', int2str(sils.n), '.csv')));
            sils.y = table2array(readtable(append('../data/y_', int2str(sils.n), '.csv')));
            sils.init_res = norm(sils.y - sils.R * sils.x0);
        end

        function sils = write_to_nc(sils)
            [x_R, x_R_LLL, res, res_LLL, ~] = sils_seach_round(sils);
            disp([res, res_LLL]);
            R_A = zeros(sils.n * (sils.n + 1)/2,1);
            index = 1;
            for i=1:sils.n
                for j=i:sils.n
                    if sils.R(i,j)~=0
                        R_A(index) = sils.R(i,j);
                        index = index + 1;
                    end
                end
            end
            R_A_LLL = zeros(sils.n * (sils.n + 1)/2,1);
            index = 1;
            for i=1:sils.n
                for j=i:sils.n
                    if sils.R(i,j)~=0
                        R_A_LLL(index) = sils.R_LLL(i,j);
                        index = index + 1;
                    end
                end
            end
            filename = append('../data/new', int2str(sils.n), '_', int2str(sils.SNR), '_',int2str(sils.k),'.nc');
            nccreate(filename, 'R_A', 'Dimensions', {'y',index});
            nccreate(filename, 'R_A_LLL', 'Dimensions', {'y',index});
            nccreate(filename, 'x_t', 'Dimensions', {'x',sils.n});
            nccreate(filename, 'y', 'Dimensions', {'x',sils.n});
            nccreate(filename, 'x_R', 'Dimensions', {'x',sils.n});
            nccreate(filename, 'x_R_LLL', 'Dimensions', {'x',sils.n});
            nccreate(filename, 'y_LLL', 'Dimensions', {'x',sils.n});
            ncwrite(filename,'R_A',R_A);
            ncwrite(filename,'R_A_LLL',R_A_LLL);
            ncwrite(filename,'x_t',sils.x0);
            ncwrite(filename,'x_R',x_R);
            ncwrite(filename,'x_R_LLL',x_R_LLL);
            ncwrite(filename,'y',sils.y);
            ncwrite(filename,'y_LLL',sils.y_LLL);
        end

        function sils = write_to_files(sils)
            [x_R, res, avg] = sils_seach_round(sils);
            disp([res, avg]);
            R_A = zeros(sils.n * (sils.n + 1)/2,1);
            index = 1;
            for i=1:sils.n
                for j=i:sils.n
                    if sils.R(i,j)~=0
                        R_A(index) = sils.R(i,j);
                        index = index + 1;
                    end
                end
            end
            writematrix(sils.R, append('../data/R_', int2str(sils.n), '.csv'));
            writematrix(R_A, append('../data/R_A_', int2str(sils.n), '.csv'));
            writematrix(sils.x0, append('../data/x_', int2str(sils.n), '.csv'));
            writematrix(x_R, append('../data/x_R_', int2str(sils.n), '.csv'));
            writematrix(sils.y, append('../data/y_', int2str(sils.n), '.csv'));
        end

        %Search - find the Babai solustion to the reduced problem
        function [z_B, res, tEnd] = sils_search_babai(sils, init_value)

            if init_value ~= -1
                z_B = zeros(sils.n, 1) + init_value;
            else
                [z_B, ~, ~] = find_real_x0(sils);
            end

            tStart = tic;
            for j = sils.n:-1:1
                z_B(j) = (sils.y(j) - sils.R(j, j + 1:sils.n) * z_B(j + 1:sils.n)) / sils.R(j, j);
                z_B(j) = round(z_B(j));
            end
            tEnd = toc(tStart);
            res = norm(sils.y - sils.R * z_B);
        end

        %Search - round the real solution.
        function [x_R, x_R_LLL, res, res_LLL, avg] = sils_seach_round(sils)
            tStart = tic;
            for j = sils.n:-1:1
                sils.x0_R(j) = (sils.y(j) - sils.R(j, j + 1:sils.n) * sils.x0_R(j + 1:sils.n)) / sils.R(j, j);
                sils.x0_R_LLL(j) = (sils.y_LLL(j) - sils.R_LLL(j, j + 1:sils.n) * sils.x0_R_LLL(j + 1:sils.n)) / sils.R_LLL(j, j);
            end
            
            tEnd = toc(tStart);
            x_R = round(sils.x0_R);
            x_R_LLL = round(sils.x0_R_LLL);
            avg = tEnd;
            res = norm(sils.x0 - x_R);
            res_LLL = norm(sils.x0 - x_R_LLL);
        end

         %Search - rescurisive block solver.
        function [x, res] = sils_block_search(sils, Rb, yb, x, d, init_value)
            %BOB with input partition vector d
            [~, l] = size(Rb);
            [ds, ~] = size(d);
            %1 dimension
            if ds == 1
                if d == 1
                    x = round(yb / Rb);
                    return
                else
                    x = sils_search(Rb, yb, 1);
                    return
                end

            else
                %Babai
                if ds == l
                    [x, res] = sils.sils_search_babai(sils, init_value);
                    return
                else
                    q = d(1);
                    xx1 = sils_block_search(Rb(q + 1:l, q + 1:l), yb(q + 1:l), x, d(2:ds));
                    yb2 = yb(1:q) - Rb(1:q, q + 1:l) * xx1;
                    if q == 1%Babai
                        xx2 = round(yb2 / Rb(1, 1));
                    else
                        Rb(1:q, 1:q)
                        xx2 = sils_search(Rb(1:q, 1:q), yb2, 1);
                    end
                    x = [xx2; xx1];
                end
            end
            res = norm(yb - Rb * x);

        end



    end

end
