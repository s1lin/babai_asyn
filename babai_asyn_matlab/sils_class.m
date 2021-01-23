classdef sils_class

    properties
        A, Q, R, R_LLL, x0, x0_R, x0_R_LLL, y, y_LLL, n, k, SNR, z, init_res, init_res_LLL, sigma;
    end

    methods(Static)
        function auto_gen()
            for k = 1:2:3
                for SNR = 15:10:45
                    	m = 10;
                        s = sils_class(k, m, SNR);
                        s.write_to_nc();
                        %s.write_to_files();
                end
            end
        end
    end
    methods
        
        %Constructor
        function sils = sils_class(k, m, SNR)
            %Initialize Variables
            sils.k = k;
            sils.n = 2^m; %The real size
            sils.SNR = SNR;            
            sils.z = ones(sils.n, 1);
            sils.x0_R = zeros(sils.n, 1);
            sils.sigma = sqrt(((4^k-1)*sils.n)/(6*10^(SNR/10)));            
            
            %Initialize A:
            Ar = normrnd(0, sqrt(1/2), sils.n/2, sils.n/2);
            Ai = normrnd(0, sqrt(1/2), sils.n/2, sils.n/2);
            Abar = [Ar -Ai; Ai, Ar];
            var(Abar,0, 'all')
            sils.A = 2 * Abar;
            
            %True parameter x:
            low = -2^(k-1);
            upp = 2^(k-1) - 1;
            xr = 1 + 2 * randi([low upp], sils.n/2, 1);
            xi = 1 + 2 * randi([low upp], sils.n/2, 1);
            xbar = [xr; xi];
            sils.x0 = (2^k - 1 + xbar)./2;                    
            
            %Noise vector v:
            vr = normrnd(0, sils.sigma, sils.n/2, 1);
            vi = normrnd(0, sils.sigma, sils.n/2, 1);
            v = [vr; vi];
            sqrt(var(v))
            %Get Upper triangular matrix
            [sils.Q, sils.R] = qr(sils.A);
            
             %Right-hand side y:
            sils.y = sils.R * sils.x0 + sils.Q' * v;
            
            sils.init_res = norm(sils.y - sils.R * sils.x0)
            
            disp([sils.init_res]);
        end

        function sils = init_from_files(sils)
            sils.R = table2array(readtable(append('../data/R_', int2str(sils.n), '.csv')));
            sils.x0 = table2array(readtable(append('../data/x_', int2str(sils.n), '.csv')));
            sils.y = table2array(readtable(append('../data/y_', int2str(sils.n), '.csv')));
            sils.init_res = norm(sils.y - sils.R * sils.x0);
        end

        function sils = write_to_nc(sils)
            [x_R, res, ~] = sils_seach_round(sils);
            disp([res]);
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
          
            filename = append('../data/new', int2str(sils.n), '_', int2str(sils.SNR), '_',int2str(sils.k),'.nc');
            delete(filename);
            nccreate(filename, 'R_A', 'Dimensions', {'y',index});
            nccreate(filename, 'x_t', 'Dimensions', {'x',sils.n});
            nccreate(filename, 'y', 'Dimensions', {'x',sils.n});
            nccreate(filename, 'x_R', 'Dimensions', {'x',sils.n});
            ncwrite(filename,'R_A',R_A);
            ncwrite(filename,'x_t',sils.x0);
            ncwrite(filename,'x_R',x_R);
            ncwrite(filename,'y',sils.y);
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
            writematrix(R_A, append('../data/R_A_', int2str(sils.n), '_', int2str(sils.SNR), '_',int2str(sils.k), '.csv'));
            writematrix(sils.x0, append('../data/x_', int2str(sils.n), '_', int2str(sils.SNR), '_',int2str(sils.k), '.csv'));
            writematrix(x_R, append('../data/x_R_', int2str(sils.n), '_', int2str(sils.SNR), '_',int2str(sils.k), '.csv'));
            writematrix(sils.y, append('../data/y_', int2str(sils.n), '_', int2str(sils.SNR), '_',int2str(sils.k), '.csv'));
        end

        %Search - find the Babai solustion to the reduced problem
        function [z_B, res, tEnd] = sils_search_babai(sils, init_value)
            upper = 2^sils.k - 1;
            if init_value ~= -1
                z_B = zeros(sils.n, 1) + init_value;
            else
                [z_B, ~, ~] = find_real_x0(sils);
            end

            tStart = tic;
            for j = sils.n:-1:1
                z_B(j) = (sils.y(j) - sils.R(j, j + 1:sils.n) * z_B(j + 1:sils.n)) / sils.R(j, j);
                if(round(z_B(j)) > (2^sils.k - 1))
                    z_B(j) = upper;
                elseif (round(z_B(j)) < 0)
                    z_B(j) = 0;
                else
                    z_B(j) = round(z_B(j));
                end
            end
            tEnd = toc(tStart);
            res = norm(sils.y - sils.R * z_B);
        end

        %Search - round the real solution.
        function [x_R, res, tEnd] = sils_seach_round(sils)
            tStart = tic;
            for j = sils.n:-1:1
                sils.x0_R(j) = (sils.y(j) - sils.R(j, j + 1:sils.n) * sils.x0_R(j + 1:sils.n)) / sils.R(j, j);
            end            
            tEnd = toc(tStart);
            x_R = round(sils.x0_R);
            res = norm(sils.x0 - x_R);
        end

        %Search - rescurisive block solver.
        function x = sils_block_search2(sils, Rb, yb, x, d, l, u)
            %BOB with input partition vector d
            [~, nn] = size(Rb);
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
                if ds == nn
                    raw_x0 = zeros(nn, 1);
                    for h = nn:-1:1
                        raw_x0(h) = (yb(h) - Rb(h, h + 1:nn) * x(h + 1:nn)) / Rb(h, h);
                        raw_x0(h) = round(raw_x0(h));
                    end
                    x = raw_x0;
                    return
                else
                    q = d(1);
                    xx1 = sils.sils_block_search2(Rb(q + 1:nn, q + 1:nn), yb(q + 1:nn), x, d(2:ds), l, u);
                    yb2 = yb(1:q) - Rb(1:q, q + 1:nn) * xx1;
                    if q == 1 %Babai
                        xx2 = round(yb2 / Rb(1, 1));
                    else
                        xx2 = sils_search(Rb(1:q, 1:q), yb2, 1);
                    end
                    x = [xx2; xx1];
                end
            end   
        end
           
        %find BER
        function ber = sils_ber(sils, xt, xb)
            ber = 0;
            for i = 1:sils.n
                if xt(i) ~= xb(i)
                    ber = ber + 1;
                end
            end
            ber = ber / (sils.n * sils.k);
        end


    end

end
