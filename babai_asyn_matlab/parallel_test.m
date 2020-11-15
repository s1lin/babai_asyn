function parallel_test(A, n)
    Acpu = A(:, :, 1:n / 2); %chunk #1 : send to CPU
    Agpu = gpuArray(A(:, :, n / 2 + 1:end)); %chunk #2 : send to GPU with device index 1
    p = parpool(2);
    F(1) = parFeval(p, @deploy, 2, Acpu);
    F(2) = parFeval(p, @deploy, 2, Agpu, 1);
    [Q, R] = fetchOutputs(F, 'UniformOutput', false); % Blocks until complete
    Q = cat(3, Q{1}, gather(Q{2}))
    R = cat(3, R{1}, gather(R{2}))
end

function [q, r] = deploy(a, Id)
    if nargin > 1, gpuDevice(Id); end

    for i = size(a, 3):-1:1
        [q(:, :, i), r(:, :, i)] = qr(A(:, :, i));
    end

end


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
        
        %Find raw x0 with parallel pooling CPU ONLY for now.
        function x = deploy(bsa)
            res_tim = zeros(20, 1);
            res_res = zeros(20, 1);

            for i = 1:20
                [~, res_res(i), res_tim(i)] = find_raw_x0(bsa);
            end

            writematrix(res_tim, append('../data/Tim_', int2str(bsa.n), '.csv'));
            writematrix(res_res, append('../data/Res_', int2str(bsa.n), '.csv'));
        end