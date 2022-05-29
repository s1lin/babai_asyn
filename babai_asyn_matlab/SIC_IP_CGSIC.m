function [x, v_norm, HH, Piv] = CGSIC(H, y, bound)

%Corresponds to Algorithm 2 (SIC) in Report 10

% [x, v_norm, HH, Piv] = SIC_IP(H, y, N, bound) applies the SIC
% initial point method
% 
% Inputs:
%     H - K-by-N real matrix 
%     y - K-dimensional real vector
%     N - integer scalar
%     bound - integer scalar for the constraint
% 
% Outputs:
%     x - N-dimensional integer vector for the initial point
%     v_norm - real scalar for the norm of the residual vector
%     corresponding to x
%     HH - K-by-N real matrix, permuted H for sub-optimal methods
%     Piv - N-by-N real matrix where HH*Piv'=H

[K, N] = size(H);
x=zeros(N,1);
Piv=eye(N);
C = H' * H;
b = H' * y;
rho = norm(y)^2;
for j=N:-1:1
    max_res=inf;
    % Determine the j-th column
    for i=1:j
        s_temp_unrounded = HH(:,i)'*y_temp / (HH(:,i)' * HH(:,i));
        s_temp = round_int(s_temp_unrounded, 0, bound);
        res = norm(y_temp-s_temp*HH(:,i));
        if res < max_res
            k=i;
            s_est=s_temp;
            max_res = res;
        end
    end
    if k~=0
        HH(:,[k,j])=HH(:,[j,k]);
        Piv(:,[k,j])=Piv(:,[j,k]);    
        x(j)=s_est;
    end
    y_temp=y_temp-x(j)*HH(:,j);
    %y_temp'
end
v_norm = norm(y_temp);