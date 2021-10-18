function [s_bar_IP, v_norm, HH, Piv] = SIC_IP(H, y, N, bound)

%Corresponds to Algorithm 2 (SIC) in Report 10

% [s_bar_IP, v_norm, HH, Piv] = SIC_IP(H, y, N, bound) applies the SIC
% initial point method
% 
% Inputs:
%     H - K-by-N real matrix 
%     y - K-dimensional real vector
%     N - integer scalar
%     bound - integer scalar for the constraint
% 
% Outputs:
%     s_bar_IP - N-dimensional integer vector for the initial point
%     v_norm - real scalar for the norm of the residual vector
%     corresponding to s_bar_IP
%     HH - K-by-N real matrix, permuted H for sub-optimal methods
%     Piv - N-by-N real matrix where HH*Piv'=H


s_bar_IP=zeros(N,1);
Piv=eye(N);
y_temp=y;
HH = H;
k = 0;
s_est = 0;
for j=N:-1:1
    max_res=inf;
    % Determine the j-th column
    for i=1:j
        s_temp_unrounded = HH(:,i)'*y_temp / (HH(:,i)' * HH(:,i));
        s_temp = round_int(s_temp_unrounded, -bound, bound);
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
        s_bar_IP(j)=s_est;
    end
    y_temp=y_temp-s_bar_IP(j)*HH(:,j);
end
v_norm = norm(y_temp);