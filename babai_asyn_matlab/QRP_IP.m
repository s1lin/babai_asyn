function [s_bar_IP, v_norm, HH, Piv] = QRP_IP(H, y, bound)

%Corresponds to Algorithm 3 (QRP) in Report 10

% [s_bar_IP, v_norm, HH, Piv] = QRP_IP(H, y, bound) applies the QRP
% initial point method
% 
% Inputs:
%     H - K-by-N real matrix 
%     y - K-dimensional real vector
%     bound - integer scalar for the constraint
% 
% Outputs:
%     s_bar_IP - N-dimensional integer vector for the initial point
%     v_norm - real scalar for the norm of the residual vector
%     corresponding to s_bar_IP
%     HH - K-by-N real matrix, permuted H for sub-optimal methods
%     Piv - N-by-N real matrix where HH*Piv'=H


[m, n] = size(H);
s_bar_IP = zeros(n, 1);

HH = H;

[~, R, y_tilde] = qrmgs_row(H, y);
Piv=eye(n);
k = 0;
x_est = 0;
for j = n:-1:(m+1)
    max_res = inf;
   for i = (m+1):j
       x_temp = round_int(y_tilde(m)/R(m, i), -bound, bound);
       res = norm(y_tilde - x_temp * R(:, i));
       if res < max_res
          k = i;
          x_est = x_temp;
          max_res = res;
       end
   end
   if k ~= 0
       HH(:,[k,j])=HH(:,[j,k]);
       R(:,[k,j])=R(:,[j,k]);
       Piv(:,[k,j])=Piv(:,[j,k]);
       s_bar_IP(j) = x_est;
       y_tilde = y_tilde - x_est * R(:,j);
   end
end

%Compute the Babai point to get the first 1:m entries of s_bar_IP
for i=m:-1:1
    if i==m
        s_temp=y_tilde(i)/(R(i,i));
    else
        s_temp=(y_tilde(i)- R(i,i+1:m)*s_bar_IP(i+1:m))/(R(i,i));
    end
    s_bar_IP(i)=round_int(s_temp, -bound, bound);
end

v_norm = norm(y_tilde - R(:,1:m) * s_bar_IP(1:m) );

