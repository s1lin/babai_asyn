function [zhat,flag,ratio_sub_beta_beta]= ubils_search_lowerbound_parallel_simple_gen(R,y,l,u,z0,beta,duetime,noise_var,lower_bound_levels)
%
% [zhat flag] = ubils_search(R,y,l,u,z0,beta,duetime) produces the optimal
% solution or a suboptimal solution to the upper triangular box-constrained
% integer least squares problem min_{z}||y-Rz|| s.t. z in [l, u] by
% a search algorithm.

% Inputs:
%    R - m by n real upper triapezoidal matrix
%    y - m-dimensional real vector
%    l - n-dimensional integer vector, lower bound
%    u - n-dimensional integer vector, upper bound
%    z0- n-dimensional initial integer point found in the reduction process
%    beta - initial search radius
%    duetime - time limit (a time instance) for the search process
%
% Outputs:
%    zhat - n-dimensional integer vector (in double precision).
%    flag - flag = 0, zhat is the optiaml solution
%           flag = 1, zhat is a suboptimal solution

% Subfunctions (included in this file):
%    shift, initialSearch, bound, init, update

% Main References:
% [1] X.-W. Chang and X. Yang. An efficient tree search
%     decoder with column reordering for underdetermined MIMO systems.
%     Proceedings of IEEE GLOBECOM 2007, 5 pages.
% [2] Jing Zhu. Numerical methods for underdetermined box-constrained
%     integer least squares problems, M.Sc. Thesis, School of Computer
%     Science, McGill University, 2016.

% Authors: Xiao-Wen Chang, www.cs.mcgill.ca/~chang
%          Xiangyu Ren, Zhilong Chen
% Copyright (c) 2019. Scientific Computing Lab, McGill University.
% Last revision: June 2019

tic;

% ------------------------------------------------------------------
% --------  Initialization  ----------------------------------------
% ------------------------------------------------------------------
[m,n] = size(R);

% Current point
z = [];
zhat = [];

% c(k) = (y(k)-R(k,k+1:n)*z(k+1:n))/R(k,k)
c = zeros(n,1);

% Store some quantities for efficiently calculating c
% S(k,n) = y(k),
% S(k,j-1) = y(k) - R(k,j:n)*z(j:n) = S(k,j) - R(k,j)*z(j), j=k+1:n
S = zeros(n,n);

% lflag(k) = 1 if the lower bound is reached at level k
lflag = zeros(size(l));
% uflag(k) = 1 if the upper bound is reached at level k
uflag = lflag;

% Partial squared residual norm for z
% prsd(k) = (norm(y(k+1:n)-R(k+1:n,k+1:n)*z(k+1:n)))^2
prsd = zeros(n,1);

% path(k): record information for updating S(k,k:path(k)-1)
path = n * ones(n,1);

% The level at which search starts to move up to a higher level
ulevel = 0;


% ------------------------------------------------------------------
% --------  Search process  ----------------------------------------
% ------------------------------------------------------------------

% Shift variables
[R,y,u,l,col_neg,s,new_z0] = shift(R,y,l,u,z0);


[sub_beta_square,~,~,~] = sub_residual_admm(R(1:m-1,:),y(1:m-1,:),l,u,noise_var);
ratio_sub_beta_beta = sqrt(max(0,sub_beta_square/beta^2));
% if R(m,n)~=0
% l(n) = ceil(max(l(n),y(m,1) - (sqrt(max(beta^2-sub_beta_square,0))+R(m,m:n-1)*u(m:n-1))/R(m,n)));
% u(n) = floor(min(u(n),y(m,1) + sqrt(max(beta^2-sub_beta_square,0))/R(m,n)));
% end

l_original = l;
u_original = u;

unacceptable_zn = zeros(u(n)+1,1);
unacceptable_number = 0;
if lower_bound_levels>=1
    parfor i=0:u(n)
        [lowerbound_for_first_n_1_cols,~,~,~]  = lowerbound_for_first_x_cols(R(1:end,1:n-1),y-i*R(1:end,n),l(1:n-1,1),u(1:n-1,1),noise_var,beta^2);
        
        if lowerbound_for_first_n_1_cols > beta^2
            unacceptable_zn(i+1,1) = 1;
            unacceptable_number = unacceptable_number + 1;
        end
    end
    if unacceptable_number*2<(u(n)-l(n)+1)
        lower_bound_levels = 0;
    end
end

utemp = u;

% S(k,n) = y(k), k=1:m-1, S(k,n) = y(m), k=m:n
S(m:n,n) = y(m);
S(1:m-1,n) = y(1:m-1);

% Choose search order

% Initialization
interval_length = u(n) - l(n);
if interval_length<0
    zhat = z0;
    flag = 0;
    return
end




all_dis = zeros(interval_length+1,1);
all_dis_zn_z0n = zeros(interval_length+1,1);
%results = cell(interval_length+1,1);
results1 = ones(interval_length+1,n,1);
results2 = ones(interval_length+1,n,1);
results3 = ones(interval_length+1,n,1);
results4 = ones(interval_length+1,n,1);
results5 = ones(interval_length+1,1,1);
results6 = ones(interval_length+1,n,1);
results7 = ones(interval_length+1,n,n);
results8 = ones(interval_length+1,n,1);
results9 = ones(interval_length+1,n,1);
results10 = ones(interval_length+1,n,1);
results11 = ones(interval_length+1,m+1,1);
results12 = ones(interval_length+1,n,interval_length+1);

visit_order = zeros(interval_length+1,1);


%sub_beta = sub_residual_admm(R(1:m-1,:),y(1:m-1,1),l,u);

% Preprocessing

for i = 1:interval_length+1
    if unacceptable_zn(i,1) == 1
        all_dis(i,1) = Inf;
        all_dis_zn_z0n(i,1) = Inf;
        continue
    end
    copy_c = c;
    copy_c(n,1) = i - 1;
    [dis,branch_info] = initialSearch(S,R,copy_c,l,u,beta,y,sub_beta_square,l_original,u_original,noise_var,lower_bound_levels);
    
    if size(branch_info,1) == 0
        all_dis(i,1) = Inf;
        all_dis_zn_z0n(i,1) = Inf;
    else
        all_dis(i,1) = dis;
        if dis == Inf
            all_dis_zn_z0n(i,1) = Inf;
        else
            all_dis_zn_z0n(i,1) = abs(new_z0(n,1)-i+1);
        end
        
        results1(i,:,:) = branch_info{1};
        results2(i,:,:) = branch_info{2};
        results3(i,:,:) = branch_info{3};
        results4(i,:,:) = branch_info{4};
        results5(i,:,:) = branch_info{5};
        results6(i,:,:) = branch_info{6};
        results7(i,:,:) = branch_info{7};
        results8(i,:,:) = branch_info{8};
        results9(i,:,:) = branch_info{9};
        results10(i,:,:) = branch_info{10};
        results11(i,:,:) = branch_info{11};
        results12(i,:,:) = branch_info{12};
        
    end
    
end
% Find visiting order
cur_length = 1;

% Check if the cell aray {results} is empty
if min(all_dis) == Inf
    zhat = z0;
    flag = 0;
    return
end

% Sort the order
while cur_length <= interval_length+1
    cur_min = min(all_dis);
    if(cur_min ~= Inf)
        indexes = find(all_dis == cur_min);
        for j = 1:1:size(indexes,1)
            cur_index = indexes(j,1);
            visit_order(cur_length) = cur_index;
            cur_length = cur_length + 1;
            all_dis(cur_index,1) = Inf;
        end
    else
        break;
    end
end


% while cur_length <= interval_length+1
%     cur_min = min(all_dis_zn_z0n);
%     if(cur_min ~= Inf)
%         indexes = find(all_dis_zn_z0n == cur_min);
%         for j = 1:1:size(indexes,1)
%             cur_index = indexes(j,1);
%             visit_order(cur_length) = cur_index;
%             cur_length = cur_length + 1;
%             all_dis_zn_z0n(cur_index,1) = Inf;
%         end
%     else
%         break;
%     end
% end




% Set up the first branch to be searched
cur_index = 1;


cur1 = results1(visit_order(cur_index),:,:)';
cur2 = results2(visit_order(cur_index),:,:)';
cur3 = results3(visit_order(cur_index),:,:)';
cur4 = results4(visit_order(cur_index),:,:)';
cur5 = results5(visit_order(cur_index),:,:);
cur6 = results6(visit_order(cur_index),:,:)';
cur7 = reshape(results7(visit_order(cur_index),:,:),n,n);
cur8 = results8(visit_order(cur_index),:,:)';
cur9 = results9(visit_order(cur_index),:,:)';
cur10 = results10(visit_order(cur_index),:,:)';
cur11 = results11(visit_order(cur_index),:,:)';
cur12 = reshape(results12(visit_order(cur_index),:,:),n,interval_length+1);


l = cur1;u = cur2;
lflag = cur3;      % lflag(k) = 1 if lower bound is reached at level k
uflag = cur4;      % uflag(k) = 1 if upper bound is reached at level k
ulevel = cur5;     % The level where search moves up to a higher level
path = cur6;       % Record information for updating S(k,k:path(k)-1)
S = cur7;
c = cur8;
z = cur9;          % Current point
d = cur10;         % d(k): left or right search direction at level k
prsd = cur11;      % Partial squared residual norm for z
tried = cur12;

k = max(m - 1,1);
dflag = 1;             % Intend to move down to a lower level

%update for generating mex
%'cputime' not supported
ct = 0;

while 1 && toc < duetime
    if k < m && prsd(k) > beta^2
        dflag = 0;
    end
    if k == m && prsd(k) > beta^2-sub_beta_square
        dflag = 0;
    end
    if dflag == 1
        if lflag(k) ~= 1 || uflag(k) ~= 1
            if k ~= 1 % Move to level k-1
                
                k1 = k - 1;
                tried(k1,:) = 0;
                
                % Update path
                if ulevel ~= 0
                    path(ulevel:k-1) = k;
                    for j = ulevel-1:-1:1
                        if path(j) < k
                            path(j) = k;
                        else
                            break; % Note path(1:j-1) >= path(j)
                        end
                    end
                end
                
                % Update S
                if k <= m
                    for j = path(k1):-1:k1 + 1
                        S(k1,j-1) = S(k1,j) - R(k1,j) * z(j);
                    end
                    c(k1) = S(k1,k1) / R(k1,k1);
                else
                    S(m:k1,k1) = S(k1,k1+1) - R(m,k1+1) * z(k1+1);
                    if R(m,k1)~=0
                        c(k1) = S(k1,k1) / R(m,k1);
                    else
                        c(k1) = l(k1);
                    end
                end
                
                % Compute the new bound at level k1
                [l(k1),u(k1)] = bound(c(k1),R,utemp,beta,prsd,k1,sub_beta_square);
                % Find the initial integer in [l(k1), u(k1)]
                if l(k1) > u(k1)
                    lflag(k1) = 1;
                    uflag(k1) = 1;
                else
                    if k1<=n-lower_bound_levels
                        [z(k1),d(k1),lflag(k1),uflag(k1)] = init(c(k1),l(k1),u(k1));
                        
                        
                        if k1<=m
                            gamma = R(k1,k1)*(c(k1) - z(k1));
                            prsd(k1)= prsd(k1+1) + gamma * gamma;
                        end
                        
                    else
                        [z(k1),d(k1),lflag(k1),uflag(k1)] = init(c(k1),l(k1),u(k1));
                        if lflag(k) ~= 1 || uflag(k) ~= 1
                            if k1<=m
                                last_rows_norm_square = norm(y(k1:m,1)-R(k1:m,k1:end)*z(k1:end,1))^2;
                            else
                                last_rows_norm_square = 0;
                            end
                            [lowerbound_for_first_cols,~,~,~]  = lowerbound_for_first_x_cols(R(1:min(m,k1-1),1:k1-1),y(1:min(m,k1-1),1)-R(1:min(m,k1-1),k1:end)*z(k1:end,1),l_original(1:k1-1,1),u_original(1:k1-1,1),noise_var,beta^2 - last_rows_norm_square);
                            tried(k1,z(k1,1)+1) = 1;
                            
                            if lowerbound_for_first_cols>beta^2 - last_rows_norm_square
                                parfor z_value_at_k1=l(k1):u(k1)
                                    copy_z = z;
                                    copy_z(k1,1) = z_value_at_k1;
                                    [lowerbound_for_first_cols,~,~,~]  = lowerbound_for_first_x_cols(R(1:min(m,k1-1),1:k1-1),y(1:min(m,k1-1),1)-R(1:min(m,k1-1),k1:end)*copy_z(k1:end,1),l_original(1:k1-1,1),u_original(1:k1-1,1),noise_var,beta^2 - last_rows_norm_square);
                                    if lowerbound_for_first_cols>beta^2 - last_rows_norm_square
                                        tried(k1,z_value_at_k1+1) = 1;
                                    end
                                end
                                while tried(k1,z(k1,1)+1) == 1
                                    [z(k1),d(k1),lflag(k1),uflag(k1)] = update(z(k1),d(k1),lflag(k1),uflag(k1),l(k1),u(k1));
                                    if lflag(k1) == 1 && uflag(k1) == 1
                                        break;
                                    end
                                end
                                if lflag(k1) ~= 1 || uflag(k1) ~= 1
                                    tried(k1,z(k1,1)+1) = 1;
                                end
                            end
                        end
                        if k1<=m
                            gamma = R(k1,k1)*(c(k1) - z(k1));
                            prsd(k1)= prsd(k1+1) + gamma * gamma;
                        end
                    end
                end
                k = k1;
                ulevel = 0;
            else % A valid point is found
                zhat = z;
                beta = sqrt(prsd(1));
                for j = 1:n
                    [l(j),u(j)] = bound(c(j),R,utemp,beta,prsd,j,sub_beta_square);
                end
                dflag = 0;
            end
        else % Will move back to a higher level
            dflag = 0;
        end
    else
        if k == n - 1 % The optimal solution has been found, terminate
            cur_index = cur_index + 1;
            if cur_index == cur_length
                break;
            end
            
            % else Switch to the next ordered branch
            l = results1(visit_order(cur_index),:,:)';
            u = results2(visit_order(cur_index),:,:)';
            lflag = results3(visit_order(cur_index),:,:)';
            uflag = results4(visit_order(cur_index),:,:)';
            ulevel = results5(visit_order(cur_index),:,:);
            path = results6(visit_order(cur_index),:,:)';
            S = reshape(results7(visit_order(cur_index),:,:),n,n);
            c = results8(visit_order(cur_index),:,:)';
            z = results9(visit_order(cur_index),:,:)';
            d = results10(visit_order(cur_index),:,:)';
            prsd = results11(visit_order(cur_index),:,:)';
            tried = reshape(results12(visit_order(cur_index),:,:),n,interval_length+1);
            k = max(m - 1,1); % In case m = 1
            dflag = 1;
            if k == 1
                dflag = 0;
            end
        else
            % Move back to level k+1
            if ulevel == 0
                ulevel = k;
            end
            k = k + 1;
            if lflag(k) ~= 1 || uflag(k) ~= 1
                % Find a new integer at level k
                if k <= n-lower_bound_levels
                    [z(k),d(k),lflag(k),uflag(k)] = ...
                        update(z(k),d(k),lflag(k),uflag(k),l(k),u(k));
                    
                    if k<=m
                        
                        gamma = R(k,k)*(c(k)-z(k));
                        % Update partial squared residual norm
                        prsd(k)= prsd(k+1) + gamma * gamma;
                    end
                else
                    [z(k),d(k),lflag(k),uflag(k)] = ...
                        update(z(k),d(k),lflag(k),uflag(k),l(k),u(k));
                    
                    
                    if lflag(k) ~= 1 || uflag(k) ~= 1
                        if k<=m
                            last_rows_norm_square = norm(y(k:m,1)-R(k:m,k:end)*z(k:end,1))^2;
                        else
                            last_rows_norm_square = 0;
                        end
                        if tried(k,z(k,1)+1) == 1
                            parfor tried_index_z_value =l(k)+1:u(k)+1
                                z_value_at_k = tried_index_z_value-1;
                                
                                if tried(k,tried_index_z_value)==1
                                    continue
                                end
                                copy_z = z;
                                copy_z(k,1) = z_value_at_k;
                                [lowerbound_for_first_cols,~,~,~]  = lowerbound_for_first_x_cols(R(1:min(k-1,m),1:k-1),y(1:min(k-1,m),1)-R(1:min(k-1,m),k:end)*copy_z(k:end,1),l_original(1:k-1,1),u_original(1:k-1,1),noise_var,beta^2-last_rows_norm_square);
                                if lowerbound_for_first_cols>beta^2 - last_rows_norm_square
                                    tried(k,tried_index_z_value) = 1;
                                end
                            end
                            while tried(k,z(k,1)+1) == 1
                                [z(k),d(k),lflag(k),uflag(k)] =update(z(k),d(k),lflag(k),uflag(k),l(k),u(k));
                                if lflag(k) == 1 && uflag(k) == 1
                                    break
                                end
                            end
                            if lflag(k) ~= 1 || uflag(k) ~= 1
                                tried(k,z(k,1)+1)=1;
                            end
                        end
                    end
                    
                    
                    if k<=m
                        
                        gamma = R(k,k)*(c(k)-z(k));
                        % Update partial squared residual norm
                        prsd(k)= prsd(k+1) + gamma * gamma;
                    end
                end
                if lflag(k) ~= 1 || uflag(k) ~= 1
                    dflag = 1;
                end
            end
        end
    end
end

if toc >= duetime % Exceed due time
    flag = 1;
else
    flag = 0;
end

if isempty(zhat) % No integer point found within the search radius
    zhat = z0;
else  % Shift back, see subfunction shift
    zhat(col_neg) = -zhat(col_neg);
    zhat = zhat + repmat(s,1,1);
end

end


% ------------------------------------------------------------------
% --------  Subfunctions  ------------------------------------------
% ------------------------------------------------------------------

function [R,y,u,l,col_neg,s,new_z0] = shift(R,y,l,u,z0)

% Perform transformations:
%    z(j) := -l(j) + z(j) if r_{tj} >= 0  for t=m or t=j
%    z(j) := u(j) - z(j)  if r_{tj} < 0   for t=m or t=j
% So that z(j) is in {0,1, . . . ,u_j-l_j} and R(t,j)>=0, j=1:n

[m,n] = size(R);
R_temp = R;
[m,n]=size(R);
if m == 1
    col_neg = R < 0;
else
    col_neg = [diag(R)', R(m,m+1:n)] < 0;
end
R(:,col_neg) = - R(:,col_neg);
s = l .* (1 - col_neg)' + u .* col_neg';
new_z0 = (z0-l) .* (1 - col_neg)' + (u-z0) .* col_neg';
y = y - R_temp * s;
u = u - l;
l = zeros(size(l));

end


function [dis,branch_info] = initialSearch(S,R,c,l,u,beta,y,sub_beta_square,l_original,u_original,noise_var,lower_bound_levels)
% Search z(m:n) and get relevant information to be used in determining
% the branch order for the search process

[m,n] = size(R);
tried = zeros(n,u_original(n,1)-l_original(n,1)+1);
% Initialization
utemp = u;
dflag = 1; % dflag for down or up search direction
k = n;
dis = Inf;

% update for generating mex
%branch_info = [];
branch_info = {ones(n,1),ones(n,1),ones(n,1),ones(n,1),0,ones(n,1),ones(n,n),ones(n,1),ones(n,1),ones(n,1),ones(m+1,1),ones(n,u_original(n,1)-l_original(n,1)+1)};



lflag = zeros(n,1); % lflag(k) = 1 if lower bound is reached at level k
uflag = zeros(n,1); % uflag(k) = 1 if upper bound is reached at level k
d = zeros(n,1);     % Left or right search direction at level k
path = n*ones(n,1); % path(k): record information for updating S(k,k:path(k)-1)
ulevel = 0;         % The level where search moves up to a higher level
prsd = zeros(m+1,1); % Partial squared residual norm for z
z = zeros(n,1);
%[l(k),u(k)] = bound(c(k),R,u,beta,prsd,k,sub_beta_square);
[z(k),d(k),lflag(k),uflag(k)] = init(c(k),l(k),u(k));
while 1
    if k < m && prsd(k) > beta^2
        dflag = 0;
    end
    if k == m && prsd(k) > beta^2 - sub_beta_square
        dflag = 0;
    end
    if dflag == 1
        
        if lflag(k) ~= 1 || uflag(k) ~= 1
            if k ~= m-1 && k ~=1 % Move to level k-1
                k1 = k - 1;
                tried(k1,:) = 0;
                % Update path
                if ulevel ~= 0
                    path(ulevel:k-1) = k;
                    for j = ulevel-1:-1:1
                        if path(j) < k
                            path(j) = k;
                        else
                            break; % Note path(1:j-1) >= path(j)
                        end
                    end
                end
                % Update S
                if k <= m
                    for j = path(k1):-1:k1+1
                        S(k1,j-1) = S(k1,j) - R(k1,j) * z(j);
                    end
                    c(k1) = S(k1,k1) / R(k1,k1);
                    
                else
                    S(m:k1,k1) = S(k1,k1+1) - R(m,k1+1) * z(k1+1);
                    if R(m,k1) ~= 0
                        c(k1) = S(k1,k1) / R(m,k1);
                    else
                        c(k1) = l(k1);
                    end
                end
                
                % Compute new bound at level k1
                [l(k1),u(k1)] = bound(c(k1),R,utemp,beta,prsd,k1,sub_beta_square);
                
                % Find the initial integer in [l(k1), u(k1)]
                if l(k1) > u(k1)
                    lflag(k1) = 1;
                    uflag(k1) = 1;
                    dflag = 0;
                else
                    if k1<=n-lower_bound_levels
                        [z(k1),d(k1),lflag(k1),uflag(k1)] = init(c(k1),l(k1),u(k1));
                        
                        
                        if k1<=m
                            gamma = R(k1,k1)*(c(k1) - z(k1));
                            prsd(k1)= prsd(k1+1) + gamma * gamma;
                        end
                        
                    else
                        [z(k1),d(k1),lflag(k1),uflag(k1)] = init(c(k1),l(k1),u(k1));
                        if lflag(k) ~= 1 || uflag(k) ~= 1
                            if k1<=m
                                last_rows_norm_square = norm(y(k1:m,1)-R(k1:m,k1:end)*z(k1:end,1))^2;
                            else
                                last_rows_norm_square = 0;
                            end
                            [lowerbound_for_first_cols,~,~,~]  = lowerbound_for_first_x_cols(R(1:min(m,k1-1),1:k1-1),y(1:min(m,k1-1),1)-R(1:min(m,k1-1),k1:end)*z(k1:end,1),l_original(1:k1-1,1),u_original(1:k1-1,1),noise_var,beta^2 - last_rows_norm_square);
                            tried(k1,z(k1,1)+1) = 1;
                            
                            if lowerbound_for_first_cols>beta^2 - last_rows_norm_square
                                parfor z_value_at_k1=l(k1):u(k1)
                                    copy_z = z;
                                    copy_z(k1,1) = z_value_at_k1;
                                    [lowerbound_for_first_cols,~,~,~]  = lowerbound_for_first_x_cols(R(1:min(m,k1-1),1:k1-1),y(1:min(m,k1-1),1)-R(1:min(m,k1-1),k1:end)*copy_z(k1:end,1),l_original(1:k1-1,1),u_original(1:k1-1,1),noise_var,beta^2 - last_rows_norm_square);
                                    if lowerbound_for_first_cols>beta^2 - last_rows_norm_square
                                        tried(k1,z_value_at_k1+1) = 1;
                                    end
                                end
                                while tried(k1,z(k1,1)+1) == 1
                                    [z(k1),d(k1),lflag(k1),uflag(k1)] = update(z(k1),d(k1),lflag(k1),uflag(k1),l(k1),u(k1));
                                    if lflag(k1) == 1 && uflag(k1) == 1
                                        break;
                                    end
                                end
                                if lflag(k1) ~= 1 || uflag(k1) ~= 1
                                    tried(k1,z(k1,1)+1) = 1;
                                end
                            end
                        end
                        
                        if k1<=m
                            gamma = R(k1,k1)*(c(k1) - z(k1));
                            prsd(k1)= prsd(k1+1) + gamma * gamma;
                        end
                    end
                    
                end
                k = k1;
                ulevel = 0;
            else % The first valid x_{m:n} at this branch has been found
                dis_y = (y(m)-R(m,m:n)*z(m:n))^2;
                y_new = y(1:m-1)-R(1:m-1,m:n)*z(m:n);
                lowerbound = sub_beta_square;
                if lowerbound+dis_y > beta^2
                    dflag = 0;
                else
                    branch_info = {l,u,lflag,uflag,ulevel,path,S,c,z,d,prsd,tried};
                    %                     x_rls = R(1:m-1,1:m-1)\y_new;
                    %                     x_bils = median([l(1:m-1),round(x_rls),u(1:m-1)],2);
                    dis = 1;
                    
                    break;
                end
            end
        else % Will move back to a higher level
            dflag = 0;
        end
    else
        if k == n-1 % The optimal solution has been found, terminate
            break;
        else
            % Move back to level k+1
            if ulevel == 0
                ulevel = k;
            end
            k = k + 1;
            if lflag(k) ~= 1 || uflag(k) ~= 1
                % Find a new integer at level k
                if k <= n-lower_bound_levels
                    [z(k),d(k),lflag(k),uflag(k)] = ...
                        update(z(k),d(k),lflag(k),uflag(k),l(k),u(k));
                    
                    if k<=m
                        
                        gamma = R(k,k)*(c(k)-z(k));
                        % Update partial squared residual norm
                        prsd(k)= prsd(k+1) + gamma * gamma;
                    end
                else
                    [z(k),d(k),lflag(k),uflag(k)] = ...
                        update(z(k),d(k),lflag(k),uflag(k),l(k),u(k));
                    
                    if lflag(k) ~= 1 || uflag(k) ~= 1
                        if k<=m
                            last_rows_norm_square = norm(y(k:m,1)-R(k:m,k:end)*z(k:end,1))^2;
                        else
                            last_rows_norm_square = 0;
                        end
                        if tried(k,z(k,1)+1) ==1
                            parfor z_value_at_k=l(k):u(k)
                                if tried(k,z_value_at_k+1)==1
                                    continue
                                end
                                copy_z = z;
                                copy_z(k,1) = z_value_at_k;
                                [lowerbound_for_first_cols,~,~,~]  = lowerbound_for_first_x_cols(R(1:min(k-1,m),1:k-1),y(1:min(k-1,m),1)-R(1:min(k-1,m),k:end)*copy_z(k:end,1),l_original(1:k-1,1),u_original(1:k-1,1),noise_var,beta^2-last_rows_norm_square);
                                if lowerbound_for_first_cols>beta^2 - last_rows_norm_square
                                    tried(k,z_value_at_k+1) = 1;
                                end
                            end
                            while tried(k,z(k,1)+1) == 1
                                [z(k),d(k),lflag(k),uflag(k)] =update(z(k),d(k),lflag(k),uflag(k),l(k),u(k));
                                if lflag(k) == 1 && uflag(k) == 1
                                    break
                                end
                            end
                            if lflag(k) ~= 1 || uflag(k) ~= 1
                                tried(k,z(k,1)+1)=1;
                            end
                        end
                    end
                    
                    
                    if k<=m
                        
                        gamma = R(k,k)*(c(k)-z(k));
                        % Update partial squared residual norm
                        prsd(k)= prsd(k+1) + gamma * gamma;
                    end
                end
                if lflag(k) ~= 1 || uflag(k) ~= 1
                    dflag = 1;
                end
            end
        end
    end
end

end


function [l_k,u_k] = bound(c_k,R,u,beta,prsd,k,sub_beta_square)

% Compute new lower bound and upper bound for z_k


m = size(R,1);
if k > m - 1
    %lambda_k = c_k;
    %mu_k = c_k;
    lambda_k = c_k - (sqrt(max(beta^2-sub_beta_square,0))+R(m,m:k-1)*u(m:k-1))/R(m,k);
    mu_k = c_k + sqrt(max(beta^2-sub_beta_square,0))/R(m,k);
else
    %lambda_k = c_k;
    %mu_k = c_k;
    lambda_k = c_k - sqrt(max(beta^2-prsd(k+1),0))/R(k,k);
    mu_k = c_k + sqrt(max(beta^2-prsd(k+1),0))/R(k,k);
    
    
end

if lambda_k - floor(lambda_k) < 1e-12 && lambda_k ~= 0
    lambda_k = floor(lambda_k);
end

if ceil(mu_k) - mu_k < 1e-12 && mu_k ~= u(k)
    mu_k = ceil(mu_k);
end

if lambda_k == Inf || lambda_k == -Inf
    l_k = 0;
    u_k = u(k);
else
    l_k = ceil(max(0,lambda_k));
    u_k = floor(min(u(k),mu_k));
end

end


function [z_k,d_k,lflag_k,uflag_k] = init(c_k,l_k,u_k)

% Find the initial integer and the search direction at level _k
%

% if u_k == l_k
%     z_k = l_k;
%     lflag_k=1;
%     uflag_k=1;
%     d_k=1;
%     return
% end


z_k = round(c_k);

if z_k <= l_k
    z_k = l_k;
    lflag_k = 1;  % The lower bound is reached
    uflag_k = 0;
    d_k = 1;
elseif z_k >= u_k
    z_k = u_k;
    uflag_k = 1;  % The upper bound is reached
    lflag_k = 0;
    d_k = -1;
else
    lflag_k = 0;
    uflag_k = 0;
    
    
    if c_k > z_k
        d_k = 1;
    else
        d_k = -1;
    end
end

end


function [z_k,d_k,lflag_k,uflag_k] = update(z_k,d_k,lflag_k,uflag_k,l_k,u_k)

% Find a new integer at level k and record it if it hits a boundary.

if lflag_k == 0 && uflag_k == 0
    zk = z_k + d_k;
    if zk > u_k
        uflag_k = 1;
    elseif zk < l_k
        lflag_k = 1;
    else
        z_k = zk;
        if d_k > 0
            d_k = -d_k - 1;
        else
            d_k = -d_k + 1;
        end
    end
end

if lflag_k == 1 && uflag_k == 0
    z_k = z_k + 1;
    if z_k > u_k
        uflag_k = 1;
    end
elseif lflag_k == 0 && uflag_k == 1
    z_k = z_k - 1;
    if z_k < l_k
        lflag_k = 1;
    end
end

end



