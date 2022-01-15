function [H, s, v, y, sigma_squared] = getSystem(SNR, Htype, K, N, condNum)

% [H, s, v, y, sigma_squared] = getSystem(SNR, Htype, K, N, condNum) generates the
% y=H*s+v system
% 
% Inputs:
%     SNR - integer scalar
%     Htype - string, either 'random' or 'ill'
%     K - integer scalar
%     N - integer scalar
%     condNum - real scalar
% 
% Outputs:
%     H - K-by-N real matrix
%     s - N-dimensional real vector
%     v - K-dimensional real vector
%     y - K-dimensional real vector
%     sigma_squared - real scalar

% Subfunctions: getH

rng('shuffle')
s=2*randi([0,1],[N,1])-1;
SNR_prime = 10^(SNR/10);

if strcmp(Htype,'random')
    H=randn(K,N);
    sigma_squared = N / (SNR_prime);
end

if strcmp(Htype,'ill')
    [H, sing_sum] = getH(N, K, condNum);
    sigma_squared = sing_sum/ (K * SNR_prime);
end
v=sqrt(sigma_squared)*randn(K,1);
y=H*s+v;
end





function [H, sing_sum] = getH(N,K,condNum)

% [H, sing_sum] = getH(N,K,condNum) generates an ill-conditioned model 
% matrix H with the desired condition number
% 
% Inputs:
%     K - integer scalar
%     N - integer scalar
%     condNum - real scalar
% 
% Outputs:
%     H - K-by-N real matrix
%     sing_sum - real scalar (sum of singular values of H)


[U,~] = qr(randn(K,K));
[V,~] = qr(randn(N,N));

alpha = exp(-N/(10*(K-1)) * log(condNum));
S = zeros(K,N);
sing_sum = 0;

for k=1:K
    diag_entry = alpha^(10*(k-1)/N);
    S(k,k) = diag_entry;
    sing_sum = sing_sum + diag_entry;
end
H = U * S * V';
end