function [f,g] = trip_evfg(x, tent,frobTen)
% trip_evfg  evaluates the function value and associated gradient vector 
%  of a nonlinear least squares fitting at x.
%
% Input:
%     x         ---   current estimater [xA,xB,xC]
%     tent      ---   ten_I*J*K
% Output:
%     f         ---   function value
%     g         ---   gradient vector
%
% Reference
% [1] Yannan Chen, Xinzhen Zhang, Liqun Qi, and Yanwei Xu,
%     A Barzilai--Borwein gradient algorithm for spatio-temporal internet 
%     traffic data completion via tensor triple decomposition, 
%     Journal of Scientific Computing, 88:65, (2021).
%
% Yannan Chen  ... July 23, 2024

% parameters
[I,J,K] = size(tent);                               % tensor size
L = round(sqrt(numel(x)/(I+J+K)));  M = L;  N = L;  % triple rank
TenI_ = reshape(tent,[I,J*K]);
if nargin == 2
    frobTen = norm(TenI_,'fro');
end

p = 1;    q = I*M*N;    xA = reshape(x(p:q),[I,M*N]);
p = q+1;  q = q+L*J*N;  xB = reshape(permute(reshape(x(p:q),[L,J,N]),[2,1,3]),[J,L*N]);
p = q+1;  q = q+L*M*K;  xC = reshape(x(p:q),[L*M,K]);

% evaluate a function value
AA = reshape(permute(reshape(xA'*xA,[M,N,M,N]),[1,3,2,4]),[M*M,N*N]);  % left conjugate
BB = reshape(permute(reshape(xB'*xB,[L,N,L,N]),[1,3,2,4]),[L*L,N*N]);  % left conjugate
CC = reshape(permute(reshape(xC*xC',[L,M,L,M]),[1,3,2,4]),[L*L,M*M]);  % right conjugate

TA = reshape(permute(reshape(xA'*TenI_,[M,N,J,K]),[2,3,1,4]),[N*J,M*K]);
TAB = reshape(reshape(reshape(xB,[J*L,N]).',[N*J,L])'*TA,[L*M,K]);
f = real( sum(dot(BB*AA.',CC))-2*sum(real(dot(xC,TAB)))+frobTen*frobTen )/2;

% evaluate a gradient vector
if nargout==2
    Ten_K = reshape(tent,[I*J,K]);
    TC = reshape(permute(reshape(Ten_K*xC',[I,J,L,M]),[2,3,1,4]),[J*L,I*M]);
    TCB = reshape((reshape(xB,[J*L,N])'*TC).',[I,M*N]);
    TCA = reshape(TC*conj(reshape(xA,[I*M,N])),[J,L*N]);

    gA = xA*reshape(permute(reshape(CC.'*conj(BB),[M,M,N,N]),[1,3,2,4]),[M*N,M*N]) - TCB;
    gB = xB*reshape(permute(reshape(CC*conj(AA),[L,L,N,N]),[1,3,2,4]),[L*N,L*N])   - TCA;
    gC = reshape(permute(reshape(BB*AA.',[L,L,M,M]),[1,3,2,4]),[L*M,L*M])*xC       - TAB;
    g = [gA(:); reshape(permute(reshape(gB,[J,L,N]),[2,1,3]),[L*J*N,1]); gC(:)];
end

