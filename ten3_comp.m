function [tcom,U,V,W,info] = ten3_comp(tent,rank)
% ten3_comp  Compressor of 3D tensors.
%
% Yannan Chen, July 12, 2025

% parameters
tic;  fprintf('\nTensor compressor ......\n');
[I,J,K] = size(tent);
if numel(rank) == 1
    L = rank;  M = rank;  N = rank;
else
    L = rank(1);  M = rank(2);  N = rank(3);
end

tcom = reshape(tent,[I,J*K]);
U = randomSVD(tcom,L);   tcom = reshape((U'*tcom).',[J,K*L]);
V = randomSVD(tcom,M);   tcom = reshape((V'*tcom).',[K,L*M]);
W = randomSVD(tcom,N);   test = W'*tcom;  % N*LM
tcom = reshape((test).',[L,M,N]);

% test = U*reshape((W*reshape(reshape(test,[N*L,M])*V.',[N,L*J])).',[L,J*K]);  % I*JK
% info.err = norm(test-tent,'fro')/frobNorm;  info.cput = toc;
Jc = min(J,10);  omega   = randn(Jc,J)+1i*randn(Jc,J);
Kc = min(K,10);  upsilon = randn(Kc,K)+1i*randn(Kc,K);
test = U*reshape(((upsilon*W)*reshape(reshape(test,[N*L,M])*(omega*V).',[N,L*Jc])).',[L,Jc*Kc]);
tent = reshape(tent,[I,J*K])*kron(upsilon.',omega.');
info.err = norm(test-tent,'fro') ./ norm(tent,'fro');  info.cput = toc;

fprintf('  estimated compression error  ---  %e \n',info.err);
fprintf('  CPU time (second)            ---  %f \n',info.cput);


function U = randomSVD(B,k)
% Algorithm 8 in
%   P.-G. Martinsson and J. A. Tropp, Randomized Numerical Linear Algebra 
%   -- Foundations & Algorithms, Acta Numerica, (2020), pp. 403-572.
%
[m,n] = size(B);  ell = min([2*k+1,k+10,m,n]);
Q = randomRangeFinder(B,ell);
[Uc,~,~] = svd(Q'*B(:,randperm(n,min([m*2+1,4*ell+2,ell+50,n]))),'econ');
U = Q*Uc(:,1:k);

function Q = randomRangeFinder(B,ell)
% Algorithm 7 in 
%   P.-G. Martinsson and J. A. Tropp, Randomized Numerical Linear Algebra 
%   -- Foundations & Algorithms, Acta Numerica, (2020), pp. 403-572.
%
n = size(B,2);
[Q,~] = qr(B(:,randperm(n,min(ell+30,2*ell+1))),0);
