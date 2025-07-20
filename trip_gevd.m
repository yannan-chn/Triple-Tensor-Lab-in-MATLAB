function [tenv,info] = trip_gevd(tent,rank,param)
% trip_gevd  compute the triple decomposition of a full tensor.
%
% Input:
%     tent      ---   ten_I*J*K
%     rank      ---   triple rank L
% Output:
%     tenv      ---   [a_IMN,b_LJN,C_LMK]
%
% Reference
% [1] Yannan Chen, Liqun Qi, A direct method for solving the complex-valued
%     triple decomposition of third-order tensors, submitted 2024.
% [2] P.-G. Martinsson and J. A. Tropp, Randomized Numerical Linear Algebra 
%     --Foundations & Algorithms, Acta Numerica, (2020), pp. 403-572.
%
% Yannan Chen  ... July 20, 2025
tic;  fprintf('Direct method for the triple tensor decomposition ...\n');

% parameters
[I,J,K] = size(tent);
if rank*rank > min([I,J,K])
    error('Assumption min(I,J,K) >= R*R fails. \n    [I,J,K]=[%d,%d,%d] and Rank=%d',I,J,K,rank);
end
L = rank;  M = L;  N = L;

% generalized eigenvalues and associated eigenvectors
NOS = min([5,I,J,K]);  weights = randn(NOS,4)+randn(NOS,4)*1i;

Tio = squeeze(sum(tent(randperm(I,NOS),:,:).*reshape(weights(:,1),[NOS,1,1]),1));
Tip = squeeze(sum(tent(randperm(I,NOS),:,:).*reshape(weights(:,2),[NOS,1,1]),1));
[Ui,Di,Vi] = svd(Tio);  Di = Di(1:M*L,1:M*L);  Vi = Vi(:,1:M*L);  Tip = Ui(:,1:M*L)'*Tip*Vi;
sigma = cluste(eig(Tip,Di),M);

Tjo = squeeze(sum(tent(:,randperm(J,NOS),:).*reshape(weights(:,3),[1,NOS,1]),2));
Tjp = squeeze(sum(tent(:,randperm(J,NOS),:).*reshape(weights(:,4),[1,NOS,1]),2));
[Uj,Dj,Vj] = svd(Tjo);  Dj = Dj(1:M*L,1:M*L);  Vj = Vj(:,1:M*L);  Tjp = Uj(:,1:M*L)'*Tjp*Vj;
upsilon = cluste(eig(Tjp,Dj),L);  Vj = Vi'*Vj;

[Xest,Dj] = eig(Tip*Vj*pinv(Dj)*Tjp*Vj',Di);  Xest = Vi*Xest;    % single eigenvalues
if sqrt(numel(Dj))==M*L
    fprintf('  Find L*M single generalized eigenvalues.\n');
end

% matching eigenvalues
Di = kron(sigma,upsilon);  Dj = diag(Dj);  
idx = matchpairs(abs(Di - Dj.'),1000);
pmat = sparse(idx(:,1),idx(:,2),1,L*M,L*M);  Dj = pmat*Dj;  eigErr = norm(Di-Dj,1); %  [Di,Dj]
Xest = reshape(Xest*pmat',[K,L,M]);  % size K*L*M
fprintf('  Matching error of eigenvalues  ---  %e\n',eigErr);

% find At_IMN, Bt_LJN and Ct_LMK
tent = reshape(tent,[I*J,K]);
NOS = randi(L,2);  Aest = tent*squeeze(Xest(:,NOS(1),:));
[Ui,Di,Vi] = svd(reshape(Aest(:,NOS(2)),[I,J]));
Di = diag((1.0)./sqrt(diag(Di(1:N,1:N))));  Ui = Di*Ui(:,1:N)';  Vi = Vi(:,1:N)*Di;
Aest = reshape(permute(reshape(Aest,[I,J,M]),[1,3,2]),[I*M,J])*Vi;  % IM*N
Best = Ui*reshape(tent*Xest(:,:,NOS(2)),[I,J*L]);                   % N*JL

ABest = reshape(permute(reshape(Aest*Best,[I,M,J,L]),[1,3,4,2]),[I*J,L*M]);
if eigErr <= sqrt(eps)
    NOS = randperm(I*J,10);  Xest = reshape(Xest,[K,L*M]);
    Cest = pinv(Xest*diag(sum(ABest(NOS,:),1) ./ sum(tent(NOS,:)*Xest,1)));  % LM*K
else
    % Cest = ABest \ tent;
    Cest = (ABest'*ABest) \ (ABest'*tent);
end
tenv = [Aest(:);  reshape(permute(reshape(Best,[N,J,L]),[3,2,1]),[L*J*N,1]); Cest(:)];

if nargin <= 2 || param(1) ~= 'e'
    IJc = min(I*J,10);  omega = randn(IJc,I*J)+randn(IJc,I*J)*1i;
    test = (omega*ABest)*Cest;  tent = omega*tent;
    info.reErr = norm(test-tent,'fro') ./ norm(tent,'fro'); % randomly estimated rel-error
else
    tenFrob = norm(tent,'fro');
    info.reErr = norm(tent-ABest*Cest,'fro')/tenFrob;         % exact relative error
end
fprintf('  Relative error of fitting      ---  %e\n',info.reErr);
info.CPUtm = toc;
fprintf('  CPU time (second)              ---  %e\n',info.CPUtm);


function sigma = cluste(nu,L)
% clust  cluster L*L numbers (nu) into L classes with equal size.
%
% Yannan Chen  ... July 6, 2025
sigma = zeros(L,1);    distMat = abs(nu-nu.') + 1024*eye(L*L);
for ell=1:L
    [~,kkk] = min(distMat(:));        kkk = mod(kkk(1)-1,L)+1;
    [~,idx] = sort(abs(nu-nu(kkk)));  idx = idx(1:L);
    sigma(ell) = sum(nu(idx))/L;      nu(idx) = [];
    distMat(idx,:) = [];              distMat(:,idx) = [];
end

