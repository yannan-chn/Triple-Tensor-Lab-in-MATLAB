function [tenv,info] = trip_als(tent,rank)
% trip_als  alternating least squares method for finding triple factors.
%
% Input:
%     tent      ---   ten_I*J*K
%     rank      ---   triple rank L or initial iterate
% Output:
%     tenv      ---   [a_IMN,b_LJN,C_LMK]
%
% References
% [1] Liqun Qi, Yannan Chen, Mayank Bakshi, and Xinzhen Zhang, 
%     Triple decomposition and tensor recovery of third order tensors,
%     SIAM Journal on Matrix Analysis and Applications, 
%     42(1): 299--329, (2021).
%
% Yannan Chen  ... June 2, 2024
tic;  fprintf('\nAlternating least squares algorithm ... \n');

% parameters
[I,J,K] = size(tent);
if isscalar(rank)
    L = rank;  M = L;  N = L;
    if isreal(tent)
        randomFun = @(n) randn(n,1);
    else
        randomFun = @(n) randn(n,1)+randn(n,1)*1i;
    end
    tenv = trip_rand([I,J,K,L],randomFun);  % random initial iterate
else
    L = round(sqrt(numel(rank)/(I+J+K)));  M = L;  N = L;
    tenv = rank;                           % input initial iterate
end

maxiter = 200;      % iteration
epsStop = 6.e-6;    % stoping tolerance (eps^(1/3))
gamma   = 1.0;      % extrapolation
sigma   = eps^(1/4)*eye(L*L);

% initialization
p = 1;    q = I*M*N;    xA = reshape(tenv(p:q),[I,M*N]);
p = q+1;  q = q+L*J*N;  xB = reshape(permute(reshape(tenv(p:q),[L,J,N]),[2,1,3]),[J,L*N]);
p = q+1;  q = q+L*M*K;  xC = reshape(tenv(p:q),[L*M,K]);
Ten_K = reshape(tent,[I*J,K]);  frobTen = norm(Ten_K,'fro');  
TenI_ = reshape(tent,[I,J*K]);

AA = reshape(permute(reshape(xA'*xA,[M,N,M,N]),[1,3,2,4]),[M*M,N*N]);  % left conjugate
BB = reshape(permute(reshape(xB'*xB,[L,N,L,N]),[1,3,2,4]),[L*L,N*N]);  % left conjugate
CC = reshape(permute(reshape(xC*xC',[L,M,L,M]),[1,3,2,4]),[L*L,M*M]);  % right conjugate

fprintf('  iter |    LS cost    |     dA    |     dB    |     dC    \n');
TA = reshape(permute(reshape(xA'*TenI_,[M,N,J,K]),[2,3,1,4]),[N*J,M*K]);
TAC = TA*reshape(xC,[L,M*K])';  % NJ*L
foRe = sqrt(abs(real( sum(dot(BB*AA.',CC))-2*dot(reshape(TAC.',[L*N*J,1]),reshape(xB.',[L*N*J,1])) )/(frobTen*frobTen)+1));
fprintf('  %4d | %13.6e ',0,foRe);

reErr = [foRe, zeros(1,maxiter)];
rdABC = zeros(1,maxiter+1);
CPUtm = [toc, zeros(1,maxiter)];

for iter=1:maxiter
    % Update A_IMN
    TC  = reshape(permute(reshape(Ten_K*xC',[I,J,L,M]),[2,3,1,4]),[J*L,I*M]);
    TBC = reshape((reshape(xB,[J*L,N])'*TC).',[I,M*N]);  % right conjugate
    yA = TBC / (sigma + reshape(permute(reshape(CC'*BB,[M,M,N,N]),[1,3,2,4]),[M*N,M*N])).';
    dA = yA-xA;    xA = (1-gamma) * xA + gamma * yA;
    AA = reshape(permute(reshape(xA'*xA,[M,N,M,N]),[1,3,2,4]),[M*M,N*N]);
    
    % Update B_JLN
    TCA = reshape(TC*reshape(conj(xA),[I*M,N]),[J,L*N]);
    yB = TCA / (sigma + reshape(permute(reshape(CC*conj(AA),[L,L,N,N]),[1,3,2,4]),[L*N,L*N]));
    dB = yB-xB;    xB = (1-gamma) * xB + gamma * yB;
    BB = reshape(permute(reshape(xB'*xB,[L,N,L,N]),[1,3,2,4]),[L*L,N*N]);
    
    % Update C_LMK
    TA = reshape(permute(reshape(xA'*TenI_,[M,N,J,K]),[2,3,1,4]),[N*J,M*K]);
    TAB = reshape(reshape(reshape(xB,[J*L,N]).',[N*J,L])'*TA,[L*M,K]);
    yC = (sigma + reshape(permute(reshape(BB*AA.',[L,L,M,M]),[1,3,2,4]),[L*M,L*M])) \ TAB;
    dC = yC-xC;    xC= (1-gamma) * xC + gamma * yC;
    CC = reshape(permute(reshape(xC*xC',[L,M,L,M]),[1,3,2,4]),[L*L,M*M]);
    
    rdA = norm(dA,'fro')/norm(xA,'fro');  rdB = norm(dB,'fro')/norm(xB,'fro');
    rdC = norm(dC,'fro')/norm(xC,'fro');  fprintf('| %9.3e | %9.3e | %9.3e \n',rdA,rdB,rdC);
    % Check convergence
    foRe = sqrt(abs(real(sum(dot(BB*AA.',CC))-2*sum(dot(TAB,xC)))/(frobTen*frobTen)+1));
    reErr(iter+1) = foRe;  CPUtm(iter+1) = toc;
    dABC = max([rdA,rdB,rdC]);  rdABC(iter+1) = dABC;
    if dABC < epsStop || foRe < epsStop*epsStop
        break;
    else
        fprintf('  %4d | %13.6e ',iter,foRe);
    end
end
tenv = [xA(:); reshape(permute(reshape(xB,[J,L,N]),[2,1,3]),[L*J*N,1]); xC(:)];
foRe = norm(reshape(trip_full(tenv,[I,J,K,L])-tent,[I*J,K]),'fro')/frobTen;
fprintf('Relative error of fitting      ---  %13.6e \n',foRe);
reErr(iter+1) = foRe;  CPUtm(iter+1) = toc;

info.Iter  = 0:iter;
info.reErr = reErr(1:iter+1);
info.rdABC = rdABC(1:iter+1);
info.CPUtm = CPUtm(1:iter+1);




