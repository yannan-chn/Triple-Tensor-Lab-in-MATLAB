function frobnorm = trip_norm(tenv,sz)
% trip_norm  computes the Frobenius norm of a (triple) tensor.  
%
% Input:
%     tent      ---   tent = T_IJK or tenv = [a_IMN,b_LJN,C_LMK]
%     sz        ---   [I,J,K,L], tensor size (I,J,K), triple rank L
% Output:
%     frobnorm  ---   Frobenius norm
%
% Yannan Chen  ... July 10, 2025

I = sz(1);  J = sz(2);  K = sz(3);     % tensor size
L = sz(4);  M = L;  N = L;             % triple rank

if length(tenv) == I*M*N+L*J*N+L*M*K
    p = 1;    q = I*M*N;    XA = reshape(tenv(p:q),[I,M*N]);
    p = q+1;  q = q+L*J*N;  XB = reshape(permute(reshape(tenv(p:q),[L,J,N]),[2,1,3]),[J,L*N]);
    p = q+1;  q = q+L*M*K;  XC = reshape(tenv(p:q),[L*M,K]);
    
    AA = reshape(permute(reshape(XA'*XA,[M,N,M,N]),[1,3,2,4]),[M*M,N*N]);  % left conjugate
    BB = reshape(permute(reshape(XB'*XB,[L,N,L,N]),[1,3,2,4]),[L*L,N*N]);  % left conjugate
    CC = reshape(permute(reshape(XC*XC',[L,M,L,M]),[1,3,2,4]),[L*L,M*M]);  % right conjugate
    
    frobnorm = sqrt(sum(real(dot(BB*AA.',CC))));
else
    frobnorm = norm(tenv(:));
end
