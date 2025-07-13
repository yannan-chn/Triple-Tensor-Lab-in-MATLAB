function tent = trip_full(tenv,sz)
% trip_full  forms a full third order tensor by its factors.  
%
% Input:
%     tenv      ---   [a_IMN,b_LJN,C_LMK]
%     sz        ---   [I,J,K,L], tensor size (I,J,K), triple rank L
% Output:
%     tent      ---   ten_I*J*K
%
% Yannan Chen  ... June 1, 2024

I = sz(1);  J = sz(2);  K = sz(3);     % tensor size
L = sz(4);  M = L;  N = L;             % triple rank
p = 1;    q = I*M*N;    XA = reshape(tenv(p:q),[I*M,N]);
p = q+1;  q = q+L*J*N;  XB = reshape(tenv(p:q),[L*J,N]);
p = q+1;  q = q+L*M*K;  XC = reshape(tenv(p:q),[L*M,K]);

tent = reshape(reshape(permute(reshape(XB*XA.',[L,J,I,M]),[3,2,1,4]),[I*J,L*M])*XC,[I,J,K]);
