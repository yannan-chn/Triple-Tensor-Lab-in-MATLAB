function [tenv,info] = trip_gevdcomp(tent,rank)

[I,J,K] = size(tent);
if rank*rank > min([I,J,K])
    error('Assumption min(I,J,K) >= R*R fails. \n    [I,J,K]=[%d,%d,%d] and Rank=%d',I,J,K,rank);
end

if min([I,J,K]) > rank*rank*2+1
    [tcom,U,V,W,infoc] = ten3_comp(tent,rank*rank);
    [tenv,infoe]       = trip_gevd(tcom,rank);
    
    tic;  [Ic,Jc,Kc] = size(tcom);
    L = rank;  M = L;  N = L;
    p = 1;    q = Ic*M*N;    xA = reshape(tenv(p:q),[Ic,M*N]);
    p = q+1;  q = q+L*Jc*N;  xB = reshape(permute(reshape(tenv(p:q),[L,Jc,N]),[2,1,3]),[Jc,L*N]);
    p = q+1;  q = q+L*M*Kc;  xC = reshape(tenv(p:q),[L*M,Kc]);

    tenv = [reshape(U*xA,[I*M*N,1]); ...
            reshape(permute(reshape(V*xB,[J,L,N]),[2,1,3]),[L*J*N,1]); ...
            reshape(xC*W.',[L*M*K,1])];
    
    fprintf('Finally, gevd+comp results in ...... \n');
    info.reErr = infoc.err + infoe.reErr;
    info.CPUtm = toc + infoc.cput + infoe.CPUtm;
    fprintf('  Relative error of fitting      ---  %e\n',info.reErr);
    fprintf('  CPU time (second)              ---  %e\n',info.CPUtm);
else
    [tenv,info] = trip_gevdcomp(tent,rank);
end