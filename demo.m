clear
rng('default')

% tensor parameters
I = 400;  J = 400;  K = 400;  % size of 3D tensor
L = 5;                        % triple rank
sz = [I,J,K,L];

% generator a random triple tensor
randomFun = @(n) randn(n,1)+randn(n,1)*1i;
tenv = trip_rand(sz,randomFun);  tenNorm = trip_norm(tenv,sz);
tent = trip_full(tenv,sz);

% solver
% direct method (GEVD)
[estTv_gevd,info_gevd] = trip_gevd(tent,L); % info_gevd

% direct method + compression
[estTv_comp,info_comp] = trip_gevdcomp(tent,L);

% alternating least squares (ALS) minimization
xo = trip_rand(sz,randomFun);
[estTv_als,info_als] = trip_als(tent,xo);

% Barzilai-Borwein gradient descent algorithm
myfun = @(z) trip_evfg(z,tent,tenNorm);
[estTv_bbgd,info_bbgd] = gen_bbgd(myfun,xo);

figure(1)
semilogy(info_gevd.CPUtm,info_gevd.reErr,'rp' ...
    ,info_comp.CPUtm,info_comp.reErr,'ks' ...
    ,info_als.CPUtm,info_als.reErr,'bo-' ...
    ,info_bbgd.CPUtm,info_bbgd.reErr/tenNorm,'m+--','linewidth',2)
xlabel('CPU time (second)');  ylabel('relative error');
legend('gevd','comp+gevd','als','bbgd','location','SouthEast'), grid on


