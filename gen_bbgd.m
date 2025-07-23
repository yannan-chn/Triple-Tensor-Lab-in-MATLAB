function [xn,info] = gen_bbgd(fun,xo)
% bbgd  Barzilai-Borwein gradient descent algorithm
% 
% Input:
%     fun  ---  evaluate function value and gradient vector
%     xo   ---  starting point
% Output
%     xn   ---  solution
%
% Yannan Chen    June 3, 2024
tic;  fprintf('Barzilai-Borwein gradient descent algorithm ... \n');

% parameters
maxiter = 200;        % iteration
epsStop = eps^(1/3);  % stoping tolerance

[fo,go] = fun(xo);  alpha = 0.0001;    % grad = [go,conj(go)]
fprintf('  iter |    function   |    gradient   | stepsize \n');
fprintf('  %4d | %13.6e | %13.6e ',0,fo,norm(go,inf));

reErr = [sqrt(abs(2*fo)),  zeros(1,maxiter)];
rdABC = zeros(1,maxiter+1);
CPUtm = [toc,              zeros(1,maxiter)];

% main loop
for iter=1:maxiter
    % Armijo line-search (faster)
    for inner=0:30
        xn = xo - alpha*go;    fn = fun(xn);
        if fo-fn >= 0.001*alpha*norm(go)^2
            [fn,gn] = fun(xn);  break;
        else
            alpha = alpha/2;
        end
    end
    
    % % Wolfe line search
    % alphaLow = 0;  phiLow = fo;  alphaHigh = 10*abs(alpha);  po = -go;
    % for iii=1:30
    %     fprintf('Wolfe LS: %4d | %10.8f | [%10.8f,%10.8f]\n',iii,alpha, alphaLow,alphaHigh);
    %     xn = xo + alpha*po;  [fn,gn] = fun(xn);
    %     phiD0 = real(dot(po,go));  phiDev = real(dot(po,gn));
    %     if fn > fo + 0.001*alpha*phiD0 || fn >= phiLow
    %         alphaHigh = alpha;
    %     else
    %         if abs(phiDev) <= -0.9*phiD0
    %             break;
    %         end
    %         if phiDev*(alphaHigh-alphaLow) >= 0
    %             alphaHigh = alphaLow;
    %         end
    %         alphaLow = alpha;  phiLow = fn;
    %     end
    %     
    %     alpha = -phiD0*alpha*alpha/(2*(fn-fo-phiD0*alpha));
    %     if (alpha-alphaLow)*(alpha-alphaHigh) >= 0
    %         alpha = 0.9*alphaLow + 0.1*alphaHigh;
    %     end
    % end
    fprintf('| %8.2e \n',alpha);
    fprintf('  %4d | %13.6e | %13.6e ',iter,fn,norm(gn,inf));
    
    % Check convergence
    foRe = sqrt(2*abs(fn));                   reErr(iter+1) = foRe;
    dx = alpha*go;  rdx = norm(dx)/norm(xn);  rdABC(iter+1) = rdx;
    if rdx < epsStop || foRe < epsStop*epsStop
        break;
    end
    
    % Here, a real step size is taken because alpha approximates the 
    % inverse of the Hessian matrix, which is a first-order Hermitian 
    % matrix and thus must be real.
    dy = go-gn;
    if randi(2)==1
        alpha = norm(dx)^2 / real(dot(dx,dy));
    else
        alpha = real(dot(dx,dy)) / norm(dy)^2;
    end
    alpha = min(max(1.5e-8,abs(alpha)),1000);
    xo = xn;  fo = fn;  go = gn;
    
    CPUtm(iter+1) = toc;
end
fprintf('\n');

CPUtm(iter+1) = toc;
info.Iter  = 0:iter;
info.reErr = reErr(1:iter+1);
info.rdABC = rdABC(1:iter+1);
info.CPUtm = CPUtm(1:iter+1);


