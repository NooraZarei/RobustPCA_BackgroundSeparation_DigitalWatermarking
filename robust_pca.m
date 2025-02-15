function [L, S] = robust_pca(X, lambda, mu, tol, max_iter)

    [M, N] = size(X);
    unobserved = isnan(X);
    X(unobserved) = 0;
    normX = norm(X, 'fro');

    L = zeros(M, N);
    S = zeros(M, N);
    Y = zeros(M, N);
    
    for iter = (1:max_iter)
        L = Do(1/mu, X - S + (1/mu)*Y);
        S = So(lambda/mu, X - L + (1/mu)*Y);
        Z = X - L - S;
        Z(unobserved) = 0;
        Y = Y + mu*Z; 
        err = norm(Z, 'fro') / normX;
        if (iter == 1) || (mod(iter, 10) == 0) || (err < tol)
            fprintf(1, 'iter: %04d\terr: %d\n',iter, err);
        end
        if (err < tol) 
            break; 
        end
    end
end

function r = So(tau, X)
    r = sign(X) .* max(abs(X) - tau, 0);
end

function r = Do(tau, X)
    [U, S, V] = svd(X, 'econ');
    r = U*So(tau, S)*V';
end