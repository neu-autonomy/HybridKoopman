function [Y, K, G, gramRank] = koopman_embed(X, T, k, keq, lambdag, lambda, lambda1, m, maxcount)
%KOOPMAN_EMBED  Embed trajectories with low-order linear dynamics by iterative reweighted SDP.
%
%   X        data matrix, one sample per column, trajectories of length T stored consecutively
%   T        length of each trajectory
%   k        number of nearest neighbours whose distances are preserved
%   keq      sample indices where the kernel diagonal equals that of the data (gain 1)
%   lambdag  weight of the rank surrogate of the Gram matrices of the Hankel matrices
%   lambda   weight of the rank surrogate of the kernel
%   lambda1  weight of the neighbour distance term
%   m        largest dynamics order considered
%   maxcount maximum number of reweighting iterations
%
%   Y        embedded samples (rows that hold 99% of the kernel spectrum)
%   K        kernel of the embedded samples
%   G        Gram matrix of the Hankel matrix of each trajectory
%   gramRank order of the dynamics: largest 99% rank over the trajectories
%
%   Requires CVX with the SDPT3 solver.

n = size(X, 2);
if mod(n, T) ~= 0
    error("Each trajectory in X must have the same length");
end
p = n / T;

D = zeros(n, n);
for i = 1:n-1
    for j = i+1:n
        D(i, j) = norm(X(:, i) - X(:, j))^2;
        D(j, i) = D(i, j);
    end
end
Kx = X' * X;
Eta = getNNmap(D, k);

W1 = cell(1, p);
for i = 1:p
    W1{i} = eye(m);
end
W2 = eye(n);
K = zeros(n, n);
K_pre = ones(n, n);
count = 1;
while norm(K - K_pre, 'fro') > 1e-4 && count < maxcount
    K_pre = K;

    cvx_begin sdp
    cvx_quiet true
    cvx_solver SDPT3
    cvx_precision low
    variable K(n, n)
    K == semidefinite(n);
    for ii = keq
        K(ii, ii) == Kx(ii, ii);
    end

    % Preserve the squared distances to the nearest neighbours.
    obj_space = 0;
    for i = 1:n
        for j = 1:n
            if Eta(i, j) == 1 && i ~= j
                obj_space = obj_space + norm((K(i, i) + K(j, j) - 2 * K(i, j)) - D(i, j));
            end
        end
    end

    % Gram matrix of the order-m Hankel matrix of each trajectory.
    objg = 0;
    G = cell(1, p);
    for i = 1:p
        G{i} = zeros(m, m);
        for j = (i-1)*T+1 : i*T-m
            G{i} = G{i} + K(j:j+m-1, j:j+m-1);
        end
        objg = objg + trace(W1{i} * G{i});
    end

    minimize(lambdag * objg + lambda * trace(W2 * K) + lambda1 * obj_space)
    cvx_end

    if issparse(K)
        K = full(K);
    end

    % Reweighting for the log-det rank heuristic.
    for i = 1:p
        W1{i} = inv(G{i} + 1e-5 * eye(m));
        W1{i} = W1{i} / norm(W1{i});
    end
    W2 = inv(K + 1e-5 * eye(n));
    W2 = W2 / norm(W2);

    count = count + 1;
    fprintf("reweighting iteration %d\n", count);
end

[~, S, V] = svd(K);
R = S.^0.5 * V';
c = cumsum(diag(S)) / sum(diag(S));
Y = R(1:nnz(c < 0.99) + 1, :);

gramRank = 0;
for i = 1:p
    si = svd(G{i});
    ci = cumsum(si) / sum(si);
    gramRank = max(gramRank, nnz(ci < 0.99) + 1);
end
end
