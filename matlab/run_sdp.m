%RUN_SDP  Compute the SDP Koopman embedding and eigenvalues for each system.
%
%   Reads ../data/csv/{System}_train1_x.csv and writes ../data/sdp/{System}.mat with
%   Yn (embedding), Kn (kernel), Gn (Gram matrices), koop (Koopman matrix of the
%   Hankel embedding), gramRank (dynamics order), E (log-eigenvalues of koop) and dim.
%   The python code reads gramRank and E. Only the first training trajectory is used.

systems = ["DiscreteSpectrumExample", "FluidFlowOnAttractor", "Pendulum", "Lorenz"];
traj_lens = [51, 121, 51, 121];

kn = 5;         % number of neighbours
mg = 6;         % largest dynamics order considered
maxcount = 5;   % reweighting iterations
lambdag = 1; lambda = 10; lambda1 = 1;

for s = 1:numel(systems)
    name = systems(s);
    Ti = traj_lens(s);
    X = readmatrix("../data/csv/" + name + "_train1_x.csv")';
    Xs = X(:, 1:Ti);

    fprintf("%s: embedding\n", name);
    tic
    [Yn, Kn, Gn, gramRank] = koopman_embed(Xs, Ti, kn, 1:Ti, lambdag, lambda, lambda1, mg, maxcount);
    toc

    % Least-squares Koopman matrix on the Hankel matrix of the embedding.
    H = hank(Yn(:, 1:Ti), gramRank);
    koop = H(:, 2:Ti-gramRank) * pinv(H(:, 1:Ti-1-gramRank));

    ev = eig(koop);
    E = zeros(numel(ev), 1);
    for i = 1:numel(ev)
        if abs(imag(ev(i))) < 1e-8
            E(i) = log(real(ev(i)));
        else
            E(i) = log(abs(ev(i))) + 1j * atan2(imag(ev(i)), real(ev(i)));
        end
    end
    dim = size(koop) / gramRank;
    save("../data/sdp/" + name + ".mat", "Yn", "Kn", "Gn", "koop", "gramRank", "E", "dim")
end
