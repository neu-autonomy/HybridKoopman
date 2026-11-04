% script_names = ["DiscreteSpectrumExample", "FluidFlowBox", "FluidFlowOnAttractor", "Pendulum", "Lorenz"];
% traj_lens = [51, 101, 121, 51];
script_names = ["FluidFlowOnAttractor"];
traj_lens = [121];
do_normalization = false;
for name = 1:length(script_names)
    clearvars -except script_names traj_lens name do_normalization

    filename = script_names(name);
    Ti = traj_lens(name);

    X = readmatrix("../csv/" + filename + "_train1_x.csv")';
    if (do_normalization)
        min_X = min(X,[],"all");
        max_X = max(X,[],"all");
        X = 2*(X-min_X)/(max_X-min_X) -1;
    end
    samples = Ti*1;
    Xs=X(:,1:1:samples);
keq=[1:1:samples]; % these are the points where gain=1 is enforced
% kl=[1:1:samples];  % these are points where rank(L) is penalized
kn=5; % number of neighbors
mg=6; % look for up to 5th order dynamics
maxcount=5; % iterations of reweight

% lambdag=1; lambda=10; lambda1=1; lambda2=1; % weight in the loss func.
%lambdag=100; lambda=1; lambda1=1; lambda2=1; % weight in the loss func.
% lambdag=1; lambda=0.1; lambda1=0.1; lambda2=1;
lambdag=1; lambda=10; lambda1=1; % lambda2=1; 

fprintf(name + "th Embedding... \n");
%
%
% let's go ahead and embedd
% lambdag: weight for G (to avoid numerical problems)
%     lambda: weight for K
% .   lambda1: weight for spatial constraints
%     lambda2: weight for L
%
tic
[Yn,Kn,Gn, gramRank] = koopman_embed_new(Xs,Ti, kn,keq,lambdag,lambda,lambda1,mg,maxcount);  %embedd
toc
%% Old Koop Code
% z=iddata(Yn',[],1);
% 
% % gramRank=2;
% 
% m=ssest(z,gramRank,'ts',1);  % identify a model
% Yp=predict(m,z,1);Ypred=Yp.y'; %predict its one step ahead output
% koop = [];
% for i = 1:size(m.C, 1)
%     obsv_i = obsv(m.A, m.C(i,:));
%     k = obsv_i * m.A * inv(obsv_i);
%     k = m.C * m.A * inv(obsv_i);
%     koop = blkdiag(koop,k);
% end
%% New Koop Code
% gramRank = 3
% H = [Yn(:,1:Ti-gramRank)];
% for i = 2:gramRank
%     H =[H;Yn(:,i:Ti-gramRank-1+i)];
% end
% H = hank(Yn(:,1:Ti), gramRank);
% z = iddata(H',[],1);
% m=ssest(z,size(H,1),'Ts',0);  % identify a model
% koop = m.C*m.A*inv(m.C);
%% New Koop Code 2
% % gramRank = 3
H = hank(Yn(:,1:Ti), gramRank);
H1 = Yn(:,1:Ti-1);
H2 = Yn(:,2:Ti);
H1 = H(:,1:Ti-1-gramRank);
H2 = H(:,2:Ti-gramRank);
koop = H2*pinv(H1);
% %% New Koop Code 3
% H1 = [];
% H2 = [];
% for i = 1:dim(1)
%     H1 =[H1, hank(Yn(i, 1:Ti-1), gramRank )];
%     H2 =[H2, hank(Yn(i, 2:Ti), gramRank )];
% end
% [U,S,V] = svd(H1);
% A = U'*H2*V'*inv(S);
% koop = H2*pinv(H1);
% % hankel should be [hankel dim 1, hankel dim 2], do svd over this
%%
pK = koop;
eigs = eig(koop);
E = [];
for i = 1:size(eigs,1)
    e = eigs(i);
    if abs(imag(e)) < 1e-8
        E = [E; log(real(e))];
    else
        E = [E; log(norm(e)) + atan2(imag(e), real(e))*1j];
    end
end
dim = size(koop) / gramRank;
save(filename + "_observables.mat", "Yn", "Kn", "Gn","koop", "gramRank", "E", "dim")
fprintf(name + "th Embedding Done\n");

end
