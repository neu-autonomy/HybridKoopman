script_names = ["DiscreteSpectrumExample", "FluidFlowBox", "FluidFlowOnAttractor", "Pendulum"];
traj_lens = [400, 400, 400, 400];
for i = 1:length(script_names)
    clearvars -except script_names traj_lens i

    filename = script_names(i);
    Ti = traj_lens(i);

    run(filename + ".m");
    X = X_train';
    
    Xs=X(:, 1:1:Ti);

    keq=[1:1:Ti]; % these are the points where gain=1 is enforced
    kl=[1:1:Ti];  % these are points where rank(L) is penalized
    kn=5; % number of neighbors
    mg=6; % look for up to 5th order dynamics
    maxcount=3; % 5 iterations of reweight

    lambdag=1; lambda=10; lambda1=1; lambda2=1; 


    [Yn,Kn,Ldirn,Gn] = koopman_embed_new(Xs,kn,keq,kl,lambdag,lambda,lambda1,lambda2,mg,maxcount);

    z=iddata(Yn',[],1);
    m=ssest(z,7,'ts',1);  % identify a model
    
    pK = m.C*m.A*pinv(m.C'*m.C)*m.C';
    Yp=predict(m,z,1);Ypred=Yp.y'; %predict its one step ahead output

    % pK = m.C*m.A*pinv(m.C'*m.C)*m.C'; % gets the koopman through the use of a psudo-inverse, also ignoring noise
    % Yp = pK * Yn;

    [Xpval2,Linv,sL] = koopman_reconstruct(Xs,Yn,Ypred,3,0.1,10,0.1,maxcount);

    save(filename + "_solutions.mat", "Yn", "Kn", "Ldirn", "Gn", "Ypred", "Xpval2", "Linv", "sL", "pK");

end