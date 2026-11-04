script_names = ["Pendulum"]; %["DiscreteSpectrumExample", "FluidFlowBox", "FluidFlowOnAttractor", "Pendulum"];
traj_lens = [400];

%
for i = 1:1
    clearvars -except script_names traj_lens i

    filename = script_names(1);
    Ti = traj_lens(1);

    X = readmatrix(filename + "_train1_x.csv")';
    Xs=X(:,1:1:Ti);
maxcount=3; % iterations of reweight

% results of embedding
load(filename + "_solutions.mat", "Yn","Ldirn");

% % %  Ldir are the Loewner matrices for the embedding X--> Y
% % % 
% % % l .  lambda: weight for K
% % % .   lambda1: weight for spatial constraints
% % %     lambda2: weight for L
% % %     maxcount: max# iterations for reweighted heuristic
% % 
% %  % best so far  
% %  % lambda1=1;lambda2=1; kn=3;lambda=10;
% %  % lambda1=1;lambda2=1; kn=4;lambda=10;
% %  lambda1=10;lambda2=1; kn=4;lambda=10;
% %  % lambdag=1; lambda=0.1; lambda1=0.1; lambda2=1;kn=4; % magic numbers !!!
% %  %
% %   fprintf("Reconstruction without prediction\n");
% %   tic
% %   [Xpval,~,~] = koopman_reconstruct(Xs,Yn,Yn,4,lambda,lambda1,lambda2,maxcount);
% %   toc
% %   fprintf("Reconstruction Done\n");
% %   %
% %   % 
% %  %Linv contains the local Loewner matrix for the inverse mapping Y-->X
% %  % sL has the the inverse of the condition numbers. Should be small
% %   %
% %   % do a sanity check to test if we are reconstructing well
% % %   %
% %   Xpval=Xpval';
% % 
% %   close all
% % 
% %   if i ==1 || i == 4
% %         plot(Xs(1,:),Xs(2,:),'r*')
% %         hold on
% %         plot(Xpval(1,:),Xpval(2,:),'bo')
% %         xlabel('x1')
% %         ylabel('x2')
% %         legend('ground truth', 'lifting of embedded training data')
% %         title('validation: embedding and lifting training data')
% %   else
% %         plot3(Xs(1,:),Xs(2,:),Xs(3,:),'r*')
% %         hold on
% %         plot3(Xpval(1,:),Xpval(2,:),Xpval(3,:),'bo')
% %         xlabel('x1')
% %         ylabel('x2')
% %         zlabel('x3')
% %         legend('ground truth', 'lifting of embedded training data')
% %         title('validation: embedding and lifting training data')
% %   end
 

% try a cheap reconstruction


 z=iddata(Yn',[],1);
 m=ssest(z,7,'ts',1);  % identify a model
 Yp=predict(m,z,1);Ypred=Yp.y'; %predict its one step ahead output

% pK = m.C*m.A*pinv(m.C'*m.C)*m.C'; % gets the koopman through the use of a psudo-inverse, also ignoring noise
% Ypred = pK * Yn;


fprintf("Reconstruction with Prediction \n");
load(filename + "_solutions.mat", "Xpval2","Linv");
fprintf("Reconstruction with Prediction Done \n");
%
% reconstruct the one step ahead prediction of the training data
% do a cheap reconstruction using the Loewner matrices 
if i == 1 || i == 4
    Xcheap=zeros(2,Ti);
else
    Xcheap=zeros(3,Ti);
end
fprintf("Cheap Reconstruction \n");
tic
   for j=1:1:Ti
       Xcheap(:,j)=loewner_embed_new(Ypred(:,j),Yn,Xs,Linv{j});
   end
toc
fprintf("Cheap Reconstruction Done \n");
%
% plot the one steo ahead predictions versus the ground truth
 % 
 if i ==1 || i == 4
    figure
    plot(Xs(1,:),Xs(2,:),'r*')
    hold on
    plot(Xcheap(1,:),Xcheap(2,:),'bo')
    xlabel('x1')
    ylabel('x2')
    title('one step ahead prediction of training data')
    legend('ground truth', 'lifting of one step ahead training data')
 else
    figure
    plot3(Xs(1,:),Xs(2,:),Xs(3,:),'r*')
    hold on
    plot3(Xcheap(1,:),Xcheap(2,:),Xcheap(3,:),'bo')
    xlabel('x1')
    ylabel('x2')
    zlabel('x3')
    title('one step ahead prediction of training data')
    legend('ground truth', 'lifting of one step ahead training data')
 end
   %
   % now implement the full pipeline with new data 
   % pipeline X(k) --> Y(k)---> Y(k+1)--> X(k)
   %
   th=Ti %prediction horizon
tstart=Ti+1; %start predicting here.

Xrecon=[]
Yrecon=[]
Yreconz=[];
Ynewsave=[];
Yfor=[];
Xf=[];
ts=1;

% retrieve the models and find the ARX model

r=size(m.A,1); % memory
mr=charpoly(m.A);
mr=fliplr(mr);
mr=mr(1:end-1)';


Xtopred=X(:,tstart:ts:tstart+ts*(th-1)); % this is the trajectory to predict

% At time k get Y for t=k,k-1,..k-r
%
fprintf("Reconstruction Novel Pred \n");
tic
for ii=r+1:1:th
Xm=Xtopred(:,-r+ii:-1+ii);  % past states

Ym=[];
for j=1:1:r;
    % 
    % given a point X to embed, we first find its closest neighbor
    % in the training data and use the corresponding L matrix
    %
    in=knnsearch(Xs',Xm(:,j)','K',1);    
    Ynew=loewner_embed_new(Xm(:,j),Xs,Yn,Ldirn{in});
    Ynewsave=[Ynewsave Ynew];
    Ym=[Ym Ynew];
end
% use Ym to predict new Y
%
Yfor=-[Ym(:,end-r+1:end)*mr];
%
z=iddata(Ym',[],1);
Yfz=forecast(m,z,1);Yforz=Yfz.y'; %predict its one step ahead output
%  Yfor is the predicted value for Y(k+1) 
%
Yrecon=[Yrecon Yfor];  % save all predictions
Yreconz=[Yreconz Yforz];
%


% now go back. Start by finding the nearest neighbor to Yfor
%

 in=knnsearch(Yn',Yforz','K',1);  

% now go back using the Loewner matrix of the inverse mapping.
%

 Xf=loewner_embed_new(Yforz,Yn,Xs,Linv{in});
 %
 % Xf is the predicted value of X(k+1)
 Xrecon=[Xrecon Xf];
end
toc
fprintf("Reconstruction Novel Pred Done \n");

if i ==1 || i == 4
    figure
    plot(Xtopred(1,:),Xtopred(2,:),'b*-')
    hold on
    plot(Xrecon(1,:),Xrecon(2,:),'ro')
    plot(Xtopred(1,1:r),Xtopred(2,1:r),'g*-')
    legend('new data','one step ahead prediction','initial conditions')
    xlabel('x1');ylabel('x2')
    title('One step ahead prediction of new data (x1,x2)')
else
    figure
    plot3(Xtopred(1,:),Xtopred(2,:),Xtopred(3,:),'b*-')
    hold on
    plot3(Xrecon(1,:),Xrecon(2,:),Xrecon(3,:),'ro')
    plot3(Xtopred(1,1:r),Xtopred(2,1:r),Xtopred(3,1:r),'g*-')
    legend('new data','one step ahead prediction','initial conditions')
    xlabel('x1');ylabel('x2')
    title('One step ahead prediction of new data')

end
end