function [Xpred,LL,sLL] = koopman_reconstruct(X,Y,Ypred,knew,lambda,lambda1,lambda2,maxcount)

%function [Xpred,L] = koopman_reconstruct(X,Ypred,kn, lambdag,lambda,lambda1,lambda2,maxcount)
% 
% Reconstructs the points in X space from the embeddings Ypred and data X
%
%
% %     INPUTS
%     
%     X: data matrix (columnwise)
%     Y: embeddings corresponding to X
%     Ypred: data to lift
%     kn: #of neighbours . (scalar)
%     lambda: weight for K
% .   lambda1: weight for spatial constraints
%     lambda2: weight for L
%     maxcount: max# iterations for reweighted heuristic
%
%  The total loss function is:
%  obj =lambda*nuc_norm(K)+ lambda1*sum((yj-yk)^2-(x_j-xk)^2)^2
%  ...+ lambda2*nuc_norm(L)
%   
%     OUTPUTS
%     
%     Xpred: reconstructed points in ambient space
%     L: stacked loewner matrices.
%     sL: vector of min singular values of L(i)
%
% .
% this assumes that the embedded Ys are in Y and the predicted in Ypred
% it takes each element of Ypred and find its corresponding X
%
% . Initial code by MS, 4/17/2020
%
npred=size(Ypred,2);
ny=size(Y,2);
Xpred=[];
LL= cell(1,npred);
Dsave= cell(1,npred);
sLL=[];
for ii=1:npred
Ynew=Ypred(:,ii);
%this is the predicted value of Y
%now find the neighboors of Ynew
%
Dnew=[];
for i=1:1:ny
    Dnew(i)=norm(Ynew-Y(:,i))^2;
end
[~, ind] = sort(Dnew,'ascend');
nnIndex = ind(1:knew);  % these are the neighbors
irep=min(size(find(Dnew(nnIndex)==0))); % dont want point to be its neighbor
nnIndex=ind(1+irep:knew+irep);
nnIndex=sort(nnIndex,'ascend'); % sort them to keep consistent labeling
Yn=Y(:,nnIndex); Xn=X(:,nnIndex);
Kny=Yn'*Yn; Knx=Xn'*Xn;  % Kernel entries corresponding to the neighbors
%
Knyy=[];Knxx=[];Kynew=[Ynew'*Ynew];
%
% vectorize to compute the Loewner matrix but
% keep only the unique elements
for i=1:1:knew
    Knyy=[Knyy;Kny(i:end,i)];
    Knxx=[Knxx;Knx(i:end,i)];
    Kynew=[Kynew; Ynew'*Yn(:,i)];
end

%
ml=(knew)*(knew+1)/2 ; % this is the number of elements in the L. matrix
%                       not related to the unknown



%
% L1 has the known portion of the Loewner matrix
W1=eye(knew+1,knew+1);
W2=eye(ml,ml);
W3=eye(knew+1,knew+1);
count=1;
while count<maxcount
cvx_begin sdp
    cvx_quiet true;
    cvx_solver SDPT3
    cvx_precision best
    variable L(knew+1,ml);
    variable Dh(1,knew+1);
    variable P(knew+1,knew+1);
    variable Q(ml,ml);
    variable Kd(knew+1,knew+1);
    Kd==semidefinite(knew+1);
    P == semidefinite(knew+1);
    Q == semidefinite(ml)
for i=1:knew+1;
        for j=1:ml;
          %  L(i,j)==(Dh(i)-Knxx(j))/(Kynew(i)-Knyy(j));
            L(i,j)==sign(Kynew(i)-Knyy(j))*(Dh(i)-Knxx(j))/(abs(Kynew(i)-Knyy(j))+1e-12) % avoid zeros
            % L(i,j)==sign(Kynew(i)-Knyy(j))*sign(Dh(i)-Knxx(j))*(abs(Dh(i)-Knxx(j))+1e-6)/(abs(Kynew(i)-Knyy(j))+1e-6) % avoid zeros
            
            
        end
end
obj1=0;
for i=1:1:knew;
       obj1=obj1+norm(Dh(1)-2*Dh(1+i)+Xn(:,i)'*Xn(:,i)-(Ynew'*Ynew-2*Ynew'*Yn(:,i)+Yn(:,i)'*Yn(:,i)));
end
 
    Dh(1)==Ynew'*Ynew;
    minimize lambda2*(trace(W1*P)+trace(W2*Q)) + lambda*trace(W3*Kd)+lambda1*obj1;
    %  minimize trace(W1*P)+trace(W2*Q);
    subject to
   [P L;L' Q] == semidefinite(knew+1+ml);
 %   Kd==[Dh;[Dh(2:end)' Xn'*Xn]];
   Kd==[Dh(1,1) Dh(2:end);[Dh(2:end)' Xn'*Xn]];
%    Kd+1e-3*eye(knew+1,knew+1)>=0;
cvx_end
sL=svds(L);
sLm=sL(end);
sK=svds(Kd);
sP=svd(P); sQ=svd(Q);
    W1 = inv(P + (1e-4+sP(end))*eye(knew+1));
   % W1 = inv(G + 1e-5*eye(m));
   % W1 = W1/norm(W1);
    W2 = inv(Q + (1e-4+sQ(end))*eye(ml));
    W3 = inv(Kd + (1e-4+sK(end))*eye(knew+1));
    W1=W1/(norm(W1));
    W2=W2/(norm(W2));
    W3=W3/norm(W3);
   % W2 = W2/norm(W2);
    count=count+1;
    %svd(L)
    %svd(Kd)
end
Xpred=[Xpred;Kd(1,2:end)/Xn];
LL{ii}=L;
%Dsave{ii}=Dh;
sLL=[sLL sL(end)/sL(1)];
end
end