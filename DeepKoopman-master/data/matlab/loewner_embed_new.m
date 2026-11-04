function Ynew = Loewner_embed(Xnew,X,Y,L)

%function Ypred = Loewner_embed(Xnew,X,Y)

% This function uses the rational function implicity defined by the
% Loewner matrix L to find the embeding Y corresponding to a given point
% Xnew
%
% inputs
%
% Xnew . point to embed (column vector)
% X,Y  data used to train the matrix L,  2xn_points matrices
% L    Loewner matrix to be used to find the rational mapping g:X-->Y
% kn   number of neighbors to use to find th
%
% Output
%  Ynew:  embedding of the point Xnew
%
% Initial code by M.S., 4/17/2020

%
reg=0.01; % regularization constant, you may have to change this !!
%
[nl,ml]=size(L); % use the same dimensions as in the original L matrix
kn = nl-1; % number of neighbors to use.
[U,S,V]=svd(L);
v=V(:,end);
nx=max(size(X));
Xr=[];
Yr=[];
% 
% first find the nearest neighbors
%
[nnIndex,Dnew]=knnsearch(X',Xnew','K',kn);
nnmin=nnIndex(1);  % this is the closest neighbor

if Dnew(1) < 1e-5 % point is in the training set, return it
    Ynew=Y(:,nnmin);
else  % go ahead and interpolate
    nnIndex=sort(nnIndex,'ascend'); % sort them to keep consistent labeling
%
% find the submatrices of Kx, Ky corresponding to the neighbors
%
    Ynn=Y(:,nnIndex); 
    Xnn=[Xnew X(:,nnIndex)];
    Ky=Ynn'*Ynn;
    Kx=Xnn'*Xnn;
%
% keep only the unique points and vectorize
    Lx=Lvec(Kx);
    Ly=Lvec(Ky);
%
% now estimate the elements of the Kernel matrix 
% we will estimate Ynew'*Y(ml+1:end)
%

    num=ones(kn,ml);
    den=ones(kn,ml);
    Kinterp=zeros(1,kn);
   


    for id=1:1:kn+1;  % don't need to interpolate K(1,1)
        for j=1:1:ml
            den(id,j)=1/(Lx(id)-Lx(kn+j+1));
            num(id,j)=Ly(j).*den(id,j);
        end
        Kinterp(id)=(num(id,:)*v)/(den(id,:)*v);
    end
% now estimate Ynew by factorization
%Ynew=Kinterp/Y(:,nnIndex);
    

 A=Y(:,nnIndex)';
 sa=norm(A*A');
     Kinterp1=Kinterp(2:end)';
     Ynear=Y(:,nnmin);
   % Ynew=Ynear-A'*inv(A*A'+reg*normeye(size(A*A')))*(A*Ynear-Kinterp1);
    Ynew=Ynear-A'*inv(A*A'+reg*sa*eye(size(A*A')))*(A*Ynear-Kinterp1);
%     
%     
%     A=Y(:,nnIndex);
%     Kinterp1=Kinterp(2:end)';
%     Ynew=inv(A*A')*A*Kinterp1;
%      cond(A*A')
%      Ynew=inv(A*A'+reg*eye(size(A*A')))*A*Kinterp1;
%
% This is a sanity check that can be eliminated later
%
% if ((Ynew'*Ynew)/(Xnew'*Xnew)>1.1)|((Ynew'*Ynew)/(Xnew'*Xnew)<0.9)
%      display(['Warning, Interpolant does not agree with Ynear, difference=' num2str((Ynew'*Ynew)/(Xnew'*Xnew))])
%      display('Regularizing')
%         %     keyboard
%        % Ynew=inv(A*A'+reg*norm(A*A')*eye(size(A*A')))*A*Kinterp1;
% 
% 
% display(['Regularized value =' num2str((Ynew'*Ynew)/(Xnew'*Xnew))])
%        Ytemp= [Ynew Y(:,nnIndex(1:end))];
%        Xtemp=  [Xnew X(:,nnIndex(1:end))];
%        Kyt=Ytemp'*Ytemp;
%        Kxt=Xtemp'*Xtemp;
%        Lxt=Lvec(Kxt);
%        Lyt=Lvec(Kyt);
%        nrows=size(L,1);
%        Lt=Loewner(Lxt,Lyt,nrows);
%       % nv=norm(Lt*v);
%           % if (nv>1e-1)
%           %  display(['Warning, Interpolant does not agree with L. matrix, nv=' num2str(nv)])
%            % keyboard
%              end
       end
end
