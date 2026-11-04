function [Y,K,G, gramRank] = koopman_embed_new(X, T, k,keq,lambdag,lambda,lambda1,m,maxcount)
%     
%function [Y,K,LL,G] = koopman_embed(X, k,keq,kl,lambdag,lambda,lambda1,lambda2,m,maxcount)
%     INPUTS
%     
%     X: data matrix (columnwise)
%     T: length of each trajectory (should be T or q?)
%     k: #of neighbours . (scalar)
%    keq: indexes of points where the equality constraint is enforced
%     kl: indexes of points where the L matrix is considered
%     m: order of the dynamics
%     lambdag: weight for G (to avoid numerical problems)
%     lambda: weight for K
% .   lambda1: weight for spatial constraints
%     lambda2: weight for L
%
%  The total loss function is:
%  obj = lambdag*trace(W1*G) +
%  lambda*trace(W2*K)+lambda2*obj1+lambda1*sum((yj-yk)^2-(x_j-xk)^2)^2
%     
%     OUTPUTS
%     
%     Y: mapped points in the manifold
%     K: Kernel of the embedded data points
%     LL: cell array containg the Lowner matrices. LL{i} = Loewner at point i 
%     G:  Grammian of the hankel
%
%  Initial code by MS, 4/17/2020
%
n = size(X,2);      % # of samples
p = n / T;          % # of trajectories
nkeq=max(size(keq));% these are the points where the gain=1 constraint is enforced
% nl=max(size(kl));  % points where the L matrix is considered      %%%

% Sanity check for T and p
if mod(n, T) ~= 0
    error("Each trajectory in X must have the same length");
end

D=zeros(n,n);
for i=1:1:n-1
    for j=i+1:n
        D(i,j)=norm(X(:,i)-X(:,j))^2;
        D(j,i)=D(i,j);
    end
end
   Kx=X'*X;
Eta = getNNmap(D, k);% k NNs are in each column;
% nncell=cell(1,n);  % nncell                                       %%%
% for i=1:1:n; nncell{i}=find(Eta(:,i)==1);end
% ml=k*(k+1)/2;  % number of elements in each L matrix
%

% LX=zeros(ml,nl);                                                  %%%
% for i=1:1:nl
%        LX(:,i)=Lvec(Kx(nncell{kl(i)},nncell{kl(i)}));
% end
% elements of the Loewner matrix in X

% W1 = eye(m);

W1 = cell(1, p);
for i=1:1:p; W1{i}=eye(m);end   % generates a weight matrix for each traj

W2 = eye(n);
K = zeros(n, n);
K_pre = ones(n, n);
count=1;
while norm(K - K_pre,'fro') > 1e-4 && count<maxcount
    K_pre = K;
   
    cvx_begin sdp;
    cvx_quiet true;
    cvx_solver SDPT3
    cvx_precision low
    variable K(n, n);
    % variable L(k,ml-k,nl);    %%%
    % variable P(k,ml-k,n);         % Is this variable ever used in any version?
    % variable LY(ml,nl);       %%%
    K == semidefinite(n);
    for i=1:1:nkeq
        ii=keq(i);
        K(ii,ii)==Kx(ii,ii);
    end
 %  sum(K(:)) == Kxx;           %%%
  %  for i=1:1:nl;
  %      ii=kl(i);
  %      LY(:,i)==Lvec(K(nncell{ii},nncell{ii}));
  %  end
  % for i=1:1:nl
  %   %  L(:,:,i)==Loewner(LX,LY);
  %     L(:,:,i)==Loewner(LX(:,i),LY(:,i),k);
  % end
%%%%
obj_space=0;            % lipschitz constraint w/ nearest samples (Absorbed both constraints)
     for i = 1:n
      
        for j=1:n

                 if (Eta(i,j)==1)*(i~=j)
               
                obj_space=obj_space+ norm((K(i,i)+K(j,j)-2*K(i,j))-D(i,j));
              
                end
         end
     end
%%%%%
%%%%%%
    
    % G = zeros(m,m);
    % for i = 1:n-m
    %     G = G + K(i:i+m-1,i:i+m-1);
    % end

    objg = 0;
    G = cell(1, p);
    for i=1:1:p
        G{i} = zeros(m,m);
        for j=(i-1)*T+1:1:i*T-m
            G{i} = G{i} + K(j:j+m-1, j:j+m-1);
        end
        objg = objg + trace(W1{i} * G{i});
    end
    

    % obj1=0;               %%%
    % for i=1:1:nl
    %     obj1=obj1+norm_nuc(L(:,:,i));
    % end
 
    % obj = lambdag*trace(W1*G) + lambda*trace(W2*K)+lambda2*obj1+lambda1*obj_space;   %%%
    obj = lambdag*objg + lambda*trace(W2*K)+lambda1*obj_space;
 
    minimize(obj);
    
    cvx_end;

    if issparse(K)
        K = full(K);
    end
    
    % W1 = inv(G + 1e-5*eye(m));
    % W1 = W1/norm(W1);

    for i=1:1:p
        W1{i} = inv(G{i} + 1e-5*eye(m));
        W1{i} = W1{i}/norm(W1{i});
    end

    W2 = inv(K + 1e-5*eye(n));
    W2 = W2/norm(W2);

    count = count+1;
    fprintf("count: " + count + "\n")
end
[~,S,V] = svd(K);
R = S.^0.5 * V';
s = diag(S);
c = cumsum(s)/sum(s);
ind = nnz(c<0.99)+1;
Y = R(1:ind,:);

gramRank = 0;
for i = 1:p
    [~, Si, ~] = svd(G{i});
    si = diag(Si);
    ci = cumsum(si) / sum(si);
    r = nnz(ci<0.99)+1;
    if r > gramRank
        gramRank = r;
    end
end
% for i=1:1:nl
%     LL{i}=L(:,:,i);
% end

end