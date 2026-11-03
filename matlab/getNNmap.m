function Eta = getNNmap(G,k)
% 
%     INPUT
%     G: pairwise distance matrix 
%     k: # of neighbours
%     
%     OUTPUT
%     Eta : nearest neighbor map, k nearest neighbour of each data point
%           is determined by s in each column
% 

assert(ismatrix(G));
assert(size(G,2)==size(G,2));

n = size(G,1);
Eta = zeros(n,n);
% [~, ind] = sort(G,'descend');
[~, ind] = sort(G,'ascend');
nnIndex = ind(1:k,:);
for i = 1:n
    Eta(nnIndex(:,i),i) = 1;
end

end