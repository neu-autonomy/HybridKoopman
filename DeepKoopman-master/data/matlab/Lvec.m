function Lvec = Lvec(K)
%
% This function generates the vectorized data for the Loewner matrix corresponding to the Kernel
% data in the matrix K

n=size(K,2); % this is the number of elements in Y
ml=n*(n+1)/2;
Lvec=[];
for i=1:1:n
    Lvec=[Lvec;K(i:end,i)];
end
end

