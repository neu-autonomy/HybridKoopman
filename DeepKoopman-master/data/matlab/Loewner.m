function L = Loewner(Lx,Ly,nrows)
%
% creates a single Loewner matrix with nrows rows.
% L:  L(i,j}=[y(i)-y(j)][x(i)-x(j)]
%
% input
%
% Lx,Ly     data to create L: should come from Lvec
% nrows   number of rows, if omitted then L is squarish
% kn      number of neighbors
%
% output
%
% . L:   Loewner matrix
%
%
% Initial code by M.S. 4/17/2020

ml=size(Lx,1);
if (nargin < 3)
    nrows = floor(ml/2);
end
    for i=1:1:nrows
        for j=nrows+1:1:ml
         L(i,j-nrows)=(Ly(i)-Ly(j))/(Lx(i)-Lx(j));
        end
    end
end



