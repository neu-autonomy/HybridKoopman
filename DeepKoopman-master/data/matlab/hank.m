function H = hank(X, m)
H = [X(:,1:end-m)];
for i = 2:m
    H =[H;X(:,i:end-m-1+i)];
end
end