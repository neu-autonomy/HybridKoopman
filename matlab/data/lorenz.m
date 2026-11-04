
numTest_ICs = 5000;
traj_len = 121;
filenamePrefix = 'Lorenz';


seed = 1;
X_test = LorenzFn(28, 10, 8/3,numTest_ICs, traj_len, seed);
filename_test = strcat(filenamePrefix, '_test_x.csv');
dlmwrite(filename_test, X_test, 'precision', '%.14f')

seed = 2;
X_val = LorenzFn(28, 10, 8/3,numTest_ICs, traj_len, seed);
filename_val = strcat(filenamePrefix, '_val_x.csv');
dlmwrite(filename_val, X_val, 'precision', '%.14f')

for j = 1:1
	seed = 2+j;
	X_train = LorenzFn(28, 10, 8/3,numTest_ICs, traj_len, seed);
	filename_train = strcat(filenamePrefix, sprintf('_train%d_x.csv', j));
	dlmwrite(filename_train, X_train, 'precision', '%.14f')
end
