filename = "FluidFlowOnAttractor";
Ti = 121 * 2;
X = readmatrix(filename + "_train1_x.csv")';
Xs=X(:,1:1:Ti);


plot3(Xs(1,:),Xs(2,:), Xs(3,:),'r*')
hold on
xlabel('x1')
ylabel('x2')
zlabel('x3')
legend('ground truth')
title('Taj Viz')