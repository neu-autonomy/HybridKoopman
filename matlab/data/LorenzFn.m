function X = LorenzFn(rho, sigma, beta, numICs, traj_len, seed)
% LORENZ Function generates the lorenz attractor of the prescribed values
% of parameters rho, sigma, beta
%
%   [X,Y,Z] = LORENZ(RHO,SIGMA,BETA,INITV,T,EPS)
%       X, Y, Z - output vectors of the strange attactor trajectories
%       RHO     - Rayleigh number
%       SIGMA   - Prandtl number
%       BETA    - parameter
%       INITV   - initial point
%       T       - time interval
%       EPS     - ode solver precision
%
% Example.
%        [X Y Z] = lorenz(28, 10, 8/3);
%        plot3(X,Y,Z);


% if nargin<3
%   error('MATLAB:lorenz:NotEnoughInputs','Not enough input arguments.'); 
% end
% if nargin<4
%   eps = 0.000001;
%   T = [0 50];
%   initV = [0 1 1.05];
% end

rng(seed)
eps = 0.000001;
T = [0 50];

X = zeros(numICs*traj_len, 3);
for i = 1:numICs
    initV = [(10*rand - 5) (10*rand - 5) (10*rand - 5)];
    options = odeset('RelTol',eps,'AbsTol',[eps eps eps/10]);
    [T,temp] = ode45(@(T,X) F(T, X, sigma, rho, beta), T, initV, options);
    X(1+(i-1)*traj_len : traj_len + (i-1)*traj_len,:) = temp(100:5:(100+5*traj_len)-1,:);
end
return
end
function dx = F(T, X, sigma, rho, beta)
% Evaluates the right hand side of the Lorenz system
% x' = sigma*(y-x)
% y' = x*(rho - z) - y
% z' = x*y - beta*z
% typical values: rho = 28; sigma = 10; beta = 8/3;
    dx = zeros(3,1);
    dx(1) = sigma*(X(2) - X(1));
    dx(2) = X(1)*(rho - X(3)) - X(2);
    dx(3) = X(1)*X(2) - beta*X(3);
    return
end
