function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

H = X * theta;    %-- 12 * 1
constant = 1/(2*m);
theta_reg = [0;theta(2:end)]; % we are not considering theta(0) term for regularized
%
% Calculating the Regularization Term
%
Regularization = lambda * constant * sum(theta_reg' * theta_reg);
%
% Calculating the Cost Term
%
J = constant * sum((H - y).^2) + Regularization;

%
% Regularised Linear Regression Gradient
%

##grad(1) = (1/m) * sum((H- y));
##grad(2:end) = ((1/m) * sum((H- y) .* X(:,2:end))) + ((lambda/m) * theta_reg(2:end));

% X ---> 12 * 2 (after  adding bias)
% H ---> 12 * 1
grad = (1/m)*(X'*(H-y)+lambda*theta_reg);   %superconfusing   --> (2 * 1)

% =========================================================================

grad = grad(:);

end
