function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = size(X, 2);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

J = (1 / (2*m)) * dot(X*theta - y, X*theta - y);
reg = (lambda / (2*m)) * dot(theta(2:end),theta(2:end));
J = J + reg;

% calcul du gradient
s = 0;
grad = (X * theta - y);
grad = X' * grad;
grad = (1 / m) * grad;
reg = (lambda / m) * theta;
reg(1) = 0;
grad = grad + reg;

%calcul du gradient (version itérative)
% s = 0;
% for j = 1:m
%     s = s + dot(X(j, :) * theta - y(j), X(j, 1));
% end
% grad(1) = (1 / m) * s;
% 
% for i = 2:n %jusqu'à n comme implicitement ici notre n est déjà égal à n+1
%     s = 0;
%     for j = 1:m
%         s = s + dot(X(j, :) * theta - y(j), X(j, i));
%     end
%     grad(i) = (1 / m) * s + (lambda / m) * theta(i);
% end

% =========================================================================

grad = grad(:);

end
