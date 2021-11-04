function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
n = size(X, 2); % number of parameters

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
%input : X, y theta
% calcul de la valeur du coût
J = (1 / m) * ( dot(- y, log(sigmoid(X*theta))) - dot(1 - y, log(1 - sigmoid(X*theta))) );

% calcul des valeurs du gradient de 1 à n+1 
for i = 1:n %jusqu'à n comme implicitiement ici notre n est déjà égal à n+1
    s = 0;
    for j = 1:m
        s = s + dot(sigmoid(X(j, :) * theta) - y(j), X(j, i));
    end
    grad(i) = (1 / m) * s;
end
% =============================================================

end
