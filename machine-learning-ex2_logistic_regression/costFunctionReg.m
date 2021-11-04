function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(X, 2); % number of parameters

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% calcul de la valeur du coût
reg = (lambda / (2 * m)) * dot(theta(2:end), theta(2:end)); %terme de régularisation, 
                                       % on part de 2 au lieu de 1 car on ne régularise pas la première variable
J = (1 / m) * ( dot(- y, log(sigmoid(X*theta))) - dot(1 - y, log(1 - sigmoid(X*theta))) ) + reg;

% calcul des valeurs du gradient de 1 à n+1 
% PS : les indices i et j sont inversés par rapport à l'énoncé
% le cas du premier paramètre (premier indice) qu'on ne régularise pas par
% convention
s = 0;
for j = 1:m
    s = s + dot(sigmoid(X(j, :) * theta) - y(j), X(j, 1));
end
grad(1) = (1 / m) * s;

% le cas de tous les autres paramètres qu'on régularise
for i = 2:n %jusqu'à n comme implicitement ici notre n est déjà égal à n+1
    s = 0;
    for j = 1:m
        s = s + dot(sigmoid(X(j, :) * theta) - y(j), X(j, i));
    end
    grad(i) = (1 / m) * s + (lambda / m) * theta(i);
end

% =============================================================

end
