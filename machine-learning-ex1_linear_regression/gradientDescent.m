function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    s1 = 0;
    s2 = 0;
    for j = 1:m
        s1 = s1 + (dot(theta, X(j,:)) - y(j)) * X(j,1);
        s2 = s2 + (dot(theta, X(j,:)) - y(j)) * X(j,2);
    end
    s1 = (alpha / m) * s1;
    s2 = (alpha / m) * s2;
    theta(1) = theta(1) - s1;
    theta(2) = theta(2) - s2;
    J_history(iter) = computeCost(X, y, theta);
end

end
