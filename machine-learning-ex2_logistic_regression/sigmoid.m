function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

m = size(z, 1);
n = size(z, 2);
g = zeros(size(m, n));

for i = 1:m
    for j = 1:n
        g(i, j) = 1 / (1 + exp(-z(i, j)));
    end
end

% =============================================================

end
