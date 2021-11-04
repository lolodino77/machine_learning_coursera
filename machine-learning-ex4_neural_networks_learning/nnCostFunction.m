function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Setup some useful variables
m = size(X, 1);
n = size(X, 2); % avant rajout de la colonne de 1
X_old = X;
X = [ones(m, 1) X];

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
    
%{
disp("Theta1")
disp(size(Theta1))
disp("Theta2")
disp(size(Theta2))
%}

     
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. 

%forward propagation (version vectorisée)
z2 = X*Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);
hTheta = a3;

%version vectorielle
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y,:);

for i = 1:m %boucle sur les exemples    
    %version itérative
    %on transforme le label y(i) en vecteur dont la i-ème composante vaut 1
%     y_i = zeros(10,1); % label unméro i, vecteur avec que des 0 sauf un 1 à la coordonnée i
%     yLabel = y(i);
%     y_i(yLabel) = 1;
    
%     disp("y_matrix");disp(size(y_matrix))
    %calcul de J dans la boucle
    for k = 1:num_labels
        s = - y_matrix(i,k) * log(hTheta(i,k)) - (1 - y_matrix(i,k)) * log(1 - hTheta(i,k));
        J = J + s;
    end
end
J = (1/m) * J;

%calcul du terme de régularisation
%calcul de sa première partie 
reg1 = Theta1(:,2:end) .* Theta1(:,2:end);
reg1 = sum(reg1, 'all');

%calcul de sa seconde partie
reg2 = Theta2(:,2:end) .* Theta2(:,2:end);
reg2 = sum(reg2, 'all');

%somme des deux parties
reg = reg1 + reg2;
reg = (lambda / (2 * m)) * reg;
J = J + reg;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients

delta3 = hTheta - y_matrix; % bon
delta2 = delta3 * Theta2(:,2:end) .* sigmoidGradient(z2); % on se sert pas de la colonne de 1 (biais) de Theta2
Delta1 = zeros(hidden_layer_size, n+1);
Delta2 = zeros(num_labels, hidden_layer_size+1);
%{
disp("delta2")
disp(size(delta2))
disp(delta2)
disp("delta3")
disp(delta3)
disp("X_old")
disp(size(X_old))
disp("Delta1")
disp(size(Delta1))
%}

Delta1 = Delta1 + delta2' * X;

%{
disp("delta3")
disp(size(delta3))
disp("a2")
disp(size(a2))
disp("Delta2")
disp(size(Delta2))
%}

Delta2 = Delta2 + delta3' * a2;
%{
disp("Delta1")
disp(Delta1)
disp("Delta2")
disp(Delta2)
%}

Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;    
%{
disp("Theta2(:,2:end)")
disp(size(Theta2(:,2:end)))
disp("Theta1(:,2:end)")
disp(size(Theta1(:,2:end)))
%}

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end);     
grad = [Theta1_grad(:) ; Theta2_grad(:)];
%{
disp(grad)
disp("J");disp(J)
%}

end
