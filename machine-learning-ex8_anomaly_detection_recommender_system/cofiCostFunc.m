function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);
nb_param = size(X,2);
            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta

%%% Version it�rative
% for j = 1:num_users
%     for i = 1:num_movies
%         if(R(i,j) == 1)
%             J = J + (Theta(j,:) * X(i,:)' - Y(i,j))^2; % Y(j,i) 
%         end
%     end
% end
% disp(J)
% J = (1/2) * J;
% disp(J)

%%% Version vectoris�e
% Calcul du co�t J
error_factor = (X * Theta' - Y) .^ 2;
error_factor = R .* error_factor;
J = (1/2) * sum(error_factor,'all');
reg = (lambda/2) * sum(Theta .^ 2,'all') + (lambda/2) * sum(X .^ 2,'all');
J = J + reg;

error_factor = R .* (X * Theta' - Y);
X_grad = error_factor * Theta;
Theta_grad = error_factor' * X;
regX = lambda * X;
regTheta = lambda * Theta;
X_grad = X_grad + regX;
Theta_grad = Theta_grad + regTheta;

% Calcul de X_grad
% for i = 1:num_movies
%     s = 0;
%     for k = 1:nb_param
%         for j = 1:num_users
%             if(R(i,j) == 1)
%                 s = s + (Theta(j,:) * X(i,:)' - Y(i,j)) * Theta(j,k); 
%             end
%         end    
%         X(i,k) = s;
%     end
% end
%%%%
% for i = 1:num_movies
%     X_grad(i) = ;
% end
% 
% for j = 1:num_users
%     Theta_grad(j) = ;
% end

grad = [X_grad(:); Theta_grad(:)];

end
