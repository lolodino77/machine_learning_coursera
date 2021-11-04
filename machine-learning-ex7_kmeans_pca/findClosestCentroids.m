function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
n_ex = size(X,1);
K = size(centroids, 1);
nc = size(centroids, 2);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
for i = 1:n_ex 
    dist_cent_ex = sum(((zeros(K, nc) + X(i,:)) - centroids).^2, 2);
%     disp("dist_cent_ex")
%     disp(dist_cent_ex)
    [minimum, idx_dist_min] = min(dist_cent_ex);
    idx(i) = idx_dist_min;
end




end

