function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where

% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);
for i = 1:p
    if(i == 1)
        disp("X.^i")
        disp((X.^i)')
    end
    X_poly(:, i) = X.^i;
    if(i == 1)
        disp("X.^i")
        disp(X_poly(:, i)')
    end
end

end
