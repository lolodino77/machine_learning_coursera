function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.

list_C = [0.01,0.03,0.1,0.3,1,3,10,30];
list_sigma = [0.01,0.03,0.1,0.3,1,3,10,30];
results = zeros(length(list_C) * length(list_sigma), 3) ;  
tol = 0.001;
max_passes = 10;
row = 1;
for i = 1:length(list_C)
    for j = 1:length(list_sigma)
        [model] = svmTrain(X, y, list_C(i), @(x1, x2) gaussianKernel(x1, x2, list_sigma(j)), tol, max_passes);
        predictions = svmPredict(model, Xval);
        prediction_error = mean(double(predictions ~= yval));
        results(row, :) = [list_C(i) list_sigma(j) prediction_error];
        row = row + 1;
   end
end
[min_error ind] = min(results(:,3)); %ind du minimum des erreurs de prédictions
res = results(ind,:);
C = res(1);
sigma = res(2);

end
