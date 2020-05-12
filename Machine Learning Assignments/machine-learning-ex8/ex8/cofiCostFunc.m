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
%
## R: a matrix of observations (binary values). Dimensions are (movies x users)
##
## Y: a matrix of movie ratings: Dimensions are (movies x users)
##
## X: a matrix of movie features (0 to 5): Dimensions are (movies x features)
##
## Theta: a matrix of feature weights: Dimensions are (users x features)


# STEP 1
##Compute the predicted movie ratings for all users using the product of X and Theta. 
##Dimensions of the result should be (movies x users).
# movies - 5
# users - 3
predicted = X * Theta';    # 5 * 4 matrix
## STEP 2
movie_error = predicted - Y;
## STEP 3
# #Compute the "error_factor" my multiplying the movie rating error by the R matrix.
error_factor = movie_error .* R;
## Calculating the cost
##M = sum(sum((error_factor).^2))
##sum(M)
J = (1/2) * sum(sum((error_factor).^2));
## GRADIENT
X_grad = error_factor * Theta;
Theta_grad = error_factor' * X;

## Regularised term
term1 = sum(sum(Theta.^2));
term2 = sum(sum(X.^2));
Reg = term1 + term2;
J = J + ((lambda/2)*Reg);
## Regularised Gradient
X_grad = X_grad + lambda * X;
Theta_grad = Theta_grad + lambda *Theta;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
