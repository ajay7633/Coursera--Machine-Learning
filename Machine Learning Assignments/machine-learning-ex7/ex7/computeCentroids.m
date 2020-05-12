function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%
#add= []
#mean =[]
for i =1:K
   
##   mean_x = sum(X(st,1))/length(st)
##   mean_y = sum(X(st,2))/length(st)
##   centroids(i,:) = [mean_x,mean_y]

##  mean_x = sum(X(find(idx == i),1))/length(find(idx == i));
##  mean_y = sum(X(find(idx == i),2))/length(find(idx == i));
##  centroids(i,:) = [mean_x mean_y];
    add = sum(X((idx ==i),:)) ;
    total = length(find(idx == i));
    centroids(i,:) = add ./ total;
  
  
   
endfor


% =============================================================


end