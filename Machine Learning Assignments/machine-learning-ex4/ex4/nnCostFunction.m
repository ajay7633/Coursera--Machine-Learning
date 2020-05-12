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

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



%PART 1

a1 = [ones(m,1) X]  ;       %Adding Bias 1 to activation
z2 = a1* Theta1'    ;       % Z2 = A1 * THETA (Transpose)               --> (5000 * 401) * (401 * 25)
a2 = sigmoid(z2)    ;       %activation 2 found after sigmoid           -->
a2 = [ones(m,1) a2] ;       % Adding Bias to the activation at layer 2  --> 5000 * 26
z3 = a2 * Theta2'   ;       % Z# = A2 * THETA (Transpose)
a3 = sigmoid(z3)    ;       %activation of final layer                  -->5000 * 10
% The a3 we calculated is the H(x) for the output layer we need to calculate Cost function
H = a3 ;
constant = -(1/m);

% creation of y matrix , The current form is the unrolled version??  This is done so we can get 5000 * 10

y_matrix = eye(num_labels)(y,:);
%
%element-wise product of two matrices A and B.
%??A.?B=am+bn+co+dp+eq+fr
%

J = constant * sum(sum(((y_matrix.*log(H)) + ((1-y_matrix).* log(1-H)))));  %scalar  you have to elemental wise multiplication

%%
%% Now Calculating the regularization function
%%
%% Repeat the sameusing scalar multiplication
% Lets remove the row from theta1 and theta2
Theta1_without = [Theta1(:,2:end)];   %25 * 400
Theta2_without = [Theta2(:,2:end)] ;  %10 * 25
% Calculating First and Second term for the regularization function
 
first = sum(sum((Theta1_without .* Theta1_without)));
second = sum(sum((Theta2_without .* Theta2_without)));

R = (lambda /(2*m)) * (first + second);

%%%
%Final Cost Equation
J = J + R;

% -------------------------------------------------------------
% BACK PROPOGATION ALGORITHIM
%%

for i=1:m
  %  STEP 1  : Forward Propogation
     
      a1 = [ones(1) X(i,:)];          % a1 --> 1 * 401  
      z2 = a1 * Theta1';              % (1 * 401) * (25 * 401)T -->   1 * 25
      a2 = sigmoid(z2);               % 1 * 25
      a2 = [ones(1) a2];              % 1 * 26
      z3 = a2 *  Theta2';             % (1 * 26) * (10 *26)T    -->   1 * 10 
      a3 = sigmoid(z3);               % 1 * 10
      
  % STEP 2  : Calculating Delta - back propogation
    
      delta3 = a3 - y_matrix(i,:);    % 1 *10
      
      delta2 = delta3 * Theta2(:,2:end) .* sigmoidGradient(z2);%1 * 25
      %
      %Theta Gradient Accumulation
      %
      Theta2_grad = Theta2_grad + delta3' * a2;     % (10 * 1) * (1 *26)    --> 10 * 26;
      Theta1_grad = Theta1_grad + delta2' * a1;     % (25 * 1) * (1 * 401)  --> 25 * 401;
      
endfor
      
      
  Theta1_grad = Theta1_grad / m;                %-->25 * 401
  Theta2_grad = Theta2_grad / m;                %-->10 * 26
  
  %With regularization
  reg_Theta1 = Theta1(:,2:end) * (lambda / m);
  reg_Theta2 = Theta2(:,2:end) * (lambda / m);
  
  Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + reg_Theta1;
  Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + reg_Theta2;
 
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
