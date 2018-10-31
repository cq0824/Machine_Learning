function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X*theta);   %the hypothesis of predictions
J = 1/m*(-log(h)'*y - log(1-h)'*(1-y)) + lambda/(2*m).*(theta(2:end)'*theta(2:end)); %the cost function of the thetas

% grad(1) = 1/m*(X(1,:)*(h-y));   %for j = 0, 
% grad(2:end) = 1/m*(h-y)'*X(2:end,:) + lambda/m*theta(2:end);    %for j>=1

for j = 1:length(grad)
   if j == 1
       grad(j) = 1/m*(X(:,j)'*(h-y));    %for the theta of zero
   else
       grad(j) = 1/m*X(:,j)'*(h-y) + lambda/m*theta(j);  %for the theta of one to end
   end
end

% =============================================================

end
