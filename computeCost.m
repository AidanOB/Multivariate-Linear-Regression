function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
    m = length(y); % number of training examples

% You need to return the following variables correctly 
    J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
    
    %J = 1/2m sum 1:m (h_theta-y)^2

    %h function
    h_theta = X*theta;

    %h_theta-y = error
    error = h_theta - y;
    
    %Calculate the squared error
    squaredErr = error.^2;
    
    %Sum of the squared errors
    sumErr = sum(squaredErr);
    
    %Converting for the J value.
    J = (1/(2*m)) * sumErr;

% =========================================================================

end
