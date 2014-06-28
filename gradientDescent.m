function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

n = length(theta);
temp = zeros(n, 1);

for iter = 1:num_iters

    
    %Calculate the derivative, d/dTheta J_theta
    %This minimises to 1/m sum error times x_i
    
    %The h_theta function is the same as used in the cost function
    h_theta = X*theta;
    
    %The error is difference between the predicted value of h_theta and
    %real values
    error = h_theta - y;

    %Loop for each theta, allowing for simulataneous update.
    for i = 1:n
        temp(i) = theta(i) - alpha * (1/m) * sum (error .* X(:,i));
    end
    
    theta = temp;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
