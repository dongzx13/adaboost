function [e_train, e_test] = adaboost(X, y, X_test, y_test, maxIter)
% adaboost: carry on adaboost on the data for maxIter loops
%
% Input 
%     X       : n * p matirx, training data
%     y       : n * 1 vector, training label
%     X_test  : m * p matrix, testing data
%     y_test  : m * 1 vector, testing label
%     maxIter : number of loops
%
% Output
%     e_train : maxIter * 1 vector, errors on training data
%     e_test  : maxIter * 1 vector, errors on testing data


w = (1 / size(y, 1)) * ones(size(y)); % initialize

k = zeros(maxIter, 1);
a = zeros(maxIter, 1);
d = zeros(maxIter, 1);
alpha = zeros(maxIter, 1);

e_train = zeros(maxIter, 1);
e_test = zeros(maxIter, 1);
for i = 1: maxIter
    [k(i), a(i), d(i)] = decision_stump(X, y, w);
    fprintf( 'new decision stump k:%d a:%d, d:%d\n', k(i), a(i), d(i));
    
    e = decision_stump_error(X, y, k(i), a(i), d(i), w);
    alpha(i) = log((1 - e) / e)/2;
    w = update_weights(X, y, k(i), a(i), d(i), w, alpha(i));
    e_train(i) = adaboost_error(X, y, k(1:i), a(1:i), d(1:i), alpha(1:i));
    e_test(i) = adaboost_error(X_test, y_test, k(1:i), a(1:i), d(1:i), alpha(1:i));
    fprintf( 'weak learner error rate: %f\nadaboost error rate: %f\ntest error rate: %f\n\n', e, e_train(i), e_test(i));
end
subplot(1,2,1);
plot(e_train);
subplot(1,2,2);
plot(e_test);
save error.mat e_train e_test
end