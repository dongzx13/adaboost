function e = adaboost_error(X, y, k, a, d, alpha)
% adaboost_error: returns the final error rate of a whole adaboost
% 
% Input
%     X     : n * p matrix, each row a sample
%     y     : n * 1 vector, each row a label
%     k     : iter * 1 vector,  selected dimension of features
%     a     : iter * 1 vector, selected threshold for feature-k
%     d     : iter * 1 vector, 1 or -1
%     alpha : iter * 1 vector, weights of the classifiers
%
% Output
%     e     : error rate      

%%% Your Code Here %%%
iter = size(a,1);
[m,n] =size(X);

y_our = zeros(m,iter);
for i = 1:iter
    p = ((X(:, k(i)) <= a(i)) - 0.5) * 2 * d(i); % predicted label
    %e = sum((p ~= y) .* w);
    %result = ones(m,1)*(-d(i));
    %result((X(:,k(i))-a(i))<=0) = d(i);
    %y_our = y_our+result;
    result = p;
    alpha(i)
    y_our(:,i)= alpha(i)*result;
    
   % y_our = y_our+alpha(i)*result;
y_final = sum(y_our,2);
y_final(y_final>=0) = 1;
y_final(y_final<0) = -1;
e = sum(y_final ~=y)/m;
%%% Your Code Here %%%

end