function [k, a, d] = decision_stump(X, y, w)
% decision_stump returns a rule ...
% h(x) = d if x(k) â‰?a, âˆ’d otherwise,
%
% Input
%     X : n * p matrix, each row a sample
%     y : n * 1 vector, each row a label
%     w : n * 1 vector, each row a weight
%
% Output
%     k : the optimal dimension
%     a : the optimal threshold
%     d : the optimal d, 1 or -1

% total time complexity required to be O(p*n*logn) or less

%%% Your Code Here %%%
[m,n] = size(X);
minerror = inf;
minerror_a = 0;
minerror_j = -1;
minerror_d = 0;
%w = ones(1,1000);
w = w';
for i = 1:n
    comp = repmat(X(:,i),1,m)';
    ori_rep = repmat(X(:,i),1,m);
    result_gt = double(ones(m,m));
    result_gt((ori_rep - comp)>=0) = -1.0;
    label_gt = repmat(y,1,m);
    [min_value_gt,argmin_gt] = min(w*(result_gt ~= label_gt));
    result_lt = double(ones(m,m));
    result_lt((ori_rep - comp)<=0) = -1.0;
    label_lt = repmat(y,1,m);
    [min_value_lt,argmin_lt] = min(w*(result_lt ~= label_lt));
    if min_value_lt < min_value_gt
        final_error = min_value_lt;
        final_a = X(argmin_lt,i);
        final_d =  -1.0;
    else
        final_error = min_value_gt;
        final_a = X(argmin_gt,i);
        final_d =  1.0;
    end
    if minerror > final_error
        minerror = final_error
        minerror_a = final_a;
        minerror_d = final_d;
        minerror_j = i;
    end
end
k = minerror_j;
a = minerror_a;
d = minerror_d;


end
