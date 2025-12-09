function Y = bp_predict(X, W1, b1, W2, b2)
% 一隐层 BP 网络预测
% X: d×N
% 返回:
%   Y: m×N

    [~, N] = size(X);
    Z1 = W1 * X + b1 * ones(1, N);
    A1 = 1 ./ (1 + exp(-Z1));      % sigmoid
    Y  = W2 * A1 + b2 * ones(1, N);
end
