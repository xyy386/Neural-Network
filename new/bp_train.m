function [W1, b1, W2, b2, mse_history] = bp_train(X, T, n_hidden, max_epochs, eta)
% 一隐层 BP 网络训练（回归）
% X: d×N 输入矩阵（每列一个样本）
% T: m×N 目标输出矩阵（每列一个目标向量）
% n_hidden: 隐含层神经元个数
% max_epochs: 最大迭代轮数
% eta: 学习率
%
% 返回:
%   W1: n_hidden×d
%   b1: n_hidden×1
%   W2: m×n_hidden
%   b2: m×1
%   mse_history: 每轮 MSE

    [d, N] = size(X);
    [m, N2] = size(T);
    if N ~= N2
        error('X 和 T 的样本数不一致');
    end

    if nargin < 3, n_hidden = 10;      end
    if nargin < 4, max_epochs = 200;   end
    if nargin < 5, eta = 0.01;         end

    % 参数初始化（小随机数）
    rng('shuffle');
    W1 = 0.1 * randn(n_hidden, d);
    b1 = zeros(n_hidden, 1);
    W2 = 0.1 * randn(m, n_hidden);
    b2 = zeros(m, 1);

    mse_history = zeros(max_epochs, 1);

    for epoch = 1:max_epochs
        % ===== 前向传播 =====
        Z1 = W1 * X + b1 * ones(1, N);      % n_hidden×N
        A1 = 1 ./ (1 + exp(-Z1));           % sigmoid
        Y  = W2 * A1 + b2 * ones(1, N);     % m×N (线性输出)

        % ===== 误差 & MSE =====
        E = Y - T;                          % m×N
        mse = mean(E(:).^2);
        mse_history(epoch) = mse;
        % fprintf('Epoch %d, MSE = %.6f\n', epoch, mse);

        % ===== 反向传播 =====
        delta2 = E;                         % m×N （线性层）
        dW2 = (delta2 * A1') / N;           % m×n_hidden
        db2 = mean(delta2, 2);              % m×1

        delta1 = (W2' * delta2) .* A1 .* (1 - A1);  % n_hidden×N
        dW1 = (delta1 * X') / N;            % n_hidden×d
        db1 = mean(delta1, 2);              % n_hidden×1

        % ===== 参数更新 =====
        W2 = W2 - eta * dW2;
        b2 = b2 - eta * db2;
        W1 = W1 - eta * dW1;
        b1 = b1 - eta * db1;
    end
end
