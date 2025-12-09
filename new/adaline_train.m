function [W, b, mse_history] = adaline_train(X, T, max_epochs, eta)
% 自适应线性网络（Adaline）训练
% X: d×N 输入矩阵，每列一个样本
% T: m×N 目标输出矩阵，每列一个目标向量
% max_epochs: 迭代轮数
% eta: 学习率
%
% 返回:
%   W: m×d 权重矩阵
%   b: m×1 偏置向量
%   mse_history: 每轮的均方误差

    [d, N] = size(X);
    [m, N2] = size(T);
    if N ~= N2
        error('X 和 T 的样本数不一致');
    end

    if nargin < 3, max_epochs = 100; end
    if nargin < 4, eta = 0.01; end

    % 初始化
    W = zeros(m, d);
    b = zeros(m, 1);
    mse_history = zeros(max_epochs, 1);

    for epoch = 1:max_epochs
        % 在线 LMS 更新
        for i = 1:N
            xi = X(:, i);      % d×1
            ti = T(:, i);      % m×1

            yi = W * xi + b;   % m×1
            ei = ti - yi;      % m×1

            % Widrow-Hoff (LMS) 更新
            W = W + eta * ei * xi';  % m×d
            b = b + eta * ei;        % m×1
        end

        % 计算本轮 MSE
        Y = W * X + b;        % m×N
        E = T - Y;            % m×N
        mse = mean(E(:).^2);
        mse_history(epoch) = mse;
        % 可选打印
        % fprintf('Epoch %d, MSE = %.6f\n', epoch, mse);
    end
end
