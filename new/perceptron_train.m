function [w, b] = perceptron_train(X, y, max_epochs, eta)
% 感知器训练
% X: d×N 的输入矩阵，每列一个样本（d 维，N 个样本）
% y: 1×N 的标签向量，取值为 +1 或 -1
% max_epochs: 最大迭代轮数
% eta: 学习率
%
% 返回:
%   w: d×1 的权重向量
%   b: 标量偏置

    [d, N] = size(X);
    w = zeros(d, 1);
    b = 0;

    if nargin < 3
        max_epochs = 100;
    end
    if nargin < 4
        eta = 1.0;
    end

    for epoch = 1:max_epochs
        errors = 0;
        for i = 1:N
            xi = X(:, i);
            yi = y(i);
            % 预测
            y_pred = sign(w' * xi + b);
            if y_pred == 0
                y_pred = 1; % sign(0) 在 MATLAB 中是 0，这里强制成正类
            end
            % 若分类错误则更新
            if yi * (w' * xi + b) <= 0
                w = w + eta * yi * xi;
                b = b + eta * yi;
                errors = errors + 1;
            end
        end
        % 可选：若本轮没有错误，提前结束
        if errors == 0
            fprintf('Converged at epoch %d\n', epoch);
            break;
        end
    end
end
