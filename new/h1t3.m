clear; clc;

% ========== 生成 10 维线性可分数据 ==========
d = 10;      % 维度
N = 200;     % 样本数

% 类 +1：高斯分布中心 +1
X_pos = randn(d, N/2) * 0.3 + 1;

% 类 -1：高斯分布中心 -1
X_neg = randn(d, N/2) * 0.3 - 1;

X = [X_pos, X_neg];               % 10×200
y = [ones(1,N/2), -ones(1,N/2)];  % 标签

% ========== 训练感知器 ==========
[w, b] = perceptron_train(X, y, 200, 0.5);   % 使用前面提供的 perceptron_train.m

% ========== 预测 ==========
y_pred = perceptron_predict(X, w, b);        % 使用 perceptron_predict.m
acc = sum(y_pred == y) / length(y);

fprintf('10D example accuracy: %.2f%%\n', acc*100);

% ========== 显示权重信息 ==========
disp('Learned weight vector w:')
disp(w')

fprintf('Bias b = %.4f\n', b);
