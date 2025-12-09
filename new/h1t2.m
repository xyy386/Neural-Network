clear; clc; close all;

% ====== 生成近似线性可分数据（带噪声、部分重叠） ======
N = 200;

% 类 +1: 高斯分布中心 (1,1)
X1 = [randn(1, N/2)*0.6 + 1; randn(1, N/2)*0.6 + 1];
% 类 -1: 高斯分布中心 (-1,-1)
X2 = [randn(1, N/2)*0.6 - 1; randn(1, N/2)*0.6 - 1];

X = [X1, X2];
y = [ones(1, N/2), -ones(1, N/2)];

% ====== 训练感知器 ======
[w, b] = perceptron_train(X, y, 50, 0.5);

% ====== 预测 & 准确率 ======
y_pred = perceptron_predict(X, w, b);
acc = sum(y_pred == y) / length(y);
fprintf('Experiment 2 accuracy: %.2f%%\n', acc*100);

% ====== 画图 ======
figure; hold on; grid on;
plot(X1(1,:), X1(2,:), 'o', 'DisplayName', 'Class +1');
plot(X2(1,:), X2(2,:), 'x', 'DisplayName', 'Class -1');

% 决策边界
x_min = min(X(1,:)) - 1;
x_max = max(X(1,:)) + 1;
xx = linspace(x_min, x_max, 100);
if abs(w(2)) > 1e-6
    yy = -(w(1)*xx + b)/w(2);
    plot(xx, yy, '-', 'LineWidth', 2, 'DisplayName', 'Decision boundary');
end

legend;
title(sprintf('Experiment 2: Noisy data (Acc = %.2f%%)', acc*100));
xlabel('x_1'); ylabel('x_2');
