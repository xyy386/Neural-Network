clear; clc; close all;

% ====== 生成简单线性可分数据 ======
N = 100;
% 类 +1
X1 = [randn(1, N/2)*0.3 + 1; randn(1, N/2)*0.3 + 1];
% 类 -1
X2 = [randn(1, N/2)*0.3 - 1; randn(1, N/2)*0.3 - 1];

X = [X1, X2];          % 2×N
y = [ones(1, N/2), -ones(1, N/2)];

% ====== 训练感知器 ======
[w, b] = perceptron_train(X, y, 100, 1.0);

% ====== 预测 & 计算准确率 ======
y_pred = perceptron_predict(X, w, b);
acc = sum(y_pred == y) / length(y);
fprintf('Experiment 1 accuracy: %.2f%%\n', acc*100);

% ====== 画图 ======
figure; hold on; grid on;
plot(X1(1,:), X1(2,:), 'o', 'DisplayName', 'Class +1');
plot(X2(1,:), X2(2,:), 'x', 'DisplayName', 'Class -1');

% 画出决策边界 w1*x + w2*y + b = 0
x_min = min(X(1,:)) - 1;
x_max = max(X(1,:)) + 1;
xx = linspace(x_min, x_max, 100);
if abs(w(2)) > 1e-6
    yy = -(w(1)*xx + b)/w(2);
    plot(xx, yy, '-', 'LineWidth', 2, 'DisplayName', 'Decision boundary');
end

legend;
title(sprintf('Experiment 1: Linear separable data (Acc = %.2f%%)', acc*100));
xlabel('x_1'); ylabel('x_2');
