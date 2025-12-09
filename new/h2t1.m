clear; clc; close all;

%% ====== 生成 1D 数据：t = 2x + 1 + 噪声 ======
N = 100;
x = linspace(-1, 1, N);          % 1×N
noise = 0.1 * randn(1, N);       % 噪声
t = 2 * x + 1 + noise;           % 1×N

X = x;                           % d=1，X 就是 1×N
T = t;                           % m=1

%% ====== 训练 Adaline ======
max_epochs = 50;
eta = 0.05;
[W, b, mse_history] = adaline_train(X, T, max_epochs, eta);

fprintf('W(1) = %.4f, b = %.4f\n', W, b);

%% ====== 可视化：拟合直线 & 学习曲线 ======
% 预测
Y = adaline_predict(X, W, b);

figure;
subplot(1,2,1);
hold on; grid on;
plot(x, t, 'bo', 'DisplayName', '训练样本');
plot(x, Y, 'r-', 'LineWidth', 2, 'DisplayName', 'Adaline 拟合');
xlabel('x'); ylabel('t / y');
legend;
title('实验1：1D 线性拟合');

subplot(1,2,2);
plot(1:max_epochs, mse_history, 'LineWidth', 2);
xlabel('Epoch');
ylabel('MSE');
title('训练误差曲线');
grid on;
