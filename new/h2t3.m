clear; clc; close all;

%% ====== 生成 3D→2D 线性映射数据 ======
N = 300;
d = 3;
m = 2;

% 随机输入
X = randn(d, N);

% 真实的线性变换参数（ground truth）
W_true = [ 1.0  -2.0   0.5;
          -0.3   0.8   1.2];    % 2×3
b_true = [0.5; -1.0];           % 2×1

noise = 0.1 * randn(m, N);      % 噪声

T = W_true * X + b_true + noise;  % 2×N

%% ====== 训练 Adaline（多输出） ======
max_epochs = 150;
eta = 0.01;
[W, b, mse_history] = adaline_train(X, T, max_epochs, eta);

fprintf('True W:\n');
disp(W_true);
fprintf('Learned W:\n');
disp(W);

fprintf('True b:\n');
disp(b_true');
fprintf('Learned b:\n');
disp(b');

%% ====== 预测 ======
Y = adaline_predict(X, W, b);  % 2×N

%% ====== 可视化：输出对比 & MSE 曲线 ======
figure;

% 输出维度1: t1 vs y1
subplot(2,2,1);
plot(T(1,:), Y(1,:), 'o');
xlabel('t_1 (target)'); ylabel('y_1 (predicted)');
title('输出维度1: 目标 vs 预测');
grid on;

% 输出维度2: t2 vs y2
subplot(2,2,2);
plot(T(2,:), Y(2,:), 'o');
xlabel('t_2 (target)'); ylabel('y_2 (predicted)');
title('输出维度2: 目标 vs 预测');
grid on;

% 残差直方图（两个输出放一起）
subplot(2,2,3);
e = T - Y;
histogram(e(1,:), 20); hold on;
histogram(e(2,:), 20);
legend('e_1', 'e_2');
title('残差分布');
grid on;

% MSE 学习曲线
subplot(2,2,4);
plot(1:max_epochs, mse_history, 'LineWidth', 2);
xlabel('Epoch'); ylabel('MSE');
title('训练误差曲线');
grid on;
