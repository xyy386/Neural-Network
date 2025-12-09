clear; clc; close all;

%% ===== 生成 2D 数据：t = 1.5 x1 - 0.7 x2 + 0.5 + noise =====
N = 200;
x1 = rand(1, N) * 4 - 2;    % [-2,2]
x2 = rand(1, N) * 4 - 2;

noise = 0.2 * randn(1, N);
t = 1.5 * x1 - 0.7 * x2 + 0.5 + noise;

X = [x1; x2];   % d=2
T = t;          % m=1

%% ===== 训练 BP 网络 =====
n_hidden   = 15;
max_epochs = 300;
eta        = 0.02;

[W1, b1, W2, b2, mse_history] = bp_train(X, T, n_hidden, max_epochs, eta);

%% ===== 预测 =====
Y = bp_predict(X, W1, b1, W2, b2);

fprintf('实验2：最终 MSE = %.6f\n', mse_history(end));

%% ===== 可视化：3D 拟合 + MSE 曲线 =====
figure;

% 1) 三维样本点 & 拟合曲面
subplot(1,2,1);
hold on; grid on;
scatter3(x1, x2, t, 30, 'b', 'filled', 'DisplayName', '目标 t');

[x1g, x2g] = meshgrid(linspace(min(x1), max(x1), 25), ...
                      linspace(min(x2), max(x2), 25));
Xg = [x1g(:)'; x2g(:)'];
Yg = bp_predict(Xg, W1, b1, W2, b2);
Yg = reshape(Yg, size(x1g));

mesh(x1g, x2g, Yg);
alpha 0.5;
colormap jet;
xlabel('x1'); ylabel('x2'); zlabel('t / y');
title('实验2：BP 拟合 2D→1D 平面');
legend;

% 2) MSE 曲线
subplot(1,2,2);
plot(1:max_epochs, mse_history, 'LineWidth', 2);
xlabel('Epoch'); ylabel('MSE');
title('训练误差曲线');
grid on;
