clear; clc; close all;

%% ====== 生成 2D 数据：t = 1.5 x1 - 0.7 x2 + 0.5 + noise ======
N = 200;
x1 = rand(1, N) * 4 - 2;    % [-2, 2]
x2 = rand(1, N) * 4 - 2;    % [-2, 2]

noise = 0.2 * randn(1, N);
t = 1.5 * x1 - 0.7 * x2 + 0.5 + noise;  % 1×N

X = [x1; x2];   % 2×N
T = t;          % 1×N

%% ====== 训练 Adaline ======
max_epochs = 100;
eta = 0.01;
[W, b, mse_history] = adaline_train(X, T, max_epochs, eta);

fprintf('Learned weights: W = [%.4f, %.4f], b = %.4f\n', W(1), W(2), b);

%% ====== 预测 ======
Y = adaline_predict(X, W, b);

%% ====== 可视化 ======
% 1) 三维散点 + 拟合平面
figure;
subplot(1,2,1);
hold on; grid on;
scatter3(x1, x2, t, 30, 'b', 'filled', 'DisplayName', '目标 t');

% 创建网格来画平面
[x1g, x2g] = meshgrid(linspace(min(x1), max(x1), 20), ...
                      linspace(min(x2), max(x2), 20));
Xg = [x1g(:)'; x2g(:)'];
Yg = adaline_predict(Xg, W, b);
Yg = reshape(Yg, size(x1g));

mesh(x1g, x2g, Yg);
alpha 0.5;
colormap jet;
xlabel('x1'); ylabel('x2'); zlabel('t / y');
title('实验2：2D→1D 平面拟合');
legend;

% 2) MSE 学习曲线
subplot(1,2,2);
plot(1:max_epochs, mse_history, 'LineWidth', 2);
xlabel('Epoch');
ylabel('MSE');
title('训练误差曲线');
grid on;
