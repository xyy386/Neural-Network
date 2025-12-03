%% 实验1：AND逻辑问题
clear; clc; close all;

fprintf('实验1: AND逻辑问题\n');

% 准备训练数据
% 输入：2x4矩阵，每列是一个样本
P_and = [0 0 1 1;
         0 1 0 1];

% 目标输出：1x4矩阵
T_and = [0 0 0 1];

% 显示真值表
fprintf('\nAND逻辑真值表:\n');
fprintf('x1  x2  | Output\n');
fprintf('----------------\n');
for i = 1:4
    fprintf('%d   %d   |   %d\n', P_and(1,i), P_and(2,i), T_and(i));
end

% 创建感知器
% perceptron创建一个单层感知器
net = perceptron;

% 配置网络参数
net.trainParam.epochs = 20;      % 最大训练轮数
net.trainParam.goal = 0;         % 性能目标
net.trainParam.show = 5;         % 显示频率
net.trainParam.lr = 0.1;         % 学习率

% 训练网络
fprintf('\n开始训练...\n');
[net, tr] = train(net, P_and, T_and);

% 测试网络
Y_and = net(P_and);

% 显示结果
fprintf('\n训练结果:\n');
fprintf('x1  x2  | 目标  预测\n');
fprintf('----------------------\n');
for i = 1:4
    fprintf('%d   %d   |  %d     %d\n', ...
        P_and(1,i), P_and(2,i), T_and(i), round(Y_and(i)));
end

% 计算准确率
accuracy = sum(round(Y_and) == T_and) / length(T_and) * 100;
fprintf('\n准确率: %.2f%%\n', accuracy);

% 显示网络权重和偏置
fprintf('\n网络参数:\n');
fprintf('权重 W: [%.4f, %.4f]\n', net.IW{1,1}(1), net.IW{1,1}(2));
fprintf('偏置 b: %.4f\n', net.b{1});

% 可视化决策边界
figure('Name', 'AND问题的决策边界', 'NumberTitle', 'off');

% 使用plotpv绘制分类向量
subplot(1,2,1);
plotpv(P_and, T_and);
title('AND问题 - 数据分布');
xlabel('x_1');
ylabel('x_2');
grid on;

% 使用plotpc绘制分类区域
subplot(1,2,2);
plotpv(P_and, T_and);
hold on;
plotpc(net.IW{1,1}, net.b{1});
title('AND问题 - 决策边界');
xlabel('x_1');
ylabel('x_2');
legend('类别 0', '类别 1', '决策边界');
grid on;