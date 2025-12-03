%% 实验3：XOR问题 - 验证感知器的局限性
clear; clc; close all;

fprintf('实验3: XOR问题（线性不可分）\n');

% XOR问题数据
P_xor = [0 0 1 1;
         0 1 0 1];

T_xor = [0 1 1 0];

% 显示XOR真值表
fprintf('\nXOR逻辑真值表:\n');
fprintf('x1  x2  | Output\n');
fprintf('----------------\n');
for i = 1:4
    fprintf('%d   %d   |   %d\n', P_xor(1,i), P_xor(2,i), T_xor(i));
end

% 尝试用感知器解决XOR问题
net_xor = perceptron;
net_xor.trainParam.epochs = 100;     % 增加训练轮数
net_xor.trainParam.show = 10;
net_xor.trainParam.goal = 0;

fprintf('\n尝试训练单层感知器解决XOR问题...\n');
fprintf('(注意：这将无法收敛)\n');

[net_xor, tr_xor] = train(net_xor, P_xor, T_xor);

% 测试
Y_xor = net_xor(P_xor);

fprintf('\nXOR问题测试结果:\n');
fprintf('x1  x2  | 目标  预测  是否正确\n');
fprintf('--------------------------------\n');
correct = 0;
for i = 1:4
    pred = round(Y_xor(i));
    is_correct = (pred == T_xor(i));
    if is_correct
        correct = correct + 1;
    end
    fprintf('%d   %d   |  %d     %d      %s\n', ...
        P_xor(1,i), P_xor(2,i), T_xor(i), pred, ...
        char(string(is_correct)));
end

accuracy = correct / 4 * 100;
fprintf('\n准确率: %.2f%%\n', accuracy);
fprintf('结论: 单层感知器无法完美解决XOR问题！\n');

% 可视化XOR问题
figure('Name', 'XOR问题 - 线性不可分', 'NumberTitle', 'off');

% 绘制数据点
subplot(1,2,1);
hold on;
plot(P_xor(1, T_xor==0), P_xor(2, T_xor==0), 'bo', ...
     'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', '类别 0');
plot(P_xor(1, T_xor==1), P_xor(2, T_xor==1), 'rx', ...
     'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', '类别 1');
title('XOR问题数据分布');
xlabel('x_1');
ylabel('x_2');
legend('Location', 'best');
grid on;
axis([-0.5 1.5 -0.5 1.5]);

% 尝试绘制决策边界（虽然不完美）
subplot(1,2,2);
hold on;
plot(P_xor(1, T_xor==0), P_xor(2, T_xor==0), 'bo', ...
     'MarkerSize', 12, 'LineWidth', 2);
plot(P_xor(1, T_xor==1), P_xor(2, T_xor==1), 'rx', ...
     'MarkerSize', 12, 'LineWidth', 2);

% 绘制感知器的决策边界
if ~isempty(net_xor.IW{1,1})
    plotpc(net_xor.IW{1,1}, net_xor.b{1});
end

title('感知器的决策边界（无法正确分类）');
xlabel('x_1');
ylabel('x_2');
legend('类别 0', '类别 1', '决策边界');
grid on;
axis([-0.5 1.5 -0.5 1.5]);

fprintf('\n从图中可以看出，无法用一条直线将两类分开！\n');