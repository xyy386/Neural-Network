%%实验1：自适应线性元件解决线性分类问题（AND逻辑）
fprintf('实验1：自适应线性元件解决线性分类问题（AND逻辑）\n');
                            
% AND问题数据
P_and = [0 0 1 1;
         0 1 0 1];
T_and = [0 0 0 1];

fprintf('\nAND逻辑真值表:\n');
fprintf('x1  x2  | Output\n');
fprintf('----------------\n');
for i = 1:4
    fprintf('%d   %d   |   %d\n', P_and(1,i), P_and(2,i), T_and(i));
end

% 创建ADALINE网络
% 使用newlin创建线性层网络
net_adaline = newlin(P_and, T_and, 0, 0.01);  % 输入范围自动设置，学习率0.01

% 设置训练参数
net_adaline.trainParam.epochs = 100;
net_adaline.trainParam.show = 20;
net_adaline.trainParam.goal = 1e-5;

% 训练网络
fprintf('\n使用ADALINE训练AND问题...\n');
[net_adaline, tr_adaline] = train(net_adaline, P_and, T_and);

% 测试网络
Y_adaline = net_adaline(P_and);

fprintf('\nADALINE测试结果:\n');
fprintf('x1  x2  | 目标  输出值   预测类别  是否正确\n');
fprintf('--------------------------------------------\n');
correct_adaline = 0;
for i = 1:4
    pred_class = round(Y_adaline(i));
    is_correct = (pred_class == T_and(i));
    if is_correct
        correct_adaline = correct_adaline + 1;
    end
    fprintf('%d   %d   |  %d    %.4f      %d         %s\n', ...
        P_and(1,i), P_and(2,i), T_and(i), Y_adaline(i), ...
        pred_class, char(string(is_correct)));
end

fprintf('\n准确率: %.2f%%\n', correct_adaline/4*100);
fprintf('权重 W: [%.4f, %.4f]\n', net_adaline.IW{1,1}(1), net_adaline.IW{1,1}(2));
fprintf('偏置 b: %.4f\n', net_adaline.b{1});