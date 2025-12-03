%% 实验3：ADALINE解决线性回归问题
fprintf('实验3：ADALINE解决线性回归问题\n');

% 生成线性回归数据：y = 2x + 3 + 噪声
x_train = -10:0.5:10;
y_train_true = 2*x_train + 3;
y_train = y_train_true + 0.5*randn(size(x_train));  % 添加噪声

% 准备训练数据（MATLAB格式）
P_reg = x_train;
T_reg = y_train;

fprintf('\n问题描述: 拟合线性函数 y = 2x + 3\n');
fprintf('训练样本数: %d\n', length(x_train));

% 创建ADALINE网络
net_reg = newlin(minmax(P_reg), 1, 0, 0.01);

% 设置训练参数
net_reg.trainParam.epochs = 200;
net_reg.trainParam.show = 50;
net_reg.trainParam.goal = 1e-3;

% 训练网络
fprintf('\n训练ADALINE进行线性回归...\n');
[net_reg, tr_reg] = train(net_reg, P_reg, T_reg);

% 预测
Y_reg = net_reg(P_reg);

% 提取学习到的参数
w_learned = net_reg.IW{1,1};
b_learned = net_reg.b{1};

fprintf('\n回归结果:\n');
fprintf('真实参数: y = 2.0000x + 3.0000\n');
fprintf('学习参数: y = %.4fx + %.4f\n', w_learned, b_learned);

% 计算性能指标
mse_reg = perform(net_reg, T_reg, Y_reg);
r2 = 1 - sum((T_reg - Y_reg).^2) / sum((T_reg - mean(T_reg)).^2);

fprintf('均方误差 MSE: %.6f\n', mse_reg);
fprintf('决定系数 R²: %.6f\n', r2);