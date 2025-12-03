%% 实验2：ADALINE与感知器的对比
fprintf('实验2：ADALINE与感知器的对比\n');

% 创建感知器进行对比
net_perceptron = perceptron;
net_perceptron.trainParam.epochs = 100;
net_perceptron.trainParam.show = NaN;

fprintf('\n训练感知器...\n');
[net_perceptron, tr_perceptron] = train(net_perceptron, P_and, T_and);
Y_perceptron = net_perceptron(P_and);

fprintf('\n对比结果:\n');
fprintf('方法        | 权重w1    权重w2    偏置b     训练轮数\n');
fprintf('----------------------------------------------------------\n');
fprintf('感知器      | %.4f   %.4f   %.4f   %d\n', ...
    net_perceptron.IW{1,1}(1), net_perceptron.IW{1,1}(2), ...
    net_perceptron.b{1}, tr_perceptron.num_epochs);
fprintf('ADALINE     | %.4f   %.4f   %.4f   %d\n', ...
    net_adaline.IW{1,1}(1), net_adaline.IW{1,1}(2), ...
    net_adaline.b{1}, tr_adaline.num_epochs);

fprintf('\n关键区别:\n');
fprintf('1. 感知器使用硬限幅激活函数，ADALINE使用线性激活函数\n');
fprintf('2. 感知器基于分类误差更新，ADALINE基于输出误差更新\n');
fprintf('3. ADALINE使用梯度下降法，收敛更稳定\n');