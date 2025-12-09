function Y = adaline_predict(X, W, b)
% 自适应线性网络预测
% X: d×N 输入矩阵
% W: m×d 权重
% b: m×1 偏置
% 返回:
%   Y: m×N 输出

    Y = W * X + b;
end
