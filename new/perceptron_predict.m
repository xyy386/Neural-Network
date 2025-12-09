function y_pred = perceptron_predict(X, w, b)
% 感知器预测
% X: d×N 的输入矩阵（每列一个样本）
% w: d×1 的权重
% b: 标量偏置
% 返回:
%   y_pred: 1×N 的预测标签 (+1 或 -1)

    scores = w' * X + b;
    y_pred = sign(scores);
    y_pred(y_pred == 0) = 1;
end
