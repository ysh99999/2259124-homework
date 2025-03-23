import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 定义一个简单的神经网络类
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # 设置随机种子以确保结果可复现
        np.random.seed(2151300)
        # 初始化第一层的权重和偏置
        self.W1 = np.random.randn(input_size, hidden_size)  # 输入层到隐藏层的权重矩阵
        self.b1 = np.zeros(hidden_size)  # 隐藏层的偏置向量
        # 初始化第二层的权重和偏置
        self.W2 = np.random.randn(hidden_size, output_size)  # 隐藏层到输出层的权重矩阵
        self.b2 = np.zeros(output_size)  # 输出层的偏置向量

    # 定义ReLU激活函数
    def relu(self, Z):
        return np.maximum(0, Z)  # ReLU函数：f(x) = max(0, x)

    # 定义ReLU激活函数的导数
    def relu_derivative(self, Z):
        return Z > 0  # ReLU的导数：f'(x) = 1 if x > 0 else 0

    # 前向传播函数
    def forward(self, X):
        # 计算隐藏层的输入
        self.Z1 = X.dot(self.W1) + self.b1  # Z1 = X * W1 + b1
        # 应用ReLU激活函数
        self.A1 = self.relu(self.Z1)  # A1 = ReLU(Z1)
        # 计算输出层的输入
        self.Z2 = self.A1.dot(self.W2) + self.b2  # Z2 = A1 * W2 + b2
        # 返回输出层的值（未经过激活函数）
        return self.Z2

    # 计算损失函数（均方误差）
    def compute_loss(self, y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()  # MSE = mean((y_true - y_pred)^2)

    # 反向传播函数
    def backward(self, X, y, learning_rate):
        m = y.shape[0]  # 样本数量
        # 计算输出层的误差
        dZ2 = self.Z2 - y  # 输出层的误差
        # 计算第二层权重的梯度
        dW2 = self.A1.T.dot(dZ2) / m  # dW2 = (A1.T * dZ2) / m
        # 计算第二层偏置的梯度
        db2 = np.sum(dZ2, axis=0) / m  # db2 = sum(dZ2) / m
        # 计算隐藏层的误差
        dZ1 = dZ2.dot(self.W2.T) * self.relu_derivative(self.Z1)  # dZ1 = (dZ2 * W2.T) * ReLU'(Z1)
        # 计算第一层权重的梯度
        dW1 = X.T.dot(dZ1) / m  # dW1 = (X.T * dZ1) / m
        # 计算第一层偏置的梯度
        db1 = np.sum(dZ1, axis=0) / m  # db1 = sum(dZ1) / m

        # 更新权重和偏置
        self.W1 -= learning_rate * dW1  # W1 = W1 - learning_rate * dW1
        self.b1 -= learning_rate * db1  # b1 = b1 - learning_rate * db1
        self.W2 -= learning_rate * dW2  # W2 = W2 - learning_rate * dW2
        self.b2 -= learning_rate * db2  # b2 = b2 - learning_rate * db2

    # 训练函数
    def train(self, X_train, y_train, epochs, learning_rate=0.01, decay_rate=0.0):
        for epoch in range(epochs):
            # 前向传播
            y_pred = self.forward(X_train)
            # 计算损失
            loss = self.compute_loss(y_train, y_pred)
            # 每1000次迭代打印一次损失和学习率
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss}, Learning Rate: {learning_rate}")
            # 反向传播并更新参数
            self.backward(X_train, y_train, learning_rate)
            # 学习率衰减
            learning_rate *= (1. / (1. + decay_rate * epoch))


# 定义目标函数（用于生成数据）
def target_function(X):
    return X ** 3  # 目标函数：f(x) = x^3


# 生成训练数据
X_train = np.linspace(-np.pi, np.pi, 700).reshape(-1, 1)  # 生成700个训练样本
y_train = target_function(X_train)  # 计算训练样本的目标值
# 生成测试数据
X_test = np.linspace(-np.pi, np.pi, 300).reshape(-1, 1)  # 生成300个测试样本
y_test = target_function(X_test)  # 计算测试样本的目标值

# 初始化神经网络模型
model = SimpleNeuralNetwork(input_size=1, hidden_size=30, output_size=1)
# 训练模型
model.train(X_train, y_train, epochs=100000, learning_rate=0.001, decay_rate=1e-12)

# 使用训练好的模型进行预测
y_pred = model.forward(X_test)

# 计算评价指标
mse = mean_squared_error(y_test, y_pred)  # 均方误差
mae = mean_absolute_error(y_test, y_pred)  # 平均绝对误差
r2 = r2_score(y_test, y_pred)  # R²分数

# 打印评价指标
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")

# 可视化结果
plt.figure(figsize=(10, 6))
# 绘制真实函数曲线
plt.plot(X_test, y_test, label='True Function')
# 绘制模型预测曲线
plt.plot(X_test, y_pred, label='Model Prediction', linestyle='--')
# 添加图例
plt.legend()
# 设置标题和坐标轴标签
plt.title("Simple Neural Network Model vs True Function")
plt.xlabel("X")
plt.ylabel("Y")
# 显示图像
plt.show()