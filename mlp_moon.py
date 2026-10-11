# 导入需要的工具包
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import torch
from sklearn.model_selection import train_test_split


# 设置随机种子，保证每次运行结果完全一样，方便复现实验
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# 生成月牙二分类数据集
# 总共1000个样本，noise增加一点噪声，random_state固定月牙形状不变
X_data, y_label = make_moons(n_samples=1000, noise=0.15, random_state=random_seed)

# 打印原始数据维度
print("特征矩阵X的形状：", X_data.shape)
print("标签y的形状：", y_label.shape)

# 画原始月牙散点图
plt.figure(figsize=(7,6))
# 类别0
plt.scatter(X_data[y_label==0,0], X_data[y_label==0,1], c="blue", label="类别0")
# 类别1
plt.scatter(X_data[y_label==1,0], X_data[y_label==1,1], c="orange", label="类别1")
plt.xlabel("特征1")
plt.ylabel("特征2")
plt.title("make_moons月牙数据集原始可视化")
plt.legend()
# 保存图片到当前文件夹
plt.savefig("moon_raw_data.png")
plt.close()
# plt.show()
# ---------- 数据集划分 ----------
# 先分出训练集和临时集合
X_train, X_temp, y_train, y_temp = train_test_split(
    X_data, y_label,
    test_size=0.3,
    random_state=random_seed,
    stratify=y_label
)

# 把临时集拆成验证集、测试集
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=random_seed,
    stratify=y_temp
)

# 打印各组数据大小
print("\n数据集划分结果：")
print("训练集 X_train：", X_train.shape, "，标签 y_train：", y_train.shape)
print("验证集 X_valid：", X_valid.shape, "，标签 y_valid：", y_valid.shape)
print("测试集 X_test：", X_test.shape, "，标签 y_test：", y_test.shape)

import torch.nn as nn

# 搭建简单多层感知机 MLP
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 两层全连接网络
        self.layers = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

# 实例化模型
model = SimpleMLP()

# 转成tensor，pytorch只能用张量训练
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).unsqueeze(1)

# 损失函数和优化器
loss_func = nn.BCELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练循环
# 保存每一轮指标
train_loss_history = []
valid_loss_history = []
train_acc_history = []
valid_acc_history = []
epoch_num = 200
for epoch in range(epoch_num):
    # 训练模式
    model.train()
    pred = model(X_train_tensor)
    loss = loss_func(pred, y_train_tensor)

    # 反向传播更新参数
    opt.zero_grad()
    loss.backward()
    opt.step()
 # =========任务3：计算训练集准确率=========
    train_pred_label = (pred > 0.5).float()
    train_acc = (train_pred_label == y_train_tensor).float().mean()

    # 验证集评估
    model.eval()
    with torch.no_grad():
        val_pred = model(X_valid_tensor)
        val_loss = loss_func(val_pred, y_valid_tensor)
        val_pred_label = (val_pred > 0.5).float()
        valid_acc = (val_pred_label == y_valid_tensor).float().mean()

    # =========任务3：存入列表，每轮都保存=========
    train_loss_history.append(loss.item())
    valid_loss_history.append(val_loss.item())
    train_acc_history.append(train_acc.item())
    valid_acc_history.append(valid_acc.item())
    # 每20轮打印
    if epoch % 20 == 0:
        print(f'epoch:{epoch}, train_loss:{loss.item():.4f}, val_loss:{val_loss.item():.4f}, train_acc:{train_acc.item():.4f}, valid_acc:{valid_acc.item():.4f}')
print("=====准备开始画loss曲线=====")
plt.figure(figsize=(12, 5))
...

# =========任务3：绘制loss、accuracy曲线=========
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label="Train Loss")
plt.plot(valid_loss_history, label="Valid Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(train_acc_history, label="Train Acc")
plt.plot(valid_acc_history, label="Valid Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("loss_acc_curve.png")
plt.close()
print("loss&acc曲线图片已保存 loss_acc_curve.png")

print("训练结束")
# -------- 画决策边界 --------
import numpy as np
# 生成网格采样点
x_min, x_max = X_data[:,0].min()-0.5, X_data[:,0].max()+0.5
y_min, y_max = X_data[:,1].min()-0.5, X_data[:,1].max()+0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# 网格点转tensor
grid_data = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.tensor(grid_data, dtype=torch.float32)

# 预测
model.eval()
with torch.no_grad():
    z_pred = model(grid_tensor)
z = z_pred.reshape(xx.shape).numpy()

# 绘图
plt.figure(figsize=(7,6))
plt.contourf(xx, yy, z, alpha=0.4)
plt.scatter(X_data[y_label==0,0], X_data[y_label==0,1], c="blue")
plt.scatter(X_data[y_label==1,0], X_data[y_label==1,1], c="orange")
plt.title("MLP decision boundary")
plt.savefig("decision_boundary.png")
plt.close()
print("决策边界图已保存 decision_boundary.png")
# 保存训练好的模型权重
torch.save(model.state_dict(), "mlp_model.pth")
print("模型权重保存成功：mlp_model.pth")

# 加载模型（演示，测试保存加载功能）
new_model = SimpleMLP()
new_model.load_state_dict(torch.load("mlp_model.pth"))
new_model.eval()
print("模型加载完成")
# =================任务4 超参数对照实验=================
# 保持随机种子、数据集
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# 生成同样月牙数据
X_data, y_label = make_moons(n_samples=1000, noise=0.15, random_state=random_seed)
# 同样划分方式
X_train, X_temp, y_train, y_temp = train_test_split(
    X_data, y_label,
    test_size=0.3,
    random_state=random_seed,
    stratify=y_label
)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=random_seed,
    stratify=y_temp
)

# 转tensor，和前面代码保持一致
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).unsqueeze(1)

# 改造模型类：隐藏层神经元作为参数，兼容原有逻辑
class SimpleMLP_Compare(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)

# 封装训练函数
def run_experiment(hidden_size, exp_tag):
    model = SimpleMLP_Compare(hidden_size)
    loss_func = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    epoch_num = 200

    train_loss_hist = []
    valid_loss_hist = []
    train_acc_hist = []
    valid_acc_hist = []

    for epoch in range(epoch_num):
        model.train()
        pred = model(X_train_tensor)
        loss = loss_func(pred, y_train_tensor)

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_pred_label = (pred > 0.5).float()
        train_acc = (train_pred_label == y_train_tensor).float().mean()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_valid_tensor)
            val_loss = loss_func(val_pred, y_valid_tensor)
            val_pred_label = (val_pred > 0.5).float()
            valid_acc = (val_pred_label == y_valid_tensor).float().mean()

        train_loss_hist.append(loss.item())
        valid_loss_hist.append(val_loss.item())
        train_acc_hist.append(train_acc.item())
        valid_acc_hist.append(valid_acc.item())

        if epoch % 40 == 0:
            print(f"{exp_tag} | epoch:{epoch}, train_loss:{loss.item():.4f}, valid_acc:{valid_acc.item():.4f}")
    
    # 绘制本组实验loss&acc图
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_loss_hist, label="Train Loss")
    plt.plot(valid_loss_hist, label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{exp_tag} Loss Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1,2,2)
    plt.plot(train_acc_hist, label="Train Acc")
    plt.plot(valid_acc_hist, label="Valid Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{exp_tag} Accuracy Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{exp_tag}_result.png")
    plt.close()
    return valid_acc_hist[-1]

# 两组对照实验
print("=====实验1：隐藏层神经元 = 8 =====")
acc_h8 = run_experiment(hidden_size=8, exp_tag="hidden_8")

print("\n=====实验2：隐藏层神经元 =32 =====")
acc_h32 = run_experiment(hidden_size=32, exp_tag="hidden_32")

print("\n=====对照实验汇总结果=====")
print(f"隐藏层8神经元，最终验证集准确率：{acc_h8:.4f}")
print(f"隐藏层32神经元，最终验证集准确率：{acc_h32:.4f}")
# =================任务5 测试集评估+混淆矩阵=================
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 使用训练好的SimpleMLP模型，在测试集上评估
model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    test_pred = model(X_test_tensor)
    test_pred_label = (test_pred > 0.5).float()

# 计算测试集准确率
test_acc = (test_pred_label == y_test_tensor).float().mean()
print(f"\n=====任务5 测试集评估结果=====")
print(f"测试集准确率：{test_acc.item():.4f}")

# 计算混淆矩阵
y_true = y_test_tensor.squeeze().numpy()
y_pred = test_pred_label.squeeze().numpy()
cm = confusion_matrix(y_true, y_pred)

# 绘制混淆矩阵
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.title("Confusion Matrix 混淆矩阵")
plt.savefig("confusion_matrix.png")
plt.close()
print("混淆矩阵图已保存 confusion_matrix.png")

