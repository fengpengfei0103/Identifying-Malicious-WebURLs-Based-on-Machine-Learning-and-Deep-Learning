# autor：费南多
# email：fpf0103@163.com
# 基于VGG模型的深度学习模型，
# 其中将所有的卷积层都替换为了全连接层
# 用于识别是否是恶意WebUrl请求
#
import os
import urllib
import time
import html
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import torch
from torch import nn
from torchviz import make_dot


# 获取文本中的请求列表
def get_query_list(filename):
    directory = str(os.getcwd())
    print(directory)
    filepath = os.path.join(directory, filename)
    data = open(filepath, 'r', encoding='UTF-8').readlines()
    query_list = []
    for d in data:
        # 解码
        d = str(urllib.parse.unquote(d))  # converting url encoded data to simple string
        # print(d)
        query_list.append(d)
    return list(set(query_list))


# tokenizer function, this will make 3 grams of each query
# www.foo.com/1 转换为 ['www','ww.','w.f','.fo','foo','oo.','o.c','.co','com','om/','m/1']
def get_ngrams(query):
    tempQuery = str(query)
    ngrams = []
    for i in range(0, len(tempQuery) - 3):
        ngrams.append(tempQuery[i:i + 3])
    return ngrams


# 定义深度学习模型
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.conv1 = nn.Linear(input_size, 64)
        self.fc = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.fc(x)
        return x

class ResidualBlockF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlockF, self).__init__()
        self.conv1 = nn.Linear(in_channels, out_channels)
        self.conv2 = nn.Linear(out_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn(self.conv1(x)))
        out = self.bn(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out
class VGGFWithResidualF(nn.Module):
    def __init__(self, input_size, output_size):
        super(VGGFWithResidualF, self).__init__()
        self.conv1 = nn.Linear(input_size, 64)
        self.residual1 = ResidualBlockF(64, 64)
        self.conv2 = nn.Linear(64, 128)
        self.residual2 = ResidualBlockF(128, 128)
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, output_size)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.residual1(out)
        # out = self.pool(out)
        out = F.relu(self.conv2(out))
        out = self.residual2(out)
        # out = self.pool(out)
        # out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn(self.conv1(x)))
        out = self.bn(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class VGGF(nn.Module):
    def __init__(self, input_size, output_size):
        super(VGGF, self).__init__()
        self.conv1 = nn.Linear(input_size, 256)
        self.bn1= nn.BatchNorm1d(256)
        self.conv2 = nn.Linear(256, 512)
        # self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.bn1(out)
        out = F.relu(self.conv2(out))
        # out = self.bn2(out)
        out = F.relu(self.conv3(out))
        out = self.bn3(out)
        # out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.bn3(out)
        out = F.relu(self.fc1(out))
        out = self.bn3(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# 主函数
if __name__ == '__main__':

    # 获取正常请求
    good_query_listRES = get_query_list('goodqueries.txt')
    good_query_list = good_query_listRES[:40000]

    print(u"正常请求: ", len(good_query_list))
    for i in range(0, 5):
        print(good_query_list[i].strip('\n'))
    print("\n")

    # 获取恶意请求
    bad_query_listRES = get_query_list('badqueries.txt')
    bad_query_list = bad_query_listRES[:40000]
    print(u"恶意请求: ", len(bad_query_list))
    for i in range(0, 5):
        print(bad_query_list[i].strip('\n'))
    print("\n")

    # 预处理 good_y标记为0 bad_y标记为1
    good_y = [0 for i in range(0, len(good_query_list))]
    print(good_y[:5])
    bad_y = [1 for i in range(0, len(bad_query_list))]
    print(bad_y[:5])

    queries = bad_query_list + good_query_list
    y = bad_y + good_y

    # 定义矢量化 converting data to vectors
    vectorizer = TfidfVectorizer(tokenizer=get_ngrams, max_features=500)

    # 定义批次大小和分批处理数据
    batch_size = 100
    num_batches = len(queries) // batch_size + 1
    X_batches = []
    for i in range(num_batches - 1):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(queries))
        X_batch = vectorizer.fit_transform(queries[start_idx:end_idx])
        if X_batch.nnz > 0:  # 检查稀疏矩阵是否为空
            X_batches.append(X_batch)

    # 将稀疏矩阵堆叠成一个大的稀疏矩阵
    X = vstack(X_batches)
    # 将稀疏矩阵转换为稠密数组
    X_dense = X.toarray()

    # 转换为 PyTorch 张量
    X_tensor = torch.tensor(X_dense, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    # 定义模型
    input_size = X_train.shape[1]
    output_size = 1  # 二分类任务
    # model = Net(input_size, output_size)
    print(input_size)
    model = VGGF(input_size, output_size)
    # model = VGGFWithResidualF(input_size, output_size)

    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 定义数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 训练模型
    num_epochs = 20
    # 定义存储损失值的列表
    losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            # 添加高度和宽度维度，大小可以根据实际情况设置
            # inputs = inputs.unsqueeze(-1).unsqueeze(-1)
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        # 将损失值添加到列表中
        losses.append(epoch_loss)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))

    # 绘制收敛图
    plt.plot(range(1, num_epochs + 1), losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Convergence')
    plt.legend()

    # 保存图表到当前目录下
    plt.savefig('VGG_convergence_plot.png')  # 你可以指定其他文件格式如 '.pdf', '.svg', '.eps' 等

    # 显示图表
    plt.show()

    # 评估模型
    model.eval()

    # 创建虚拟输入张量（dummy input）
    dummy_input = torch.randn(batch_size, input_size)
    # 导出模型为ONNX格式
    torch.onnx.export(model, dummy_input, 'VGGnet_model.onnx', input_names=['input'], output_names=['output'])

    with torch.no_grad():
        # X_test = X_test.unsqueeze(-1).unsqueeze(-1)
        outputs = model(X_test)
        predicted = (torch.sigmoid(outputs) > 0.5).squeeze().cpu().numpy()
        accuracy = np.mean(predicted == y_test.numpy())
        print('模型的准确度: {:.4f}'.format(accuracy))
    # 将 PyTorch Tensor 转换为 NumPy 数组
    predicted_np = predicted
    y_test_np = y_test.numpy()

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_test_np, predicted_np)

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test_np),
                yticklabels=np.unique(y_test_np))
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')

    # 保存混淆矩阵到本地
    plt.savefig('VGG_confusion_matrix.png')

    plt.show()

    # 对新的请求列表进行预测
    new_queries = ['www.foo.com/id=1<script>alert(1)</script>',
                   'www.foo.com/name=admin\' or 1=1', 'abc.com/admin.php',
                   '"><svg onload=confirm(1)>',
                   'test/q=<a href="javascript:confirm(1)>',
                   'q=../etc/passwd',
                   '/stylesheet.php?version=1331749579',
                   '/<script>cross_site_scripting.nasl</script>.idc',
                   '<img \x39src=x onerror="javascript:alert(1)">',
                   '/jhot.php?rev=2 |less /etc/passwd']

    # 处理新请求列表
    new_X = vectorizer.transform(new_queries).toarray()
    new_X_tensor = torch.tensor(new_X, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(new_X_tensor)
        predictions = (torch.sigmoid(outputs) > 0.5).squeeze().cpu().numpy()

    # 输出预测结果
    res_list = []
    for q, pred in zip(new_queries, predictions):
        label = '正常请求' if pred == 0 else '恶意请求'
        res_list.append({'url': q, 'res': label})

    for n in res_list:
        print(n)