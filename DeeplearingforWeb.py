# coding: utf-8
import os
import urllib
import time
import html
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


# 获取文本中的请求列表
def get_query_list(filename):
    directory = str(os.getcwd())
    filepath = directory + "/" + filename
    data = open(filepath, 'r', encoding='UTF-8').readlines()
    query_list = []
    for d in data:
        # 解码
        d = str(urllib.parse.unquote(d))  # converting url encoded data to simple string
        query_list.append(d)
    return list(set(query_list))


# tokenizer function, this will make 3 grams of each query
def get_ngrams(query):
    tempQuery = str(query)
    ngrams = []
    for i in range(0, len(tempQuery) - 3):
        ngrams.append(tempQuery[i:i + 3])
    return ngrams


# 定义深度学习模型
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * 48, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':

    # 获取正常请求
    good_query_list = get_query_list('goodqueries.txt')
    print(u"正常请求: ", len(good_query_list))

    # 获取恶意请求
    bad_query_list = get_query_list('badqueries.txt')
    print(u"恶意请求: ", len(bad_query_list))

    # 预处理 good_y标记为0 bad_y标记为1
    good_y = [0 for i in range(0, len(good_query_list))]
    bad_y = [1 for i in range(0, len(bad_query_list))]

    queries = bad_query_list + good_query_list
    y = bad_y + good_y

    # 定义矢量化 converting data to vectors
    vectorizer = TfidfVectorizer(tokenizer=get_ngrams)

    # 把不规律的文本字符串列表转换成规律的 ( [i,j], tdidf值) 的矩阵X
    X_sparse = vectorizer.fit_transform(queries)

    # 转换为 PyTorch 的 SparseTensor
    X = torch.sparse_coo_tensor(X_sparse.nonzero(), X_sparse.data, X_sparse.shape)

    # 转换为密集张量
    X = X.to_dense()

    # 使用 train_test_split 分割 X y 列表
    X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y, test_size=0.2, random_state=42)

    # 数据预处理
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # 转换为 PyTorch 的 Tensor
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 使用 DataLoader 加载数据
    train_dataset = TensorDataset(torch.tensor(X_train), y_train)
    test_dataset = TensorDataset(torch.tensor(X_test), y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = ConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader)}")

    # 评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.unsqueeze(1))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"模型准确度: {100 * correct / total}%")

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

    # 数据预处理
    X_new_sparse = vectorizer.transform(new_queries)
    X_new = torch.tensor(X_new_sparse.toarray())

    # 进行预测
    with torch.no_grad():
        outputs = model(X_new.unsqueeze(1))
        _, predicted = torch.max(outputs, 1)

    # 输出预测结果
    res_list = []
    for q, r in zip(new_queries, predicted):
        tmp = '正常请求' if r == 0 else '恶意请求'
        q_entity = html.escape(q)
        res_list.append({'url': q_entity, 'res': tmp})

    for n in res_list:
        print(n)

