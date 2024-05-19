# autor：费南多
# email：fpf0103@163.com
# 基于VGG模型和残差模块相结合的深度学习模型，
# 其中将所有的卷积层都替换为了全连接层
# 用于识别是否是恶意WebUrl请求
# 加载并使用模型
# 在实际应用中，可以加载保存的模型并进行预测。以下是如何在Python脚本中加载和使用模型的示例：
import torch
import torch.nn as nn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import urllib

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
# 定义模型
class ResidualBlockF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlockF, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn(self.fc1(x)))
        out = self.bn(self.fc2(out))
        out += residual
        out = torch.relu(out)
        return out

class VGGFWithResidualF(nn.Module):
    def __init__(self, input_size, output_size):
        super(VGGFWithResidualF, self).__init__()
        self.cfc1 = nn.Linear(input_size, 64)
        self.bn0 = nn.BatchNorm1d(64)
        self.residual1 = ResidualBlockF(64, 64)
        self.cfc2 = nn.Linear(64, 128)
        self.residual2 = ResidualBlockF(128, 128)
        self.fc0 = nn.Linear(128, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        out = torch.relu(self.cfc1(x))
        out = self.bn0(out)
        out = self.residual1(out)
        out = torch.relu(self.cfc2(out))
        out = self.residual2(out)
        out = torch.relu(self.fc0(out))
        out = self.bn1(out)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# 加载模型
input_size = 500  # 假设你使用的是相同的输入大小
output_size = 1
model = VGGFWithResidualF(input_size, output_size)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 获取正常请求和恶意请求
good_query_listRES = get_query_list('goodqueries.txt')
good_query_list = good_query_listRES[:40000]

bad_query_listRES = get_query_list('badqueries.txt')
bad_query_list = bad_query_listRES[:40000]

# 标记请求
good_y = [0 for i in range(0, len(good_query_list))]
bad_y = [1 for i in range(0, len(bad_query_list))]

# 交替合并正常请求和恶意请求
merged_list = [query for pair in zip(bad_query_list, good_query_list) for query in pair]
queries = merged_list
y = [query for pair in zip(bad_y, good_y) for query in pair]

# 加载矢量化器
vectorizer = TfidfVectorizer(tokenizer=get_ngrams, max_features=500)
vectorizer.fit(queries)  # 假设你已经在训练数据上拟合了矢量化器

# 预测新数据
new_queries = ['www.foo.com/id=1<script>alert(1)</script>',
    '/jhot.php?rev=2 |less /etc/passwd'
    ]
new_X = vectorizer.transform(new_queries).toarray()
new_X_tensor = torch.tensor(new_X, dtype=torch.float32)
with torch.no_grad():
    outputs = model(new_X_tensor)
    predictions = (torch.sigmoid(outputs) > 0.5).squeeze().cpu().numpy()

# 输出预测结果
for q, pred in zip(new_queries, predictions):
    label = '正常请求' if pred == 0 else '恶意请求'
    print({'url': q, 'res': label})
