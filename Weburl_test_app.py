# autor：费南多
# email：fpf0103@163.com
# 基于VGG模型和残差模块相结合的深度学习模型，
# 其中将所有的卷积层都替换为了全连接层
# 用于识别是否是恶意WebUrl请求
# 使用Flask框架将模型部署为Web服务，以便从客户端发送请求进行预测。以下是一个简单的示例：
# 在命令行中运行Flask应用：python Weburl_test_app.py
# 就可以在浏览器或其他HTTP客户端（如Postman）中发送POST请求到http://localhost:5000/predict，并获取模型的预测结果。
# 例子1：http://127.0.0.1:5000/predict?queries=www.foo.com&id=1,abc.com/admin.php 确保在 URL 中正确地传递 queries 参数，使用逗号分隔多个查询。
# 例子2：http://127.0.0.1:5000/predict?queries=www.foo.com/id=1<script>alert(1)</script,abc.com/admin.php 确保在 URL 中正确地传递 queries 参数，使用逗号分隔多个查询。


from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import urllib

app = Flask(__name__)

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

input_size = 500
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
vectorizer.fit(queries)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json(force=True)
        queries = data['queries']
    else:
        # 获取 GET 请求参数
        queries = request.args.get('queries', '')
        if queries:
            queries = queries.split(',')

    if not queries:
        return jsonify({'error': 'No queries provided'}), 400

    new_X = vectorizer.transform(queries).toarray()
    new_X_tensor = torch.tensor(new_X, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(new_X_tensor)
        predictions = (torch.sigmoid(outputs) > 0.5).squeeze().cpu().numpy()

    # Debugging: print predictions type and content
    print(f"Predictions type: {type(predictions)}")
    print(f"Predictions content: {predictions}")

    if isinstance(predictions, np.ndarray):
        predictions = predictions.tolist()

    results = [{'url': q, 'res': '正常请求' if pred == 0 else '恶意请求'} for q, pred in zip(queries, predictions)]
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
