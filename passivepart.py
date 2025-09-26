import mytools as mt
import numpy as np 

# 使用新的客户端工具，连接时自动表明身份
# 注意：与Activepart的连接方式保持不变，因为它是一对一的
client_to_active = mt.ClientConnector('127.0.0.1', 5000, 'passive_to_active').get_handler()
client_to_s1 = mt.ClientConnector('127.0.0.1', 5001, 'passive').get_handler()
client_to_s2 = mt.ClientConnector('127.0.0.1', 5002, 'passive').get_handler()

# 业务逻辑和数据发送保持不变
matrix = mt.read_file('bank-full.csv')
X_train, X_test, y_train, y_test = mt.TrainTestSplit(matrix)
bucket, bucket_a, bucket_s = mt.Bucketing(X_train, bucket_num=10)

client_to_active.send_message(y_train)
client_to_active.send_message(bucket_a)

client_to_s1.send_message(bucket_s)
print("已发送bucket_s至服务器s1")
client_to_s2.send_message(bucket_s)
print("已发送bucket_s至服务器s2")

# 主循环逻辑保持不变
while True:
    idx = client_to_active.receive()
    print(idx)
    if idx is None: break
    print("收到查询最有分裂点信息请求")
    delta = np.zeros((bucket.shape[1], 1))
    for i in range(0,idx[1]):
        delta += bucket[idx[0], :, i].reshape((bucket.shape[1], 1))
    value = X_train[idx[0], idx[1]]
    client_to_active.send_message((delta, value))
    print("已发送delta和value至主动参与方")

client_to_active.close()
client_to_s1.close()
client_to_s2.close()