import numpy as np
import mytools as mt
import time
import sys
import threading
import random

# Activepart作为服务器，等待passivepart连接
server_for_passive = mt.MultiPartyServer('127.0.0.1', 5000, ['passive_to_active'])
passive_clients = server_for_passive.accept_all()
passive_handler = passive_clients['passive_to_active']

# Activepart作为客户端，连接s1和s2
client_to_s1 = mt.ClientConnector('127.0.0.1', 5001, 'active').get_handler()
client_to_s2 = mt.ClientConnector('127.0.0.1', 5002, 'active').get_handler()

# 业务逻辑函数保持不变
def secrete_share(y_trian,y_predict,bucket_a):
    g,h=mt.GradientHessianCalculator(y_trian,y_predict)
    g_s1=np.zeros(bucket_a.shape)
    h_s1=np.zeros(bucket_a.shape)
    g_s2=np.zeros(bucket_a.shape)
    h_s2=np.zeros(bucket_a.shape)
    for i in range(bucket_a.shape[0]):
        for j in range(bucket_a.shape[1]):
            for k in range(bucket_a.shape[2]):
                r_1=random.uniform(0,1)
                g_s1[i,j,k]=g[j]+r_1
                h_s1[i,j,k]=h[j]+r_1
                if bucket_a[i,j,k]==1:
                    g_s2[i,j,k]=-2*g[j]-r_1
                    h_s2[i,j,k]=-2*h[j]-r_1
                else:
                    g_s2[i,j,k]=-r_1
                    h_s2[i,j,k]=-r_1
    return g,h,g_s1,h_s1,g_s2,h_s2

# 多线程通信函数保持不变
def handle_server_comm_robust(client_handler, delta, gh_tuple, results_list, index, max_retries=5, retry_delay=2):
    for attempt in range(max_retries):
        try:
            client_handler.send_message(delta)
            client_handler.send_message(gh_tuple)
            G, H = client_handler.receive()
            if G is not None and H is not None:
                results_list[index] = (G, H)
                print(f"与 server_{index+1} 通信成功")
                return
        except Exception as e:
            print(f"与 server_{index+1} 通信失败 (尝试 {attempt+1}/{max_retries}): {e}")
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    print(f"错误：与 server_{index+1} 的所有通信尝试均失败！")
    results_list[index] = (None, None)

# 初始数据接收
y_train = passive_handler.receive()
print("active_part收到y_train")
bucket_a = passive_handler.receive()
print("active_part收到bucket_a")

# 主训练循环和业务逻辑保持不变
y_predict = np.ones(y_train.shape)*0.5

tree_num = 10
tree_depth = 3
lambda_value = 0.1

for tree_id in range(tree_num):
    delta_list = np.ones((bucket_a.shape[1], 2**tree_depth))
    g, h, g_s1, h_s1, g_s2, h_s2 = secrete_share(y_train, y_predict, bucket_a)
    print("正在训练第"+str(tree_id+1)+"棵树...")
    for node_id in range(1,2**tree_depth):
        print("正在训练第"+str(tree_id+1)+"棵树的第"+str(node_id)+"个节点...")
        if node_id <= 2**(tree_depth - 1) - 1:
            results = [None, None]
            thread1 = threading.Thread(target=handle_server_comm_robust, args=(client_to_s1, delta_list[:, node_id], (g_s1, h_s1), results, 0))
            thread2 = threading.Thread(target=handle_server_comm_robust, args=(client_to_s2, delta_list[:, node_id], (g_s2, h_s2), results, 1))
            thread1.start()
            thread2.start()
            thread1.join()
            thread2.join()

            if results[0] is None or results[1] is None:
                print("致命错误：一个或多个服务器通信失败，程序终止。")
                sys.exit(1)
            
            G1, H1 = results[0]
            G2, H2 = results[1]
            
            print("已成功接收所有G,H，开始计算最优分裂点...")
            idx = mt.fit(g, h, bucket_a, G1, H1, G2, H2, delta_list[:, node_id], lambda_value)
            
            passive_handler.send_message(idx)
            print("已发送查询最优分裂点请求给被动参与方")
            
            best_split, split_value = passive_handler.receive()

            delta_list[:, 2 * node_id]= (delta_list[:, node_id].reshape((bucket_a.shape[1],1)) * best_split).reshape((bucket_a.shape[1],))
            delta_list[:, 2 * node_id+1 ] = (delta_list[:, node_id].reshape((bucket_a.shape[1],1)) * (1 - best_split)).reshape((bucket_a.shape[1],))
        else:
            weight = -np.dot(g.T, delta_list[:, node_id].reshape((bucket_a.shape[1],1))) / (np.dot(h.T, delta_list[:, node_id].reshape((bucket_a.shape[1],1))) + lambda_value)
            print(weight)
            y_predict += weight * delta_list[:, node_id].reshape((bucket_a.shape[1],1))

passive_handler.close()
client_to_s1.close()
client_to_s2.close()
server_for_passive.close()