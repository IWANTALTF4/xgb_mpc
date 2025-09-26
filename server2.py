import mytools as mt
import numpy as np

# 使用新的服务器工具，明确期望接收 'active' 和 'passive' 两个客户端
server = mt.MultiPartyServer('127.0.0.1', 5002, ['active', 'passive'])
clients = server.accept_all()

# 从字典中获取特定客户端的处理器
active_handler = clients['active']
passive_handler = clients['passive']

# 从 passivepart 接收 bucket_s
bucket_s = passive_handler.receive()
print("server2收到bucket_s")

# 主循环逻辑保持不变，只是通信对象变了
while True:
    print("训练阶段开始")
    delta = active_handler.receive()
    if delta is None: break
    print("server2收到delta")
    
    g, h = active_handler.receive()
    
    if g is None: break
    print("server2收到g,h")
    
    print("开始计算G,H")
    G, H = mt.calculate(g, h, bucket_s, delta)
    
    active_handler.send_message((G, H))

server.close()
print("Server2关闭。")