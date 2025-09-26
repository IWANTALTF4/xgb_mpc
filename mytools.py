import numpy as np 
import pandas as pd 
import math
import socket
import pickle
import sys
import time
from collections import Counter
from sklearn.model_selection import train_test_split
import random

# ===================================================================
# 以下所有计算函数，我将保持原样，不做任何修改
# ===================================================================
def read_file(file_name,delimiter=';'):
    df=pd.read_csv(file_name,delimiter=delimiter)
    matrix=df.to_numpy()
    for column in range(matrix.shape[1]):
        column_values=matrix[:,column]
        has_string=any(isinstance(i,str) for i in column_values)
        if has_string:
            counter=Counter(column_values)
            sorted_counter=sorted(counter.items(),key=lambda x:x[1],reverse=True)
            encoding_map = {}
            for rank, (value, count) in enumerate(sorted_counter, 1):
                encoding_map[value] = rank
            for row in range(matrix.shape[0]):
                original_value=matrix[row,column]
                matrix[row,column]=encoding_map.get(original_value,-1)
    return matrix

def TrainTestSplit(matrix,train_ratio=0.8):
    X = matrix[:, :-1]
    y = matrix[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1-train_ratio, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

def Bucketing(X_train,bucket_num=10):
    for column in range(X_train.shape[1]):
        id_sorted=np.argsort(X_train[:,column])
        X_train[:,column]=id_sorted
    bucket_size=math.ceil(X_train.shape[0]/bucket_num)
    bucket_list=np.zeros((X_train.shape[1],X_train.shape[0],bucket_num),dtype=bool)
    for column in range(X_train.shape[1]):
        row_count=1
        col_count=0
        for row in range(X_train.shape[0]):
            if row_count<bucket_size:
                bucket_list[column,X_train[row,column],col_count]=True
                row_count+=1
            elif row_count==bucket_size:
                bucket_list[column,X_train[row,column],col_count]=True
                row_count=1
                col_count+=1
    bucket_a=np.random.choice([True,False],size=bucket_list.shape)
    bucket_s=bucket_a^bucket_list
    return bucket_list,bucket_a,bucket_s

def GradientHessianCalculator(y_train,y_predict,type='CrossEntropy'):
    if type=='CrossEntropy':
        gradient=y_predict-y_train
        hessian=np.ones(y_train.shape)*(y_predict*(1-y_predict))
    elif type=='MeanSquaredError':
        gradient=2*(y_predict-y_train)
        hessian=np.ones(y_train.shape)*(2)
    else:
        print("暂不支持该损失函数")
        print(gradient.shape,hessian.shape)
    return gradient,hessian

def fit(g,h,b,G1,H1,G2,H2,delta,lambda_value):
    G=np.zeros((b.shape[0],b.shape[2]))
    H=np.zeros((b.shape[0],b.shape[2]))
    G_All=np.dot(g.T,delta)
    H_All=np.dot(h.T,delta)
    for i in range(b.shape[0]):
        for j in range(b.shape[2]):
            G[i,j]=np.dot(g.T,b[i,:,j]*delta)
            H[i,j]=np.dot(h.T,b[i,:,j]*delta)
    G+=G1+G2
    H+=H1+H2
    for j in range(1,G.shape[1]):
        G[:,j]+=G[:,j-1]
        H[:,j]+=H[:,j-1]
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            if i==0:
                count=G[i,j]**2/(H[i,j]+lambda_value)+(G_All-G[i,j])**2/(H_All-H[i,j]+lambda_value)-G_All**2/(H_All+lambda_value)
                split_idx=(i,j)
            else:
                Gain=G[i,j]**2/(H[i,j]+lambda_value)+(G_All-G[i,j])**2/(H_All-H[i,j]+lambda_value)-G_All**2/(H_All+lambda_value)
                if Gain>count:
                    count=Gain
                    split_idx=(i,j)
                else:
                    continue   
    return split_idx

def calculate(g,h,b,delta):
    G=np.zeros((b.shape[0],b.shape[2]))
    H=np.zeros((b.shape[0],b.shape[2]))
    for i in range(b.shape[0]):
        for j in range(b.shape[2]):
            G[i,j]=np.dot(g[i,:,j].T,b[i,:,j]*delta)
            H[i,j]=np.dot(h[i,:,j].T,b[i,:,j]*delta)
    return G,H

# ===================================================================
# 以下是通信部分，我将进行修改
# ===================================================================
class MessageHandler:
    """一个辅助类，用于在已建立的连接上收发消息"""
    def __init__(self, conn):
        self.conn = conn

    def _recvall(self, n):
        buf = b''
        while len(buf) < n:
            chunk = self.conn.recv(n - len(buf))
            if not chunk: raise ConnectionError("连接被意外关闭")
            buf += chunk
        return buf

    def send_message(self, message):
        if self.conn is None: raise ConnectionError("连接未建立")
        serialized_message = pickle.dumps(message)
        self.conn.sendall(len(serialized_message).to_bytes(8, 'big'))
        self.conn.sendall(serialized_message)

    def receive(self):
        if self.conn is None: raise ConnectionError("连接未建立")
        try:
            length_bytes = self._recvall(8)
            length = int.from_bytes(length_bytes, 'big')
            data = self._recvall(length)
            return pickle.loads(data)
        except (ConnectionError, EOFError):
            return None

    def close(self):
        if self.conn: self.conn.close()

class MultiPartyServer:
    """一个专门的服务器类，可以接受多个指定身份的客户端"""
    def __init__(self, host, port, expected_identities):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(len(expected_identities))
        self.clients = {}
        self.expected = set(expected_identities)
        print(f"服务器在 {host}:{port} 启动，等待 {self.expected} 连接...")

    def accept_all(self):
        while self.expected - set(self.clients.keys()):
            conn, addr = self.sock.accept()
            try:
                identity = conn.recv(1024).decode().strip()
                if identity in self.expected:
                    self.clients[identity] = MessageHandler(conn)
                    print(f"已接受来自 {addr} 的客户端，身份: {identity}")
                else:
                    print(f"收到未知身份 '{identity}'，关闭连接。")
                    conn.close()
            except Exception as e:
                print(f"身份识别失败: {e}")
                conn.close()
        print("所有预期的客户端均已连接。")
        return self.clients

    def close(self):
        for handler in self.clients.values():
            handler.close()
        self.sock.close()

class ClientConnector:
    """一个专门的客户端类，连接后会表明身份"""
    def __init__(self, host, port, identity):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                sock.connect((host, port))
                sock.sendall(identity.encode())
                self.handler = MessageHandler(sock)
                print(f"作为 '{identity}' 成功连接到 {host}:{port}")
                break
            except ConnectionRefusedError:
                print(f"连接到 {host}:{port} 失败，1秒后重试...")
                time.sleep(1)
    
    def get_handler(self):
        return self.handler