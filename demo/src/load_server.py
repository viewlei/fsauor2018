# -*- coding: utf-8 -*-
import socket
import numpy as np
import loadModel
import threading

d = {-2: '<未提及>',
     -1: '<负面评价>',
     0: '<中性评价>',
     1: '<正面评价>'
     }


def handleResult(res):
    # res is a ndarray , which shape is (1,4)
    # res是全连接层的输出
    res -= np.max(res)
    sum = np.sum(np.exp(res))
    res = np.exp(res) / sum
    return np.argmax(res) - 2, np.max(res) * 100


def openSocket(port, label):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('localhost', port))
        sess, input_data, pre, name = loadModel.getPre(label)
        sock.listen(5)  # 最大连接数
        print(sock.getsockname())
        return sock, sess, input_data, pre, name
    except:
        print("init socket error!")


def thread_do(sock):
    # sock = (sock, sess, input_data, pre)
    sess = sock[1]
    input_data = sock[2]
    pre = sock[3]
    name = sock[4]
    sock = sock[0]
    sock_name = sock.getsockname()
    while True:
        print("listen for client...")
        conn, addr = sock.accept()
        conn.settimeout(30)
        szBuf = conn.recv(10240)
        string = str(szBuf, 'gbk')
        print(sock_name, "得到来自", addr, "的消息:", string[:10])

        if "0" == szBuf:
            conn.send(b"exit")
        else:
            res = sess.run([pre], feed_dict={input_data: loadModel.getMatrix(string)})
            res, confidence = handleResult(res)
            confidence = round(confidence, 2)
            print('对于', name, '评价结果：', d[int(res)], ', 置信度', confidence, '%')
            result = '对于<' + name + '>评价结果：' + d[int(res)] + ', 置信度' + str(confidence) + '%'
            result = str(result).encode('gbk')
            conn.send(result)

        conn.close()


sock_0 = openSocket(12000, 0)
t_0 = threading.Thread(target=thread_do, args=(sock_0,))
sock_1 = openSocket(12001, 1)
t_1 = threading.Thread(target=thread_do, args=(sock_1,))
sock_2 = openSocket(12002, 2)
t_2 = threading.Thread(target=thread_do, args=(sock_2,))
sock_3 = openSocket(12003, 3)
t_3 = threading.Thread(target=thread_do, args=(sock_3,))
sock_4 = openSocket(12004, 4)
t_4 = threading.Thread(target=thread_do, args=(sock_4,))
sock_5 = openSocket(12005, 5)
t_5 = threading.Thread(target=thread_do, args=(sock_5,))
sock_6 = openSocket(12006, 6)
t_6 = threading.Thread(target=thread_do, args=(sock_6,))
sock_7 = openSocket(12007, 7)
t_7 = threading.Thread(target=thread_do, args=(sock_7,))
sock_8 = openSocket(12008, 8)
t_8 = threading.Thread(target=thread_do, args=(sock_8,))
sock_9 = openSocket(12009, 9)
t_9 = threading.Thread(target=thread_do, args=(sock_9,))
sock_10 = openSocket(12010, 10)
t_10 = threading.Thread(target=thread_do, args=(sock_10,))
sock_11 = openSocket(12011, 11)
t_11 = threading.Thread(target=thread_do, args=(sock_11,))
sock_12 = openSocket(12012, 12)
t_12 = threading.Thread(target=thread_do, args=(sock_12,))
sock_13 = openSocket(12013, 13)
t_13 = threading.Thread(target=thread_do, args=(sock_13,))
sock_14 = openSocket(12014, 14)
t_14 = threading.Thread(target=thread_do, args=(sock_14,))
sock_15 = openSocket(12015, 15)
t_15 = threading.Thread(target=thread_do, args=(sock_15,))
sock_16 = openSocket(12016, 16)
t_16 = threading.Thread(target=thread_do, args=(sock_16,))
sock_17 = openSocket(12017, 17)
t_17 = threading.Thread(target=thread_do, args=(sock_17,))
sock_18 = openSocket(12018, 18)
t_18 = threading.Thread(target=thread_do, args=(sock_18,))
sock_19 = openSocket(12019, 19)
t_19 = threading.Thread(target=thread_do, args=(sock_19,))

t_0.start()
t_1.start()
t_2.start()
t_3.start()
t_4.start()
t_5.start()
t_6.start()
t_7.start()
t_8.start()
t_9.start()
t_10.start()
t_11.start()
t_12.start()
t_13.start()
t_14.start()
t_15.start()
t_16.start()
t_17.start()
t_18.start()
t_19.start()

t_0.join()
t_1.join()
t_2.join()
t_3.join()
t_4.join()
t_5.join()
t_6.join()
t_7.join()
t_8.join()
t_9.join()
t_10.join()
t_11.join()
t_12.join()
t_13.join()
t_14.join()
t_15.join()
t_16.join()
t_17.join()
t_18.join()
t_19.join()
