import threading, time
import numpy as np


def TestA():
    cond.acquire()
    print('李白：看见一个敌人，请求支援')
    cond.wait()
    print('李白：好的')
    cond.notify()
    cond.release()


def TestB():
    time.sleep(2)
    cond.acquire()
    print('亚瑟：等我...')
    cond.notify()
    print("开始跑")
    time.sleep(4)
    print("跑到了")
    cond.wait()
    print('亚瑟：我到了，发起冲锋...')


if __name__ == '__main__':
    alpha = 0.5
    al = [[1, 2, 3], [4, 5, 6]]
    bl = [[2, 3, 4], [5, 6, 7]]
    a = np.array(al)
    b = np.array(bl)
    print(a)
    print(b)
    c = ((1 - alpha) * a) + (alpha * b)
    print(c)
    cond = threading.Condition()
    testA = threading.Thread(target=TestA)
    testB = threading.Thread(target=TestB)
    testA.start()
    testB.start()
    testA.join()
    testB.join()
