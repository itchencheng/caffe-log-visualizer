#coding:utf-8

import matplotlib.pyplot as plt
import sys
import re

'''
I0329 00:23:11.442008 11635 solver.cpp:239] Iteration 2800 (5.43807 iter/s, 18.3889s/100 iters), loss = 0.542603
I0329 00:23:11.442271 11635 solver.cpp:258]     Train net output #0: SoftmaxWithLoss1 = 0.530918 (* 1 = 0.530918 loss)
I0329 00:23:11.442291 11635 sgd_solver.cpp:112] Iteration 2800, lr = 0.1
I0329 00:23:29.832137 11635 solver.cpp:239] Iteration 2900 (5.43786 iter/s, 18.3896s/100 iters), loss = 0.550942
I0329 00:23:29.832197 11635 solver.cpp:258]     Train net output #0: SoftmaxWithLoss1 = 0.476483 (* 1 = 0.476483 loss)
I0329 00:23:29.832211 11635 sgd_solver.cpp:112] Iteration 2900, lr = 0.1
I0329 00:23:47.311944 11642 data_layer.cpp:73] Restarting data prefetching from start.
I0329 00:23:48.046079 11635 solver.cpp:347] Iteration 3000, Testing net (#0)
I0329 00:23:53.550405 11643 data_layer.cpp:73] Restarting data prefetching from start.
I0329 00:23:53.774024 11635 solver.cpp:414]     Test net output #0: Accuracy1 = 0.6872
I0329 00:23:53.774075 11635 solver.cpp:414]     Test net output #1: SoftmaxWithLoss1 = 0.908618 (* 1 = 0.908618 loss)
I0329 00:23:53.960610 11635 solver.cpp:239] Iteration 3000 (4.14456 iter/s, 24.128s/100 iters), loss = 0.514695
I0329 00:23:53.960675 11635 solver.cpp:258]     Train net output #0: SoftmaxWithLoss1 = 0.470131 (* 1 = 0.470131 loss)
I0329 00:23:53.960691 11635 sgd_solver.cpp:112] Iteration 3000, lr = 0.1
'''

def parse_logs(log_file):
    f = open(log_file)

    lines = f.readlines()

    train_iter_list = []
    train_loss_list = []

    test_iter_list = []
    test_loss_list = []

    test_acc_list = []

    lr_iter_list = []
    lr_list = []

    train_iter_re = r"Iteration ([0-9]+) \("
    train_loss_re = r"loss = ([.0-9]+)"

    test_iter_re = r"Iteration ([0-9]+), Testing"
    test_loss_re = r"SoftmaxWithLoss1 = ([.0-9]+)" 

    test_acc_re = r"Accuracy1 = ([.0-9]+)"
    
    lr_re = r"Iteration ([0-9]+), lr = ([.0-9]+)"
    
    for line in lines:

        train_iter = re.findall(train_iter_re, line)
        if (len(train_iter) > 0):
            train_iter_list.append(int(train_iter[0]))
            print(train_iter)

        train_loss = re.findall(train_loss_re, line)
        if (len(train_loss) > 0):
            train_loss_list.append(float(train_loss[0]))
            print(train_loss)

        test_iter = re.findall(test_iter_re, line)
        if (len(test_iter) > 0):
            test_iter_list.append(int(test_iter[0]))
            print(test_iter)

        test_loss = re.findall(test_loss_re, line)
        if (len(test_loss) > 0):
            test_loss_list.append(float(test_loss[0]))
            print(test_loss)

        test_acc = re.findall(test_acc_re, line)
        if (len(test_acc) > 0):
            test_acc_list.append(float(test_acc[0]))
            print(test_acc)

        lr = re.findall(lr_re, line)
        if (len(lr) > 0):
            print(lr)
            lr_iter_list.append(int(lr[0][0]))
            lr_list.append(float(lr[0][1]))
            print(lr)

    f.close()



    # --------- plot -----------
    plt.title('Result Analysis')

    print(len(train_iter_list), len(train_loss_list))
    plt.plot(train_iter_list, train_loss_list[:len(train_iter_list)], color='green', label='train-loss')
    plt.plot(test_iter_list, test_acc_list, color='red', label='test-acc')
    plt.legend() # 显示图例

    plt.xlabel('iteration times')
    plt.ylabel('rate')
    plt.show()



def main():
    log_file = sys.argv[1]
    print(log_file)

    parse_logs(log_file)

    print("complete!")


if __name__ == "__main__":
    main()