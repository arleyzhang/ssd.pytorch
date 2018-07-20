import sys
import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
import time

ITERATION_START_PATTERN = re.compile(r'iter \d+.* (\|\| Loss:) \d+\.{0,1}\d*')

ITERATION_PATTERN = re.compile(r'iter \d+')
OUTPUT_PATTERN = re.compile(r'(Loss:|accuracy:) \d+\.{0,1}\d*')
LOSS_PATTERN = re.compile(r'(Loss:) \d+\.{0,1}\d*')
TEMP_PATTERN = re.compile(r'(conf_loss:) \d+\.{0,1}\d*')


def plot(name, data, label=None, save_file=None):
    data = np.array(data, dtype="float32")
    x = data[:, 0].astype(np.int)
    y = data[:, 1]
    # if name.find('loss') >= 0:
    #     y = np.log(y + 1)
    if not save_file is None:
        fw = open(save_file, 'w')
        for i in range(len(x)):
            fw.write(str(x[i]) + ' ' + str(y[i]) + '\n')
        fw.close()
    if not label is None:
        plt.plot(x, y, 'x-', label=label)
        return
    if len(x) > 2 and x[1] - x[0] < 100000:
        plt.plot(x, y, 'x-')  # train acc
    else:
        plt.plot(x, y, 'x-', linewidth='3', color='black')  # test_acc


def next_line_index(lines, pattern, start=0):
    for i in xrange(start, len(lines)):
        line = lines[i].strip()  # remove special str in front and end
        match = pattern.search(line)
        if match:
            return i
    return -1


def get_data(log_file):
    infile = open(log_file, 'r')
    lines = infile.readlines()
    start = next_line_index(lines, ITERATION_START_PATTERN, 0)

    result_dict = dict()

    while start >= 0:
        if start % 100 == 0:
            iter_match = ITERATION_PATTERN.search(lines[start])
            iter_num = 0
            if iter_match:
                iter_match = iter_match.group()  # return 'Iteration 100'
                iter_num = int(iter_match.split(' ')[-1])
            else:
                print("iter_num is bad")
                return None

            match = OUTPUT_PATTERN.search(lines[start])
            if match:
                name, value = map(lambda x: x.strip(),
                                  match.group().split(': '))  # accuracy, 0.88
                if not result_dict.has_key(name):
                    result_dict[name] = list()
                result_dict[name].append([iter_num, float(value)])

            match = LOSS_PATTERN.search(lines[start])
            if match:
                name, value = map(lambda x: x.strip(),
                                  match.group().split(': '))  # accuracy, 0.88
                if not result_dict.has_key(name):
                    result_dict[name] = list()
                result_dict[name].append([iter_num, float(value)])

            match = TEMP_PATTERN.search(lines[start])
            if match:
                name, value = map(lambda x: x.strip(),
                                  match.group().split(': '))  # accuracy, 0.88
                if not result_dict.has_key(name):
                    result_dict[name] = list()
                result_dict[name].append([iter_num, float(value)])

            start = next_line_index(lines, ITERATION_START_PATTERN, start + 1)

    return result_dict


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Hint : python plot_*.py log_*.log")
        os._exit(0)
    log_file = sys.argv[1]  # src h5 file
    # fcn_10 = get_data("log_fcn8_5c_tain0_1e-10.log")
    # fcn_9 = get_data("log_fcn8_5c_tain0_1e-9.log")
    # fcn_8 = get_data("log_fcn8_5c_tain0_1e-8.log")

    #cnn11 = get_data("./Brain_patches/log_cnn11_train0.log")

    # log_test180620.log
    repul_loss = get_data(log_file)  # log_test180619.log#test001.log
    plt.figure()

    # for key in fcn_8.keys():
    #     plot(key, fcn_8[key], 'lr=1e-8')

    # for key in fcn_9.keys():
    #     plot(key, fcn_9[key], 'lr=1e-9')

    # for key in fcn_10.keys():
    #     plot(key, fcn_10[key], 'lr=1e-10')

    for key in repul_loss.keys():
        # print key
        plot(key, repul_loss[key], '{}'.format(key))
    plt.legend(loc='upper right')  # lower right
    plt.ylabel('LOSS')
    plt.xlabel('Iteration')
    plt.show()

    plt.close()

    # display = 20
    # test_interval = 5000
    # _, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # for key in cnn11.keys():
    #     #plot(key, cnn11[key])
    #     if 'loss' in key:
    #         train_loss = []# * len(cnn11[key])
    #         for item_ in cnn11[key]:
    #             train_loss.append(item_[1])
    #             # train_loss[item_[0]] = item_[1]
    #     if 'acc' == key:
    #         train_acc= []# * len(cnn11[key])
    #         for item_ in cnn11[key]:
    #             train_acc.append(item_[1])
    #             # train_acc[item_[0]] = item_[1]
    #     if 'accuracy' == key:
    #         test_acc= []# * len(cnn11[key])
    #         for item_ in cnn11[key]:
    #             test_acc.append(item_[1])
    #             # test_acc[item_[0]] = item_[1]

    # ax1.plot(display * np.arange(len(train_loss)), train_loss,'b')
    # ax2.plot(display * np.arange(len(train_acc)), train_acc,'g')
    # ax2.plot(test_interval * np.arange(len(test_acc)), test_acc,'r')

    # ax1.set_xlabel('iter')
    # ax1.set_ylabel('loss')
    # ax2.set_ylabel('accuracy')
    # plt.show()
