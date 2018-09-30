from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

import numpy as np

np.random.seed(2018)


def proess(filePath,featureLen):
    """
    text process.
    :param filePath:
    :param featureLen:
    :return:
    """
    train = []
    label = []
    word_index = {}

    timestep = featureLen
    filename = []
    for i in os.listdir(filePath):
        for j in os.listdir(os.path.join(filePath, i)):
            filename.append(os.path.join(filePath, i,j))

    stop = 0
    for file in filename:
        if stop < 10000:
            with open(file, "r", encoding="utf-8") as f:
                try:
                    regex = re.compile('\s+')
                    for l in f.readlines():
                        sentence = re.split('[,，。;]', l)
                        for s in sentence:
                            w_temp = []
                            label_temp = []
                            k = re.match(u"[\u4e00-\u9fa5]+", s)
                            if k is not None:
                                # 按多个空格切分
                                k = regex.split(k.string)
                                for word in k:
                                    w_t = word.split("/")[0]
                                    w_l = list(w_t)
                                    for w in range(len(w_l)):
                                        if w_l[w] not in word_index:
                                            word_index[w_l[w]] = len(word_index) + 1
                                        w_temp.append(word_index[w_l[w]])
                                    if len(w_l) == 1:
                                        label_temp.append(4)
                                    else:
                                        for i in range(len(w_l)):
                                            if i == 0:
                                                label_temp.append(1)
                                            elif i == len(w_l) - 1:
                                                label_temp.append(3)
                                            else:
                                                label_temp.append(2)
                            if len(w_temp) != 0 and len(w_temp) - timestep < 0:
                                train.append(np.pad(w_temp, (0, timestep - len(w_temp)), mode="constant"))
                                label.append(np.pad(label_temp, (0, timestep - len(label_temp)), mode="constant"))

                except Exception as e:

                    print(e)
            stop += 1
    return train, label, word_index
