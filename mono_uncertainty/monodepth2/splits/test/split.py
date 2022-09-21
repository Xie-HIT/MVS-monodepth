from __future__ import absolute_import, division, print_function

import os
from utils import *

fpath = os.path.join(os.path.dirname(__file__), "train_files.txt")
new_fpath = os.path.join(os.path.dirname(__file__), "train_files_new.txt")

train_filenames = readlines(fpath)
with open(new_fpath, 'w+') as f:
    for idx in range(len(train_filenames)):
        line = train_filenames[idx].split()
        timestamp = line[0]
        img_name = line[1]
        img_name = img_name[:-4]

        f.writelines([img_name + '\t', '0\t', 'l\n'])
