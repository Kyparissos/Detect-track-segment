
import glob          # python自己带的一个文件操作相关模块 查找符合自己目的的文件(如模糊匹配)
import hashlib       # 哈希模块 提供了多种安全方便的hash方法
import json          # json文件操作模块
import logging       # 日志模块
import math          # 数学公式模块
import os            # 与操作系统进行交互的模块 包含文件路径操作和解析
import random        # 生成随机数模块
import shutil        # 文件夹、压缩包处理模块
import time          # 时间模块 更底层
from itertools import repeat                        # 复制模块
from multiprocessing.pool import ThreadPool, Pool   # 多线程模块 线程池
from pathlib import Path               # Path将str转换为Path对象 使字符串路径易于操作的模块
from threading import Thread           # 多线程操作模块

import cv2                             # opencv模块
import numpy as np                     # numpy矩阵操作模块
import matplotlib
import sys

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt        # matplotlib画图模块
import torch                           # PyTorch深度学习模块
import torch.nn.functional as F        # PyTorch函数接口 封装了很多卷积、池化等函数
import yaml                            # yaml文件操作模块
from PIL import Image, ExifTags        # 图片、相机操作模块
from torch.utils.data import Dataset   # 自定义数据集模块
from tqdm import tqdm                  # 进度条模块

from ultralytics.utils.ops import xywh2xyxy
# from utils.torch_utils import torch_distributed_zero_first

# Parameters
help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
num_threads = min(8, os.cpu_count())  # 定义多线程个数
logger = logging.getLogger(__name__)  # 初始化日志

def img2label_paths(img_paths):
    """用在LoadImagesAndLabels模块的__init__函数中
    根据imgs图片的路径找到对应labels的路径
    Define label paths as a function of image paths
    :params img_paths: {list: 50}  整个数据集的图片相对路径  例如: '..\\datasets\\VOC\\images\\train2007\\000012.jpg'
                                                        =>   '..\\datasets\\VOC\\labels\\train2007\\000012.jpg'
    """
    # 因为python是跨平台的,在Windows上,文件的路径分隔符是'\',在Linux上是'/'
    # 为了让代码在不同的平台上都能运行，那么路径应该写'\'还是'/'呢？ os.sep根据你所处的平台, 自动采用相应的分隔符号
    # sa: '\\images\\'    sb: '\\labels\\'
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    # 把img_paths中所以图片路径中的images替换为labels
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

def extract_boxes(path):
    """自行使用 生成分类数据集
    将目标检测数据集转化为分类数据集 集体做法: 把目标检测数据集中的每一个gt拆解开 分类别存储到对应的文件当中
    Convert detection dataset into classification dataset, with one directory per class
    使用: from utils.datasets import *; extract_boxes()
    :params path: 数据集地址
    """
    path = Path(path)  # images dir 数据集文件目录 默认'..\datasets\coco128'
    # remove existing path / 'classifier' 文件夹
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None
    files = list(path.rglob('*.*'))  # 递归遍历path文件下的'*.*'文件
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in img_formats:  # 必须得是图片文件
            # image
            im0 = cv2.imread(str(im_file))  # BGR
            im = im0[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]  # 得到这张图片h w

            # labels 根据这张图片的路径找到这张图片的label路径
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file, 'r') as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # 读取label的各行: 对应各个gt坐标

                for j, x in enumerate(lb):  # 遍历每一个gt
                    c = int(x[0])  # class
                    # 生成新'file_name path\classifier\class_index\image_name'
                    # 如: 'F:\yolo_v5\datasets\coco128\images\train2017\classifier\45\train2017_000000000009_0.jpg'
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    # f.parent: 'F:\yolo_v5\datasets\coco128\images\train2017\classifier\45'
                    if not f.parent.is_dir():
                        # 每一个类别的第一张照片存进去之前 先创建对应类的文件夹
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box   normalized to 正常大小
                    # b[2:] = b[2:].max()   pad: rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)  # xywh to xyxy

                    # 防止b出界 clip boxes outside of image
                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im0[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'

if __name__ == '__main__':
    extract_boxes(path='datasets/Protists2-1')