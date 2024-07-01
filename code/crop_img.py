import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
from ultralytics.data.utils import img2label_paths
from ultralytics.utils.ops import xywh2xyxy

IMG_FORMATS = (
    "bmp",
    "dng",
    "jpeg",
    "jpg",
    "mpo",
    "png",
    "tif",
    "tiff",
    "webp",
)

CLASS = (
    "amoeba",
    "apusomonas",
    "ciliate",
    "cysts",
    "flagellate-classical",
    "flagellate-hemi",
    "hemimastix",
    "hypotrich",
    "monothalamid",
    "nematode",
    "snail ciliate"
)

def extract_boxes(path=r'G:\lund\master_thesis\DATA_VERSION\temp.v1i.yolov8'):
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
        if im_file.suffix[1:] in IMG_FORMATS:  # 必须得是图片文件
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
                    cc = CLASS[c]
                    # 生成新'file_name path\classifier\class_index\image_name'
                    # 如: 'F:\yolo_v5\datasets\coco128\images\train2017\classifier\45\train2017_000000000009_0.jpg'
                    f = (path / 'classifier') / f'{cc}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    # f.parent: 'F:\yolo_v5\datasets\coco128\images\train2017\classifier\45'
                    if not f.parent.is_dir():
                        # 每一个类别的第一张照片存进去之前 先创建对应类的文件夹
                        f.parent.mkdir(parents=True)

                    # b = x[1:] * [w, h, w, h]  # box   normalized to 正常大小
                    b = np.multiply(x[1:], [w, h, w, h])
                    # b[2:] = b[2:].max()   pad: rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(int)  # xywh to xyxy

                    # 防止b出界 clip boxes outside of image
                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im0[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


extract_boxes()

