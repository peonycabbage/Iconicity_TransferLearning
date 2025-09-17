# -*- coding: utf-8 -*-

import cv2
import os


newpath = r"D:\Keren_SLT\CSL\sentence_frames"
path =r"D:\BaiduNetdiskDownload\CSL_SLT_dataset\CSL2018-zip-tar\sentence-zip\color-sentence\color-sentence1\color-sentence\color"


for category in os.listdir(path):
    categorypath = os.path.join(path, category)
    for video in os.listdir(categorypath):
        videopath = os.path.join(categorypath, video)
        framespath = os.path.join(newpath, category, video)
        if not os.path.exists(framespath):
            os.makedirs(framespath)
        cap = cv2.VideoCapture(videopath)
        
        success,image = cap.read()
        count = 0
        while success:
            cv2.imwrite(framespath + "/" +"frame%d.jpg" % count, image)
            success,image = cap.read()
            print('Read a new frame: ', success)
            count += 1
            