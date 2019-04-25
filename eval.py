from models.CornerNet import CornerNet
import numpy as np
import cv2
import torch
import os
from inference import kp_detection
from sample.utils import ParseLabel
from eval.BoundingBox import BoundingBox
from eval.BoundingBoxes import BoundingBoxes
from eval.utils import *
from eval.Evaluator import Evaluator
import tqdm

categories = 1
cls = ["object"]


def eval():
    split_f = "/home/cyz/data/dataset/ourlabeled/object_color/val.txt"
    gt_root = "/home/cyz/data/dataset/ourlabeled/object_color/gt_txt"
    det_root = "result_det"

    allbboxs = BoundingBoxes()

    with open(os.path.join(split_f), "r") as f:
        file_names = [x.strip() for x in f.readlines()]

    print("reading bbox ...")
    # read gt box
    for img_name in file_names:
        f = os.path.join(os.path.join(gt_root, img_name + ".txt"))
        if not os.path.isfile(f):
            print("warning: {} not exist".format(f))

        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0]  # class
            x = float(splitLine[1])  # confidence
            y = float(splitLine[2])
            w = float(splitLine[3])
            h = float(splitLine[4])

            bb = BoundingBox(
                img_name,
                idClass,
                x,
                y,
                w,
                h,
                typeCoordinates=CoordinatesType.Absolute,
                bbType=BBType.GroundTruth,
                format=BBFormat.XYX2Y2)
            allbboxs.addBoundingBox(bb)
        fh1.close()

        # read detection bbox
        f = os.path.join(os.path.join(det_root, img_name + ".txt"))
        if not os.path.isfile(f):
            print("warning: {} not exist".format(f))

        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0]  # class
            confidence = float(splitLine[1])  # confidence
            x = float(splitLine[2])
            y = float(splitLine[3])
            w = float(splitLine[4])
            h = float(splitLine[5])
            bb = BoundingBox(
                img_name,
                idClass,
                x,
                y,
                w,
                h,
                classConfidence=confidence,
                typeCoordinates=CoordinatesType.Absolute,
                bbType=BBType.Detected,
                format=BBFormat.XYX2Y2)
            allbboxs.addBoundingBox(bb)
        fh1.close()
    print("... reading over")

    evalthresh = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    evaluator = Evaluator()

    total_ap = 0
    for thre in evalthresh:
        metricsPerClass = evaluator.GetPascalVOCMetrics(allbboxs, thre, MethodAveragePrecision.EveryPointInterpolation)
        for mc in metricsPerClass:
            # Get metric values per each class
            precision = mc['precision']
            recall = mc['recall']
            average_precision = mc['AP']

            # Print AP per class
            print('thre: {} AP: {}'.format(thre, average_precision))
            total_ap += average_precision
    print("mAP: ", total_ap / len(evalthresh))


def creat_det_txt():
    split_f = "/home/cyz/data/dataset/ourlabeled/object_color/val.txt"
    img_root = "/home/cyz/data/dataset/ourlabeled/object_color"
    result_dir = "result_det"
    model_path = "result/checkpoint/0419/epoch_0_3.157.cpkt"

    print("loading model....")
    net = CornerNet()
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()
    print("...over loading")

    with open(os.path.join(split_f), "r") as f:
        file_names = [x.strip() for x in f.readlines()]

    for img_name in file_names:
        img_path = os.path.join(img_root, img_name.replace(".xml", ".jpg"))
        print("process: ", img_path)
        dets = kp_detection(net, img_path)
        for j in range(1, categories + 1):

            result_path = os.path.split(os.path.join(result_dir, img_name))[0]
            print(result_path)
            if not os.path.exists(result_path):
                os.makedirs(result_path)

            with open(os.path.join(result_dir, img_name + '.txt'), 'w') as f:
                for bbox in dets[img_path][j]:
                    score = bbox[4]
                    bbox = bbox[0:4].astype(np.int32)
                    f.write(cls[j-1])
                    f.write(' ')
                    f.write(str(score))
                    for cor in bbox:
                        f.write(' ')
                        f.write(str(cor))
                    f.write('\n')
        print("...over process")


def creat_gt_txt():
    root = "/home/cyz/data/dataset/ourlabeled/object_color"
    file_names = "/home/cyz/data/dataset/ourlabeled/object_color/trainval.txt"
    result_dir = "/home/cyz/data/dataset/ourlabeled/object_color/gt_txt"

    with open(os.path.join(file_names), "r") as f:
        file_names = [x.strip() for x in f.readlines()]

        for file_name in file_names:
            print(file_name)
            name = os.path.split(file_name)[1]
            file_path = os.path.join(root, file_name)
            parselabel = ParseLabel(file_path)
            bboxs = parselabel.get_bboxes()

            result_path = os.path.split(os.path.join(result_dir, file_name))[0]
            print(result_path)
            if not os.path.exists(result_path):
                os.makedirs(result_path)

            print("change {}".format(os.path.join(result_dir, file_name + ".txt")))
            with open(os.path.join(result_dir, file_name+".txt"), 'w') as f:
                for box in bboxs:
                    f.write("object")
                    for cor in box:
                        f.write(' ')
                        f.write(str(cor))
                    f.write('\n')


if __name__ == "__main__":
    # creat_det_txt()
    creat_gt_txt()
    # eval()
