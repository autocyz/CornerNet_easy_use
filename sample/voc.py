import torch
import numpy as np
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from sample.utils import gaussian_radius, draw_gaussian
from sample.utils import ParseLabel

output_size = [128, 128]
input_size = [511, 511]
guassion_iou = 0.3
categories = 1


class VOC(Dataset):
    def __init__(self, root, image_set="train", transform=None):
        super(VOC, self).__init__()
        self.root = root
        self.width_ratio = output_size[1] / input_size[1]
        self.height_ratio = output_size[0] / input_size[0]
        self.max_tag_len = 128
        self.mode = image_set

        annotation_dir = os.path.join(root, "Annotations")
        image_dir = os.path.join(root, "JPEGImages")
        split_dir = os.path.join(root, "ImageSets/Main")

        split_f = os.path.join(split_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError(
                'Wrong image_set entered! Please use image_set="train" '
                'or image_set="trainval" or image_set="val" or a valid'
                'image_set from the VOC ImageSets/Main folder.')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]

    def __getitem__(self, index):
        image_path = self.images[index]
        anno_path = self.annotations[index]

        parse = ParseLabel(anno_path)
        bboxs = parse.get_bboxes()

        img = cv2.imread(image_path)

        bboxs = np.asarray(bboxs, dtype=np.float64)
        img, bboxs = _resize_image(img, bboxs, input_size)
        bboxs = _clip_detections(img, bboxs)

        # src_img = img.copy()
        # for bbox in bboxs:
        #     cv2.rectangle(src_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(255, 0, 0), thickness=3)
        #

        img = img.transpose((2, 0, 1))
        img = img.astype(np.float32) / 255

        tl_heatmap = np.zeros((categories, output_size[0], output_size[1]), dtype=np.float32)
        br_heatmap = np.zeros((categories, output_size[0], output_size[1]), dtype=np.float32)
        tl_tags = np.zeros((self.max_tag_len), dtype=np.int64)
        br_tags = np.zeros((self.max_tag_len), dtype=np.int64)
        tag_masks = np.zeros((self.max_tag_len), dtype=np.uint8)

        for index, bbox in enumerate(bboxs):

            xtl, ytl = bbox[0], bbox[1]
            xbr, ybr = bbox[2], bbox[3]

            fxtl = xtl * self.width_ratio
            fytl = ytl * self.height_ratio
            fxbr = xbr * self.width_ratio
            fybr = ybr * self.height_ratio

            xtl, ytl, xbr, ybr = int(fxtl), int(fytl), int(fxbr), int(fybr)

            width = int((bbox[2] - bbox[0]) * self.width_ratio)
            height = int((bbox[3] - bbox[1]) * self.height_ratio)

            radius = gaussian_radius((height, width), guassion_iou)
            radius = max(0, int(radius))

            draw_gaussian(tl_heatmap[0], [xtl, ytl], radius=radius)
            draw_gaussian(br_heatmap[0], [xbr, ybr], radius=radius)

            tl_tags[index] = ytl * output_size[1] + xtl
            br_tags[index] = ybr * output_size[1] + xbr
            tag_masks[index] = 1


        # tl_heatmap = tl_heatmap*255
        # br_heatmap = br_heatmap*255
        # tl_heatmap = tl_heatmap.astype(np.uint8)
        # br_heatmap = br_heatmap.astype(np.uint8)
        # src_img = cv2.resize(src_img, (output_size[0], output_size[1]))
        # cv2.imshow("src_img", src_img)
        # cv2.imshow("tl_heatmap", tl_heatmap[0])
        # cv2.imshow("br_heatmap", br_heatmap[0])
        # cv2.waitKey(0)

        return img, tl_heatmap, br_heatmap, tl_tags, br_tags, tag_masks

    def __len__(self):
        return len(self.images)


def _resize_image(image, detections, size):
    detections = detections.copy()
    height, width = image.shape[0:2]
    new_height, new_width = size

    image = cv2.resize(image, (new_width, new_height))

    height_ratio = new_height / height
    width_ratio = new_width / width
    detections[:, 0:4:2] *= width_ratio
    detections[:, 1:4:2] *= height_ratio
    return image, detections


def _clip_detections(image, detections):
    detections = detections.copy()
    height, width = image.shape[0:2]

    detections[:, 0:4:2] = np.clip(detections[:, 0:4:2], 0, width - 1)
    detections[:, 1:4:2] = np.clip(detections[:, 1:4:2], 0, height - 1)
    keep_inds = ((detections[:, 2] - detections[:, 0]) > 0) & \
                ((detections[:, 3] - detections[:, 1]) > 0)
    detections = detections[keep_inds]
    return detections




if __name__ == "__main__":
    import random
    root = "/Users/chenyongzhi/data/dataset/voc/VOCdevkit/VOC2012"
    image_set = "val"
    dataset = VOC(root, image_set)

    while(1):
        index = random.randint(0, len(dataset) - 2)
        dataset.__getitem__(index)

