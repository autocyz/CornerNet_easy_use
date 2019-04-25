import numpy as np
import cv2
import torch
from utils.image import crop_image
from models.py_utils.kp_utils import _decode_no_reg
from external.nms import soft_nms

scales = [1]
# img_mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
# img_std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
categories = 1
nms_threshold = 0.5
nms_algorithm = {
    "nms": 0,
    "linear_soft_nms": 1,
    "exp_soft_nms": 2
}["exp_soft_nms"]

input_size = (511, 511)


def _rescale_dets(detections, ratios, borders, sizes):
    xs, ys = detections[..., 0:4:2], detections[..., 1:4:2]
    xs /= ratios[:, 1][:, None, None]
    ys /= ratios[:, 0][:, None, None]
    xs -= borders[:, 2][:, None, None]
    ys -= borders[:, 0][:, None, None]
    np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
    np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)


def kp_decode(net, img, K=100, ae_threshold=0.5, kernel=3):
    with torch.no_grad():
        out = net(img)
    detections = _decode_no_reg(*out[-4:], K=K, kernel=kernel, ae_threshold=ae_threshold)
    detections = detections.data.cpu().numpy()
    return detections


def kp_detection(net, img):
    """
    get detection
    Args:
        net:
        img_file:

    Returns: a dict {img_file: {cls1: }}

    """
    K = 100

    width_scale = img.shape[1] / input_size[1]
    height_scale = img.shape[0] / input_size[0]

    # >> resize
    img = cv2.resize(img, input_size)
    height, width = img.shape[0:2]
    top_bboxes = {}

    detections = []

    for scale in scales:
        new_height = int(height * scale)
        new_width = int(width * scale)
        new_center = np.array([new_height // 2, new_width // 2])

        inp_height = new_height | 127
        inp_width = new_width | 127

        images = np.zeros((1, 3, inp_height, inp_width), dtype=np.float32)
        ratios = np.zeros((1, 2), dtype=np.float32)
        borders = np.zeros((1, 4), dtype=np.float32)
        sizes = np.zeros((1, 2), dtype=np.float32)

        out_height, out_width = (inp_height + 1) // 4, (inp_width + 1) // 4
        height_ratio = out_height / inp_height
        width_ratio = out_width / inp_width

        resized_image = cv2.resize(img, (new_width, new_height))
        resized_image, border, offset = crop_image(resized_image, new_center, [inp_height, inp_width])

        resized_image = resized_image / 255.
        images[0] = resized_image.transpose((2, 0, 1))
        borders[0] = border
        sizes[0] = [int(height * scale), int(width * scale)]
        ratios[0] = [height_ratio, width_ratio]

        images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images = torch.from_numpy(images)
        images = images.cuda()
        dets = kp_decode(net, images, K)
        dets = dets.reshape(2, -1, 8)
        dets[1, :, [0, 2]] = out_width - dets[1, :, [2, 0]]
        dets = dets.reshape(1, -1, 8)

        _rescale_dets(dets, ratios, borders, sizes)
        dets[:, :, 0:4] /= scale
        detections.append(dets)

    detections = np.concatenate(detections, axis=1)

    classes = detections[..., -1]
    classes = classes[0]
    detections = detections[0]

    # reject detections with negative scores
    keep_inds = (detections[:, 4] > -1)
    detections = detections[keep_inds]
    classes = classes[keep_inds]

    for j in range(categories):
        keep_inds = (classes == j)
        top_bboxes[j + 1] = detections[keep_inds][:, 0:7].astype(np.float32)
        soft_nms(top_bboxes[j + 1], Nt=nms_threshold, method=nms_algorithm)
        top_bboxes[j + 1] = top_bboxes[j + 1][:, 0:5]
        top_bboxes[j + 1][:, 0:4:2] *= width_scale
        top_bboxes[j + 1][:, 1:4:2] *= height_scale

        top_bboxes[j + 1] = top_bboxes[j + 1][top_bboxes[j + 1][:, -1] > 0.5]

    return top_bboxes
