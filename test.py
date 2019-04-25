from models.CornerNet import CornerNet
import numpy as np
import cv2
import torch
import os
from inference import kp_detection

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


def draw_detection(net, img_file, result_path=None):
    img = cv2.imread(img_file)
    detections = kp_detection(net, img)
    image = cv2.imread(img_file)
    for j in range(1, categories + 1):
        cat_size = cv2.getTextSize(str(j), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        color = np.random.random((3,)) * 0.6 + 0.4
        color = color * 255
        color = color.astype(np.int32).tolist()

        if categories == 1:
            color = (0, 0, 255)
        for bbox in detections[j]:
            bbox = bbox[0:4].astype(np.int32)
            if bbox[1] - cat_size[1] - 2 < 0:
                cv2.rectangle(image,
                              (bbox[0], bbox[1] + 2),
                              (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 2),
                              color, -1
                              )
                cv2.putText(image, str(j),
                            (bbox[0], bbox[1] + cat_size[1] + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1
                            )
            else:
                cv2.rectangle(image,
                              (bbox[0], bbox[1] - cat_size[1] - 2),
                              (bbox[0] + cat_size[0], bbox[1] - 2),
                              color, -1
                              )
                cv2.putText(image, str(j),
                            (bbox[0], bbox[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1
                            )
            cv2.rectangle(image,
                          (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          color, 2
                          )
    if result_path is not None:
        cv2.imwrite(os.path.join(result_path, os.path.split(img_file)[1]), image)
    print(os.path.join(result_path, os.path.split(img_file)[1]))
    print("process {} over".format(img_file))


def main():
    model_path = "result/checkpoint/0419/epoch_0_3.157.cpkt"
    result_path = "result_img/"
    img_path = "img/"
    img_files = os.listdir(img_path)

    net = CornerNet()
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()
    for f in img_files:
        img_file = os.path.join(img_path, f)
        draw_detection(net, img_file, result_path)


def save_heat():
    model_file_path = "result/checkpoint/0411/epoch_18_0.710.cpkt"
    result_path  = "result_img/"
    img_dir = "img/"

    img_files = os.listdir(img_dir)
    net = CornerNet()
    print("loading model state_dict")
    net.load_state_dict(torch.load(model_file_path))
    print("loading over")

    net.cuda()
    net.eval()
    with torch.no_grad():
        for img_name in img_files:
            img_path = os.path.join(img_dir, img_name)
            print("process img: ", img_path)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (511, 511))
            img = img.transpose((2, 0, 1))
            img = img / 255
            img = torch.from_numpy(img)

            img = img.float().cuda()
            img = img.unsqueeze(0)
            print("img: ", img.shape)
            out = net(img)

            out = [a.cpu().numpy() for a in out]
            np.savez(os.path.join(result_path, img_name),
                     tl_heat=out[0],
                     br_heat=out[1],
                     tl_tag=out[2],
                     br_tag=out[3]
                     )


if __name__ == "__main__":
    main()
    # save_heat()