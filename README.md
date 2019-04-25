# CornerNet_easy_use

This is a easy used [CornerNet](https://github.com/princeton-vl/CornerNet) (https://github.com/princeton-vl/CornerNet) project.

In the original project, author use factory model to create CornerNet model and read data by his own wheel. That is not  easy for others' understand. So I simplify this project, and use pytorch's dataset and dataloader. 

### Compiling Corner Pooling Layers

You need to compile the C++ implementation of corner pooling layers.

```bash
cd <CornerNet_easy_use dir>/models/py_utils/_cpools/
python setup.py install --user
```

### Compiling NMS

You also need to compile the NMS code (originally from [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/cpu_nms.pyx) and [Soft-NMS](https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx)).

```python
cd <CornerNet_easy_use dir>/external
make
```

### Train and test

- `train.py` is a sample code for train voc, I use `tensorboardX` to show training info
- `test.py` is sample code for test you images, it will draw rect on the image

