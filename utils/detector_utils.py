# Utilities for object detector.

import numpy as np
import tensorflow as tf
import cv2
from utils import label_map_util
import time

PATH_TO_LABELS =  'model/mscoco_label_map.pbtxt'

#category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

NUM_CLASSES = 90
# load label map using utils provided by tensorflow object detection api
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

a=b=0

def draw_box_on_image(img,boxes,scores,classes,score_thres=0.25):
    img=img[:,:,::-1] # RGB to BGR
    img= np.ascontiguousarray(img)
    color = (0,255,0)#None
    for i in range(len(classes)):

        if (scores[i] > score_thres and classes[i] in [2,3,5,7]):

            (left, top, right, bottom) = (boxes[i][0],boxes[i][1], boxes[i][2],boxes[i][3])

            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))

            img=cv2.rectangle(img, p1, p2, color , 2,1)

            cv2.putText(img,category_index[c80to91(classes[i])]['name']+str(": {0:.2f}".format(scores[i])), (int(left), int(top)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 , color, 1)
    return img

# Show fps value on image.
def draw_text_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
# compute and return the distance from the hand to the camera using triangle similarity
def distance_to_camera(knownWidth, focalLength, pixelWidth):
    return (knownWidth * focalLength) / pixelWidth


def load_model(model_dir='frozen_graphs/yolov5s.pb'):
    return tf.keras.models.load_model(model_dir)

def preprocessing(img0):
    img_size = 640 # fix
    # Padded resize
    img = rsz_pad(img0, img_size)[0]
    # Convert
    # img = img[:, :, ::-1]    BGR to RGB,
    img = img.transpose(2, 0, 1)  #  to 3x416x416
    img = np.ascontiguousarray(img)
    # Normalize
    img =img/255.0  # 0 - 255 to 0.0 - 1.0
    if len(img) == 3:
        img = img[None,...]
    return img

def detect_objects(frame0,model,conf_thres_nms=0.25,iou_thres_nms=0.30,only_classes=None):
    frame = preprocessing(frame0)
    input_img=tf.convert_to_tensor(frame,dtype=tf.float32)

    pred=model(images=input_img)[0]  #prediction

    pred=non_max_suppression(pred,conf_thres_nms,iou_thres_nms,only_classes,agnostic=True)[0] #non max suppresion

    coords=pred[:,:4].numpy()
    scores=pred[:,4].numpy()
    classes=pred[:,5].numpy().astype('int')
    boxes=scale_coords(frame.shape[2:],coords,frame0.shape[:3])
    return boxes, scores, classes, len(classes)

# tensorflow version
def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,multi_label=False,
                        max_det=300):
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS
    classes= None

    t = time.time()
    output=[tf.zeros((0, 6))] * prediction.shape[0]

    for xi, x in enumerate(prediction):
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x = x.numpy()
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        x = tf.convert_to_tensor(x)

        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = tf.transpose(tf.where(a[:,5:]>conf_thres))
            x = tf.concat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf = tf.math.reduce_max(x[:,5:],1)
            j = tf.argmax(x[:, 5:],1,)
            x = tf.concat((box, conf[:,None], tf.cast(j[:,None],tf.float32)), 1)[conf > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[tf.math.reduce_any(x[:, 5:6] == tf.cast(tf.constant(classes),tf.float32),1)]

        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = tf.gather(x,tf.argsort(x[:, 4],direction='DESCENDING')[:max_nms]) # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

        i = tf.image.non_max_suppression(boxes,scores,iou_threshold=iou_thres,max_output_size=max_det)  # NMS

        output[xi] = tf.gather(x,i)

        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded
    return output

def rsz_pad(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    # print(new_unpad,dw,dh)

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def load_image(path, img_size=640, stride=32):

    img0 = cv2.imread(path)  # BGR
    assert img0 is not None, 'Image Not Found ' + path

    # Padded resize
    img = rsz_pad(img0, img_size, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    return img,img0

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.numpy() if isinstance(x, tf.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return tf.convert_to_tensor(y) if isinstance(x, tf.Tensor) else y

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.numpy() if isinstance(x, tf.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return tf.convert_to_tensor(y) if isinstance(x, tf.Tensor) else y

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0] = tf.clip_by_value(boxes[:, 0],0, img_shape[1])  # x1
    boxes[:, 1] = tf.clip_by_value(boxes[:, 1],0, img_shape[0])  # y1
    boxes[:, 2] = tf.clip_by_value(boxes[:, 2],0, img_shape[1])  # x2
    boxes[:, 3] = tf.clip_by_value(boxes[:, 3],0, img_shape[0])  # y2

    return boxes

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):

    coords=coords.numpy() if isinstance(coords,tf.Tensor) else coords

    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def c80to91(index:int):  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x[index]