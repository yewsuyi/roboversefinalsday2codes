import cv2
import numpy as np


CONF_THRES = 0.25
IOU_THRES = 0.45


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms_boxes(boxes, scores, iou_thres):
    boxes_for_nms = boxes.copy()
    boxes_for_nms[:, 2] = boxes[:, 2] - boxes[:, 0]
    boxes_for_nms[:, 3] = boxes[:, 3] - boxes[:, 1]

    idxs = cv2.dnn.NMSBoxes(
        boxes_for_nms.tolist(),
        scores.tolist(),
        score_threshold=0,
        nms_threshold=iou_thres,
    )
    if len(idxs) == 0:
        return []

    return idxs.flatten()


def decode_yolov11_rknn(outputs, img_shape, model_input_size=(640, 640)):
    pred = outputs[0]

    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]

    if pred.ndim != 2:
        raise ValueError(f"Unsupported RKNN YOLO output shape: {pred.shape}")

    if pred.shape[1] < 5 <= pred.shape[0]:
        pred = pred.transpose(1, 0)
    elif pred.shape[0] < pred.shape[1]:
        pred = pred.transpose(1, 0)

    if pred.shape[1] < 5:
        raise ValueError(f"Expected YOLO output rows to have at least 5 values, got {pred.shape}")

    boxes = pred[:, :4]
    class_scores = sigmoid(pred[:, 4:])
    scores = np.max(class_scores, axis=1)
    class_ids = np.argmax(class_scores, axis=1)

    mask = scores > CONF_THRES
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return [], [], []

    boxes = xywh2xyxy(boxes)

    gain_w = img_shape[1] / model_input_size[0]
    gain_h = img_shape[0] / model_input_size[1]

    boxes[:, [0, 2]] *= gain_w
    boxes[:, [1, 3]] *= gain_h

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img_shape[0])

    idxs = nms_boxes(boxes, scores, IOU_THRES)

    return boxes[idxs], scores[idxs], class_ids[idxs]


def draw_detections(img, boxes, scores, class_ids, class_names):
    for box, score, cls in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        class_name = class_names.get(int(cls), f"class_{int(cls)}")
        label = f"{class_name} {score:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            label,
            (x1, max(y1 - 10, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return img
