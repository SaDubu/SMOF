import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

OBJ_THRESH = 0.35
NMS_THRESH = 0.45
IMG_SIZE = 640

# add path
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
#sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('rknn_model_zoo')+1]))
sys.path.append('/home/orangepi/Projects/rknn_model_zoo/')

from py_utils.coco_utils import COCO_test_helper

class Yolov8() :
    def __init__(self, model_path, target, device_id=0x0) :
        self.model, self.platform = self.setup_model(model_path, target, device_id)

        self.co_helper = COCO_test_helper(enable_letter_box=True)

        self.stop_q = None
        self.frame_q = None
        self.box_q = None
        self.logger = None
        self.no_detect_flag = None

    def set_no_detect_flag(self, q) :
        self.no_detect_flag = q
    
    def set_logger(self, logger_object) :
        self.logger = logger_object
        self.logger.info('yolov8 class object activate')

    def is_stop(self) :
        stop_q_size = self.stop_q.qsize()
        if stop_q_size == 0 :
            return False
        return True

    def set_frame_q(self, q) :
        self.frame_q = q
    
    def set_box_q(self, q) :
        self.box_q = q

    def set_stop_q(self, q) :
        self.stop_q = q

    def setup_model(self, model_path, target, device_id):
        if model_path.endswith('.rknn'):
            platform = 'rknn'
            from py_utils.rknn_executor import RKNN_model_container 
            model = RKNN_model_container(model_path, target, device_id)
        else:
            assert False, "{} is not rknn/pytorch/onnx model".format(model_path)
        print('Model-{} is {} model, starting val'.format(model_path, platform))

    
        return model, platform

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes with object threshold.
        """
        box_confidences = box_confidences.reshape(-1)
        candidate, class_num = box_class_probs.shape

        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)

        _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
        scores = (class_max_score* box_confidences)[_class_pos]

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]

        return boxes, classes, scores

    def nms_boxes(self, boxes, scores):
        """Suppress non-maximal boxes.
        # Returns
            keep: ndarray, index of effective boxes.
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
            h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
            inter = w1 * h1

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= NMS_THRESH)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        return keep

    def dfl(self, position):
        # Distribution Focal Loss (DFL)
        import torch
        x = torch.tensor(position)
        n,c,h,w = x.shape
        p_num = 4
        mc = c//p_num
        y = x.reshape(n,p_num,mc,h,w)
        y = y.softmax(2)
        acc_metrix = torch.tensor(range(mc)).float().reshape(1,1,mc,1,1)
        y = (y*acc_metrix).sum(2)
        return y.numpy()


    def box_process(self, position):
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([IMG_SIZE//grid_h, IMG_SIZE//grid_w]).reshape(1,2,1,1)

        position = self.dfl(position)
        box_xy  = grid +0.5 -position[:,0:2,:,:]
        box_xy2 = grid +0.5 +position[:,2:4,:,:]
        xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

        return xyxy

    def post_process(self, input_data):
        boxes, scores, classes_conf = [], [], []
        defualt_branch=3
        pair_per_branch = len(input_data)//defualt_branch
        # Python 忽略 score_sum 输出
        for i in range(defualt_branch):
            boxes.append(self.box_process(input_data[pair_per_branch*i]))
            classes_conf.append(input_data[pair_per_branch*i+1])
            scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0,2,3,1)
            return _in.reshape(-1, ch)

        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores]

        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores)

        # filter according to threshold
        boxes, classes, scores = self.filter_boxes(boxes, scores, classes_conf)

        # nms
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = self.nms_boxes(b, s)

            if len(keep) != 0:
                nboxes.append(b[keep])
                nclasses.append(c[keep])
                nscores.append(s[keep])

        if not nclasses and not nscores:
            return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        return boxes, classes, scores

    def process_for_run(self, img_src, tracker_input) :
        img = self.co_helper.letter_box(im= img_src.copy(), new_shape=(IMG_SIZE, IMG_SIZE), pad_color=(0,0,0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_data = img

        outputs = self.model.run([input_data])
        boxes, classes, scores = self.post_process(outputs)

        if boxes is None :
            return 1

        for box, cla, sco in zip(boxes, classes, scores) :
            x1, y1, x2, y2 = box
            sort_input = [x1, y1, x2, y2, cla, sco]
            tracker_input.append(sort_input)
        
        return 0

    def run(self) :
        img = None
        end = None

        tracker_input = []
        while True :
            if self.is_stop() :
                break

            if self.frame_q.qsize() == 0 :
                continue

            img = self.frame_q.get()
            
            end = self.process_for_run(img, tracker_input)

            if end == 1 :
                self.logger.error('yolo error')
                self.no_detect_flag.put(True)
                continue
            
            self.box_q.put(tracker_input)

            tracker_input = []

        # release
        self.model.release()
        return 0

    def one_time_run(self, frame) :
        output = []
        middle = []
        end = self.process_for_run(frame, middle)

        if end == 1:
            return 1

        for det in middle :
            output.append({
                'box': det[0:4],
                'cls': det[4]
            })
        
        return output
