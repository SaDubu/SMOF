import os
import cv2
import sys
import argparse
import yaml

# add path
realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
#sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('rknn_model_zoo')+1]))
sys.path.append('/home/orangepi/Projects/rknn_model_zoo/')

from py_utils.coco_utils import COCO_test_helper
import numpy as np
from PIL import Image, ImageDraw, ImageFont


OBJ_THRESH = 0.25
NMS_THRESH = 0.45

# The follew two param is for map test
# OBJ_THRESH = 0.001
# NMS_THRESH = 0.65

IMG_SIZE = (480, 480)  # (width, height), such as (1280, 736)

def get_yaml_info(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    names = data.get('names', {})
    
    id_list = sorted(names.keys())
    
    class_list = [names[i] for i in id_list]
    
    return tuple(class_list), id_list

yaml_file = "data.yaml" 
CLASSES, coco_id_list = get_yaml_info(yaml_file)


def filter_boxes(boxes, box_confidences, box_class_probs):
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

def nms_boxes(boxes, scores):
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

def dfl(position):
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


def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    position = dfl(position)
    box_xy  = grid +0.5 -position[:,0:2,:,:]
    box_xy2 = grid +0.5 +position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy

def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch=3
    pair_per_branch = len(input_data)//defualt_branch
    # Python 忽略 score_sum 输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
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

    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dfl_optimized(x):
    # x의 형태가 (N, 64, H, W)이든 (N, 64)이든 대응 가능하게 수정
    shape = x.shape
    if len(shape) == 4: # 기존 방식 (전체 이미지 처리 시)
        n, c, h, w = shape
        x = x.reshape(n, 4, 16, h, w).transpose(0, 1, 3, 4, 2)
    elif len(shape) == 2: # 최적화 방식 (필터링된 데이터 처리 시)
        n, c = shape
        x = x.reshape(n, 4, 16)
    else:
        raise ValueError(f"예상치 못한 데이터 형태입니다: {shape}")

    # Softmax 연산 (16개 위치 확률 분포)
    # 수치 안정성을 위해 max 값을 빼주는 것이 좋지만, 속도를 위해 생략하거나 아래처럼 작성
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    x = exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    # 가중치 [0, 1, 2, ..., 15] 곱해서 합산
    acc_matrix = np.arange(16, dtype=np.float32)
    x = np.sum(x * acc_matrix, axis=-1)
    
    return x # (N, 4) 형태로 반환됨

def box_process_optimized(position_raw, mask_indices, grid_h, grid_w, img_size):
    """
    전체 그리드를 생성하지 않고, 선택된 인덱스(mask_indices)에 대해서만 
    박스 좌표를 복원합니다.
    """
    # 1. DFL 처리 (64채널 -> 4채널: top, left, bottom, right)
    # position_raw shape: (N, 64)
    pos = dfl_optimized(position_raw) # dfl 함수가 (N, 64) 입력을 지원해야 함
    
    # 2. 선택된 위치의 그리드 좌표 생성
    # mask_indices는 (y_coords, x_coords) 형태임
    rows, cols = mask_indices
    grid = np.stack((cols, rows), axis=1) # (N, 2) -> x, y 순서
    
    # 3. 스트라이드 계산
    stride_y = img_size[1] // grid_h
    stride_x = img_size[0] // grid_w
    stride = np.array([stride_x, stride_y])

    # 4. YOLOv8 박스 복원 공식 (Center + 0.5 기반)
    # pos[:, 0:2]는 좌상단 거리, pos[:, 2:4]는 우하단 거리
    box_xy1 = grid + 0.5 - pos[:, 0:2]
    box_xy2 = grid + 0.5 + pos[:, 2:4]
    
    # 5. 이미지 크기에 맞게 스케일링
    xyxy = np.concatenate((box_xy1 * stride, box_xy2 * stride), axis=1)
    
    return xyxy

def weck_post_process(input_data, conf_threshold=0.25) :
    logit_threshold = -np.log(1 / conf_threshold - 1) # 로짓 임계값 역산
    
    all_boxes, all_scores, all_class_ids = [], [], []

    for branch_data in input_data:
        # 1. 기본 정보 추출
        grid_h, grid_w = branch_data.shape[2:4]
        b_box_raw = branch_data[0, :64, :, :]
        b_cls_raw = branch_data[0, 64:, :, :]

        # 2. 로짓 단계에서 미리 필터링 (병목 제거 1단계)
        max_logits = np.max(b_cls_raw, axis=0)
        mask = max_logits > logit_threshold
        
        if not np.any(mask):
            continue

        # 3. 유효한 인덱스와 해당 데이터만 추출
        mask_indices = np.where(mask) # (y_indices, x_indices)
        matched_box_raw = b_box_raw[:, mask].T # (N, 64)
        matched_cls_raw = b_cls_raw[:, mask].T # (N, Class_count)

        # 4. 클래스 확률 및 스코어 계산 (병목 제거 2단계)
        probs = 1 / (1 + np.exp(-matched_cls_raw))
        scores = np.max(probs, axis=1)
        class_ids = np.argmax(probs, axis=1)

        # 5. 최적화된 박스 좌표 복원 (병목 제거 3단계)
        boxes = box_process_optimized(matched_box_raw, mask_indices, grid_h, grid_w, IMG_SIZE)

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_class_ids.append(class_ids)

    if not all_boxes:
        return None, None, None

    # 데이터 통합
    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    classes = np.concatenate(all_class_ids)

    # NMS (이미 데이터가 압축되어 매우 빠름)
    nboxes, nclasses, nscores = [], [], []
    for c_id in np.unique(classes):
        inds = np.where(classes == c_id)
        b, s = boxes[inds], scores[inds]
        
        keep = nms_boxes(b, s)
        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(np.full(len(keep), c_id))
            nscores.append(s[keep])

    if not nboxes:
        return None, None, None

    return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)

def add_post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    
    for branch_data in input_data:
        b_box_raw = branch_data[:, :64, :, :] 
        b_cls_raw = branch_data[:, 64:, :, :] 
        
        boxes.append(box_process(b_box_raw))
        
        #b_cls_prob = sigmoid(b_cls_raw)

        #classes_conf.append(b_cls_prob)
        classes_conf.append(b_cls_raw)

        #b_score = np.max(b_cls_prob, axis=1, keepdims=True)
        #scores.append(b_score)
        scores.append(np.ones_like(b_cls_raw[:, :1, :, :], dtype=np.float32))
        
    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    nboxes, nclasses, nscores = [], [], []
    for c_id in set(classes):
        inds = np.where(classes == c_id)
        b = boxes[inds]
        s = scores[inds]
        
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(np.full(len(keep), c_id))
            nscores.append(s[keep])

    if not nboxes:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def add_draw(image, boxes, scores, classes):
    '''
    orange pi에서 한글이 나오지 않아서 만듬.
    '''
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw_pil = ImageDraw.Draw(img_pil)
    
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    try:
        font = ImageFont.truetype(font_path, 20)
    except:
        font = ImageFont.load_default()

    for box, score, cl in zip(boxes, scores, classes):
        y1, x1, y2, x2 = [int(_b) for _b in box]
        print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], y1, x1, y2, x2, score))
        
        label_text = CLASSES[cl]
        full_label = f"{label_text} {score:.2f}"

        draw_pil.rectangle([y1, x1, y2, x2], outline=(255, 0, 0), width=3) 


        text_pos = (y1, x1 - 25) 
        draw_pil.text(text_pos, full_label, font=font, fill=(255, 255, 255))

    final_image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    image[:] = final_image

def setup_model(args):
    model_path = args.model_path
    if model_path.endswith('.pt') or model_path.endswith('.torchscript'):
        platform = 'pytorch'
        from py_utils.pytorch_executor import Torch_model_container
        model = Torch_model_container(args.model_path)
    elif model_path.endswith('.rknn'):
        platform = 'rknn'
        from py_utils.rknn_executor import RKNN_model_container 
        model = RKNN_model_container(args.model_path, args.target, args.device_id)
    elif model_path.endswith('onnx'):
        platform = 'onnx'
        from py_utils.onnx_executor import ONNX_model_container
        model = ONNX_model_container(args.model_path)
    else:
        assert False, "{} is not rknn/pytorch/onnx model".format(model_path)
    print('Model-{} is {} model, starting val'.format(model_path, platform))
    return model, platform

def setup_rknn(model_path, target, device_id=None, core_mask=0x0) :
    from py_utils.rknn_executor import RKNN_model_container 
    model = RKNN_model_container(model_path, target, device_id, core_mask)
    print('Model-{} is rknn model, starting val'.format(model_path))
    return model

def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False

def get_labels(txt_path):
    if not os.path.exists(txt_path): return []
    with open(txt_path, 'r') as f:
        return [list(map(float, line.split())) for line in f.readlines()]

def calculate_iou(box1, box2):
    # box = [x1, y1, x2, y2]
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_path', type=str, default='snack_yolo8s.rknn', 
                        help='model path, could be .pt or .rknn file')
    parser.add_argument('--target', type=str, default='rk3588', help='target RKNPU platform')
    parser.add_argument('--device_id', type=str, default=None, help='device id')
    parser.add_argument('--img_show', action='store_true', default=False, help='draw the result and show')
    parser.add_argument('--img_save', action='store_true', default=True, help='save the result')
    parser.add_argument('--anno_json', type=str, default='../../../datasets/COCO/annotations/instances_val2017.json', help='coco annotation path')
    parser.add_argument('--img_folder', type=str, default='../../Documents/test_img', help='img folder path')
    parser.add_argument('--coco_map_test', action='store_true', help='enable coco map test')

    args = parser.parse_args()

    model, platform = setup_model(args)

    file_list = sorted(os.listdir(args.img_folder))
    img_list = []
    for path in file_list:
        if img_check(path):
            img_list.append(path)
    co_helper = COCO_test_helper(enable_letter_box=True)

    total_time = 0
    processed_frames = 0    

    all_data_logs = []
    total_gt_count = 0
    total_tp_count = 0
    error_img_path_list = []
    iou_threshold = 0.5

    # run test
    img_list_len = len(img_list)
    print(img_list_len)

    save_path = None
    txt_file_path = None

    if args.img_save :
        save_path = os.path.join('result', 'yolov8')
        os.makedirs(save_path, exist_ok=True)

        txt_file_path = os.path.join(save_path, 'how_many_object.txt')

        if os.path.exists(txt_file_path):
            os.remove(txt_file_path)

    for i in range(img_list_len):
        img_name = img_list[i]

        img_name = img_list[i]
        img_path = os.path.join(args.img_folder, img_name)
        if not os.path.exists(img_path):
            print("{} is not found", img_name)
            continue

        img_src = cv2.imread(img_path)
        if img_src is None:
            continue

        img_basename = os.path.splitext(img_name)[0]
        label_path = f'../../Documents/test_label/{img_basename}.txt'
        gt_list = get_labels(label_path)
        total_gt_count += len(gt_list)
        matched_preds_idx = []
        img_tp = 0

        # GT를 [x1, y1, x2, y2] 픽셀 단위로 미리 변환
        processed_gts = []
        h, w, _ = img_src.shape
        for gt in gt_list:
            cls, gx, gy, gw, gh = gt
            # 정규화 xywh -> 픽셀 x1y1x2y2
            x1 = (gx - gw/2) * w
            y1 = (gy - gh/2) * h
            x2 = (gx + gw/2) * w
            y2 = (gy + gh/2) * h
            processed_gts.append({'cls': int(cls), 'box': [x1, y1, x2, y2]})

        start_tick = time.time()

        pad_color = (0,0,0)
        img = co_helper.letter_box(im= img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if platform in ['pytorch', 'onnx']:
            input_data = img.transpose((2,0,1))
            input_data = input_data.reshape(1,*input_data.shape).astype(np.float32)
            input_data = input_data/255.
        else:
            input_data = img

        outputs = model.run([input_data])
        boxes, classes, scores = add_post_process(outputs) 

        image_record = {'img_name': img_name, 'objects' : []}

        if args.img_show or args.img_save:
            print('\n\nIMG: {}'.format(img_name))
            img_p = img_src.copy()
            if boxes is not None:
                add_draw(img_p, co_helper.get_real_box(boxes), scores, classes)

            if args.img_save and boxes is not None:
                result_path = os.path.join(save_path, img_name)
                cv2.imwrite(result_path, img_p)
                print('Detection result save to {}'.format(result_path))
                        
            if args.img_show:
                cv2.imshow("full post process result", img_p)

                if cv2.waitKeyEx(1) & 0xFF == ord('q') :
                    break
        
        if txt_file_path :
            object_count = 0
            if boxes is not None :
                object_count = len(boxes)
            with open(txt_file_path, 'a', encoding='utf-8') as f:
                f.write(f"{img_name} : {object_count} objects detected\n")

        # record maps
        if args.coco_map_test is True:
            if boxes is not None:
                for i in range(boxes.shape[0]):
                    co_helper.add_single_record(image_id = int(img_name.split('.')[0]),
                                                category_id = coco_id_list[int(classes[i])],
                                                bbox = boxes[i],
                                                score = round(scores[i], 5).item()
                                                )
        end_tick = time.time()

        if boxes is not None:
            for gt in processed_gts:
                matched = False
                for p_idx, (p_box, p_cls) in enumerate(zip(boxes, classes)):
                    if p_idx in matched_preds_idx: continue
                    
                    if gt['cls'] == int(p_cls) and calculate_iou(gt['box'], p_box) >= iou_threshold:
                        img_tp += 1
                        matched = True
                        matched_preds_idx.append(p_idx)
                        image_record['objects'].append({
                            'status': 'Success (TP)',
                            'gt_id': gt['cls'], 'pred_id': int(p_cls)
                        })
                        break
                
                if not matched:
                    image_record['objects'].append({
                        'status': 'Missed (FN)',
                        'gt_id': gt['cls'], 'pred_id': None
                    })

            for p_idx, (p_box, p_cls) in enumerate(zip(boxes, classes)):
                if p_idx not in matched_preds_idx:
                    image_record['objects'].append({
                        'status': 'Extra (FP)',
                        'gt_id': None, 'pred_id': int(p_cls)
                    })
        else:
            for gt in processed_gts:
                image_record['objects'].append({'status': 'Missed (FN)', 'gt_id': gt['cls'], 'pred_id': None})

        total_tp_count += img_tp
        all_data_logs.append(image_record)
        
        pred_count = len(boxes) if boxes is not None else 0
        if pred_count > img_tp or len(gt_list) > img_tp:
            error_img_path_list.append(os.path.join(save_path, img_name))
    
        curr_time = end_tick - start_tick
        total_time += curr_time
        processed_frames += 1

        print('Inference {}/{} - Time: {:.4f}s, FPS: {:.2f}'.format(
            i+1, len(img_list), curr_time, 1/curr_time), end='\r')

    if processed_frames > 0:
        avg_fps = processed_frames / total_time
        print('\n' + '='*30)
        print('Average FPS: {:.2f}'.format(avg_fps))
        print('Average Latency: {:.4f}s'.format(total_time / processed_frames))
        print('='*30)

        recall = (total_tp_count / total_gt_count * 100) if total_gt_count > 0 else 0
        print(f"전체 정답 객체 수: {total_gt_count}개 / 검출 성공: {total_tp_count}개")
        print(f"인식률(Recall): {recall:.2f}%")

    if args.coco_map_test is True:
        pred_json = args.model_path.split('.')[-2]+ '_{}'.format(platform) +'.json'
        pred_json = pred_json.split('/')[-1]
        pred_json = os.path.join('./', pred_json)
        co_helper.export_to_json(pred_json)

        from py_utils.coco_utils import coco_eval_with_json
        coco_eval_with_json(args.anno_json, pred_json)

    # release
    model.release()