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
    
    # 1. ID 리스트 생성 (키 값들을 정렬하여 리스트로 만듦)
    # 예: [0, 1, 2, ...]
    id_list = sorted(names.keys())
    
    # 2. 클래스명 리스트 생성 (ID 순서에 맞춰서 이름 추출)
    # 예: ("(주)굿푸드블루베리엔치즈", "item2", ...)
    class_list = [names[i] for i in id_list]
    
    return tuple(class_list), id_list

# 실제 적용
yaml_file = "data.yaml"  # 파일 경로를 입력하세요
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

    # filter according to threshold
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

def add_post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    
    # 1701 채널 = 64(DFL 좌표) + 1637(클래스 점수)
    # input_data는 [ (1,1701,60,60), (1,1701,30,30), (1,1701,15,15) ] 형태입니다.
    for branch_data in input_data:
        # 1. 하나의 텐서에서 좌표와 클래스 점수를 잘라냅니다.
        # 좌표 데이터 (앞 64채널)
        b_box_raw = branch_data[:, :64, :, :] 
        # 클래스 데이터 (64번 채널부터 끝까지)
        b_cls_raw = branch_data[:, 64:, :, :] 
        
        # 2. 박스 좌표 복구 (기존 box_process 사용)
        # 480 해상도에 맞춰 60, 30, 15 그리드를 자동으로 계산합니다.
        boxes.append(box_process(b_box_raw))
        
        # 3. 클래스 정보 및 스코어용 더미 데이터 추가
        classes_conf.append(b_cls_raw)
        # filter_boxes 함수 구조를 유지하기 위해 1로 채워진 스코어 텐서를 생성합니다.
        scores.append(np.ones_like(b_cls_raw[:, :1, :, :], dtype=np.float32))

    # --- 여기서부터는 기존 sp_flatten 로직을 그대로 사용합니다 ---
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

    # filter_boxes를 통해 threshold 적용
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # NMS (scalar index 에러 방지 버전)
    nboxes, nclasses, nscores = [], [], []
    for c_id in set(classes):
        inds = np.where(classes == c_id)
        b = boxes[inds]
        s = scores[inds]
        
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            # 에러 방지: c 대신 np.full 사용
            nclasses.append(np.full(len(keep), c_id))
            nscores.append(s[keep])

    if not nboxes: # 리스트가 비어있는지 체크
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
    # 1. OpenCV(BGR)를 PIL(RGB)로 변환
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw_pil = ImageDraw.Draw(img_pil)
    
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    try:
        font = ImageFont.truetype(font_path, 20)
    except:
        font = ImageFont.load_default()

    for box, score, cl in zip(boxes, scores, classes):
        y1, x1, y2, x2 = [int(_b) for _b in box]
        
        label_text = CLASSES[cl]
        full_label = f"{label_text} {score:.2f}"
        
        # 3. 박스 그리기 (PIL에서 직접 수행)
        draw_pil.rectangle([y1, x1, y2, x2], outline=(255, 0, 0), width=3) # 기존 변수명 기준

        # 4. 한글 텍스트 배경 및 글자 그리기
        # 텍스트가 박스 밖으로 나가지 않게 y1 좌표 위쪽에 그림
        text_pos = (y1, x1 - 25) 
        draw_pil.text(text_pos, full_label, font=font, fill=(255, 255, 255)) # 흰색 글씨

    # 5. 최종본을 다시 OpenCV(BGR)로 변환
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

def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False

import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    # 1. model_path: 필수(required)를 해제하고 기본 경로 설정
    parser.add_argument('--model_path', type=str, default='snack_yolo8s.rknn', 
                        help='model path, could be .pt or .rknn file')
    
    # 2. target: 기본값을 rk3588로 변경
    parser.add_argument('--target', type=str, default='rk3588', help='target RKNPU platform')
    parser.add_argument('--device_id', type=str, default=None, help='device id')
    
    parser.add_argument('--img_show', action='store_true', default=False, help='draw the result and show')
    
    # 3. img_save: 기본값을 True로 변경 (인자 없이도 저장되도록 함)
    parser.add_argument('--img_save', action='store_true', default=True, help='save the result')

    # data params
    parser.add_argument('--anno_json', type=str, default='../../../datasets/COCO/annotations/instances_val2017.json', help='coco annotation path')
    
    # 4. img_folder: 기본 경로를 지정하신 위치로 변경
    parser.add_argument('--img_folder', type=str, default='../../Documents/test_img', help='img folder path')
    
    parser.add_argument('--coco_map_test', action='store_true', help='enable coco map test')

    args = parser.parse_args()

    # init model
    model, platform = setup_model(args)

    file_list = sorted(os.listdir(args.img_folder))
    img_list = []
    for path in file_list:
        if img_check(path):
            img_list.append(path)
    co_helper = COCO_test_helper(enable_letter_box=True)

    total_time = 0
    processed_frames = 0    

    # run test
    img_list_len = len(img_list)
    print(img_list_len)

    save_path = None
    txt_file_path = None

    if args.img_save :
        # 1. 저장할 경로 설정 (result/yolov8)
        save_path = os.path.join('result', 'yolov8')

        # 2. 폴더가 없으면 생성 (exist_ok=True는 이미 폴더가 있어도 에러를 내지 않습니다)
        os.makedirs(save_path, exist_ok=True)

        txt_file_path = os.path.join(save_path, 'how_many_object.txt')

        if os.path.exists(txt_file_path):
            os.remove(txt_file_path)

    for i in range(img_list_len):
        # \033[K 는 현재 커서 위치부터 줄 끝까지 지우라는 명령어입니다.
        #print(f'\r\033[Kinfer {i+1}/{img_list_len}', end='')

        img_name = img_list[i]
        img_path = os.path.join(args.img_folder, img_name)
        if not os.path.exists(img_path):
            print("{} is not found", img_name)
            continue

        img_src = cv2.imread(img_path)
        if img_src is None:
            continue

        '''
        # using for test input dumped by C.demo
        img_src = np.fromfile('./input_b/demo_c_input_hwc_rgb.txt', dtype=np.uint8).reshape(640,640,3)
        img_src = cv2.cvtColor(img_src, cv2.COLOR_RGB2BGR)
        '''

        start_tick = time.time()

        # Due to rga init with (0,0,0), we using pad_color (0,0,0) instead of (114, 114, 114)
        pad_color = (0,0,0)
        img = co_helper.letter_box(im= img_src.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # preprocee if not rknn model
        if platform in ['pytorch', 'onnx']:
            input_data = img.transpose((2,0,1))
            input_data = input_data.reshape(1,*input_data.shape).astype(np.float32)
            input_data = input_data/255.
        else:
            input_data = img

        outputs = model.run([input_data])
        boxes, classes, scores = add_post_process(outputs) 

        if args.img_show or args.img_save:
            print('\n\nIMG: {}'.format(img_name))
            img_p = img_src.copy()
            if boxes is not None:
                add_draw(img_p, co_helper.get_real_box(boxes), scores, classes)
                #h, w, _ = img_src.shape
                #scale_w = w / IMG_SIZE[0]
                #scale_h = h / IMG_SIZE[1]
                #boxes[:, [0, 2]] *= scale_w
                #boxes[:, [1, 3]] *= scale_h
                
                #draw(img_p, boxes, scores, classes)

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
    
        curr_time = end_tick - start_tick
        total_time += curr_time
        processed_frames += 1

        # 개별 프레임 FPS 출력 (선택 사항)
        print('Inference {}/{} - Time: {:.4f}s, FPS: {:.2f}'.format(
            i+1, len(img_list), curr_time, 1/curr_time), end='\r')

    # 루프 종료 후 평균 FPS 출력
    if processed_frames > 0:
        avg_fps = processed_frames / total_time
        print('\n' + '='*30)
        print('Average FPS: {:.2f}'.format(avg_fps))
        print('Average Latency: {:.4f}s'.format(total_time / processed_frames))
        print('='*30)

    # calculate maps
    if args.coco_map_test is True:
        pred_json = args.model_path.split('.')[-2]+ '_{}'.format(platform) +'.json'
        pred_json = pred_json.split('/')[-1]
        pred_json = os.path.join('./', pred_json)
        co_helper.export_to_json(pred_json)

        from py_utils.coco_utils import coco_eval_with_json
        coco_eval_with_json(args.anno_json, pred_json)

    # release
    model.release()