import cv2
import numpy as np
import time
from py_utils.coco_utils import COCO_test_helper
from yolo8_rknn import setup_rknn, sigmoid_post_process, pack_add_draw, post_process, pack_draw

IMG_SIZE = (640, 640)

P_SIZE = (1920, 1080)

def n_post_process(output_data, co_helper):
    boxes, classes, scores = sigmoid_post_process(output_data)

    if boxes is not None:
        co_helper.get_real_box(boxes)
        output = np.column_stack((boxes, scores, classes))
        return output, boxes, scores, classes 

    return None, None, None, None  

def main():
    target = 'rk3588'
    model_path = 'model_rknn/lotte_sand_only.rknn'
    
    model = setup_rknn(model_path, target, core_mask=0x0)
    
    cap = cv2.VideoCapture(20, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print(">>> Python USB Webcam Inference Started...")
    
    co_helper = COCO_test_helper(enable_letter_box=True)
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            frame_count += 1

            img_input = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
            
            img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)

            input_data = co_helper.letter_box(
                im=img_input.copy(), 
                new_shape=(IMG_SIZE[1], IMG_SIZE[0]), 
                pad_color=(0, 0, 0)
            )

            outputs = model.run([input_data])

            res, boxes, scores, classes = n_post_process(outputs, co_helper)

            if boxes is not None:
                frame_draw = pack_draw(img_input, boxes, scores, classes)
            else:
                frame_draw = img_input

            frame_draw = cv2.cvtColor(frame_draw, cv2.COLOR_RGB2BGR)
            #frame_draw = cv2.resize(frame_draw, P_SIZE, interpolation=cv2.INTER_LINEAR)

            cv2.imshow("RKNN USB Webcam Test", frame_draw)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopped by User.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    main()