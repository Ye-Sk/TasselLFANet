from models.drive.predict import infer
import argparse

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/cmd/LFANet.pt', help='model path')
    parser.add_argument('--source', type=str, default='config/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size h,w')
    parser.add_argument('--confidence', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--nms-iou', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--save-dir', default='runs/infer', help='save results path(s)')
    cfg = parser.parse_args()
    return cfg

if __name__ == '__main__':
    cfg = config()
    infer(cfg)
