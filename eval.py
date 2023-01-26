from models.drive.evaluat import val
import argparse

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/cmd/LFANet.pt', help='model path')
    parser.add_argument('--batch-size', type=int, default=8, help='size of each image batch')
    parser.add_argument('--imgsz', type=int, default=640, help='eval image size (pixels)')
    parser.add_argument('--dataset', type=str, default='config/data.yaml', help='config.yaml path')
    parser.add_argument('--confidence', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--nms_iou', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--save-dir', default='runs/val', help='save results path(s)')
    cfg = parser.parse_args()
    return cfg

if __name__ == '__main__':
    cfg = config()
    val(cfg)
