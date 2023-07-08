from models.drive.train_test import train
import argparse

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='transfer learning weights path')
    parser.add_argument('--cfg', type=str, default='config/baseline.yaml', help='model.yaml path')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[640, 640], help='train size (pixels)')
    parser.add_argument('--epochs', type=int, default=100, help='number of iterations')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--workers', type=int, default=8, help='number of dataloader workers')
    parser.add_argument('--dataset', type=str, default='config/data.yaml', help='config.yaml path')
    parser.add_argument('--cache', default=False, help='cache images for faster training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--save-dir', default='runs/train', help='save results path(s)')
    cfg = parser.parse_args()
    return cfg

if __name__ == '__main__':
    cfg = config()
    train(cfg)
