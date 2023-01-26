from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from models.utils.datasets import LoadStreams, LoadImages
from models.cmd.general import check_img_size, non_max_suppression, \
    scale_coords,  strip_optimizer, set_logging, increment_path
from models.utils.plots import plot_one_box
from models.utils.torch_utils import select_device, time_synchronized
from models.utils.torch_utils import attempt_load, Ensemble


def detect(cfg):
    save_img, source, weights, imgsz = True, cfg.source, cfg.model, cfg.imgsz
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(cfg.save_dir) / 'exp', exist_ok=False))  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    seen, dt = 0, [0.0]
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t0 = time_synchronized()
        pred = model(img, augment=False)[0]
        t1 = time_synchronized()
        dt[0] += t1 - t0

        # Apply NMS
        pred = non_max_suppression(pred, cfg.confidence, cfg.nms_iou, classes=0, agnostic=False)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            save_path = str(save_dir / Path(p).name)  # img.jpg

            s += '%gx%g ' % img.shape[2:]  # print string
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_img:  # Add bbox to image
                        plot_one_box(xyxy, im0, label='', color=(113,227,4), line_thickness=15)

            # Output
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.putText(im0, f"{len(det)}",(20, 240), cv2.FONT_HERSHEY_DUPLEX, 8, (255, 255, 255), 10)
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
                    cv2.imshow('LFANet', im0)
                    key = cv2.waitKey(1)  # 1 millisecond
                    if key == 32:
                        cv2.destroyAllWindows()
                        raise RuntimeError("Camera off")

    tm = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print(f' %.1fms inference' % tm) # total time
    print(f" The image with the result is saved in: {save_dir}")


def infer(cfg):
    print(f'[model={cfg.model}, source={cfg.source}, imgsz={cfg.imgsz}, '
          f'confidence={cfg.confidence}, nms_iou={cfg.nms_iou}, save_dir={cfg.save_dir}]')

    with torch.no_grad():   # torch init
        detect(cfg)
