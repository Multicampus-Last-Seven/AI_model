import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import os
import paho.mqtt.publish as publish

def detect(file_name):

    # Initialize
    source =  file_name  # file/folder, 0 for webcam
    cam_serial_num = file_name[-16:-4]
    weights='best.pt'
    view_img=False
    save_txt=False
    imgsz= 640 

    opt_name='detect'
    opt_project='static'

    opt_exist_ok=True
    opt_hide_conf=False
    opt_hide_labels=False
    opt_iou_thres=0.45
    opt_line_thickness=3
    opt_nosave=False
    opt_save_conf=False
    opt_save_crop=False
    opt_update=False
    opt_device = ''
    opt_augment =False
    opt_conf_thres = 0.25
    opt_classes =None
    opt_agnostic_nms  = False

    #화재발생 ( default - false)
    ai_fire_alarm  = False
    ai_fire_count =  0
    #화재 발생 label
    fire_labels =['fire','fire-n','smoke']

    #detection_label
    detection_labels =[]


    save_img = not opt_nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(opt_project) / opt_name, exist_ok=True)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt_device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    try:
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt_augment)[0]

            # Apply NMS

            pred = non_max_suppression(pred, opt_conf_thres, opt_iou_thres, classes=opt_classes, agnostic=opt_agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if opt_save_crop else im0  # for opt_save_crop

                #detection_labels =[]
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        #if save_txt:  # Write to file
                        #    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        #    line = (cls, *xywh, conf) if opt_save_conf else (cls, *xywh)  # label format
                        #    with open(txt_path + '.txt', 'a') as f:
                        #        #f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        #        f.write(f'{ names[int(cls)] }  : {conf :.2f} \n' )

                        if save_img or opt_save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if opt_hide_labels else (names[c] if opt_hide_conf else f'{names[c]} {conf:.2f}')
                            plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt_line_thickness)
                            if opt_save_crop:
                                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                        # retrun labels
                        detection_labels.append( label )
                        #---------------------
                        # < 화재판단 >
                        # 1. 화재 발생시 체크 : fire 가 0.5 이상인 경우 화재
                        # 2. 화재 발생 경보가  3회 이상
                        #---------------------
                        if names[int(cls)] in fire_labels and  conf >= 0.5 :
                            print(f'> 화재  label : { label }')
                            #print(' ai_fire_count' ,ai_fire_count)
                            ai_fire_count += 1
                            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                            result, imgencode = cv2.imencode('.jpg', im0, encode_param)
                            data = numpy.array(imgencode)
                            byteData = bytearray(data)
                            publish.single(f'mydata/stream/alarm/{cam_serial_num}', byteData, hostname="15.165.185.201")
                            publish.single(f'mydata/stream/alarm/motor', "open", hostname="15.165.185.201")
                            if ai_fire_count > 3 :
                                ai_fire_alarm = True
                                ai_fire_count = 0


                if save_txt:  # Write to file
                    with open(txt_path + '.txt', 'w') as f:
                        f.write('\n'.join(detection_labels))

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
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

                # ai_fire_alarm
                #if ai_fire_alarm :


            if save_txt or save_img:
                s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                #print(f"Results saved to {save_dir}{s}")
            #print(f'Detection Done.ai_fire_alarm={ai_fire_alarm} , detection_labels = {detection_labels}')

    except Exception as e:
        print(e)
    finally :
        return ai_fire_alarm, detection_labels


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    opt = parser.parse_args()
    #print(opt)
    #check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    detect(opt.source)
#-----------------------------------------------------------------
# 1) 파일 실행 : detect.py --source  static/fire_img/fire0.jpg
# 2) 웹캠 실행 : detect.py --source 0
# 3) Webserver 호출 :   detect.detect()
#-----------------------------------------------------------------
