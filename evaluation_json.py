from yolact import Yolact
from utils.augmentations import FastBaseTransform
from utils import timer
from layers.output_utils import postprocess, undo_image_transformation

from data import cfg, set_cfg

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import argparse
import time
import random
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import cv2

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model',
                        default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=5, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        help='Whether compute NMS cross-class or per-class.')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='In quantitative mode, the file to save detections before calculating mAP.')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='If display not set, this resumes mAP calculations from the ap_data_file.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
    parser.add_argument('--web_det_path', default='web/dets/', type=str,
                        help='If output_web_json is set, this is the path to dump detections into.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                        help='Equivalent to running display mode but without displaying an image.')
    parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                        help='Do not sort images by hashed image ID.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--video_multiframe', default=1, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
    parser.add_argument('--score_threshold', default=0, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
    parser.add_argument('--display_fps', default=False, dest='display_fps', action='store_true',
                        help='When displaying / saving video, draw the FPS on the frame')
    parser.add_argument('--emulate_playback', default=False, dest='emulate_playback', action='store_true',
                        help='When saving a video, emulate the framerate that you\'d get running in real-time mode.')

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False, shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False, display_fps=False,
                        emulate_playback=False)

    global args

    args = parser.parse_args(argv)

    args.score_threshold=0.4 
    args.top_k=1000
    args.display_text = False
    args.display_scores = False
    if args.output_web_json:
        args.output_coco_json = True
    
    if args.seed is not None:
        random.seed(args.seed)

iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {} # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})

def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape
    
    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, w, h, visualize_lincomb = args.display_lincomb,
                                        crop_masks        = args.crop,
                                        score_threshold   = args.score_threshold)
        cfg.rescore_bbox = save

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:args.top_k]
        
        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

        scores = t[1]
        if isinstance(scores, list):
            box_scores = list(scores[0].cpu().numpy().astype(float))
            mask_scores = list(scores[1].cpu().numpy().astype(float))
        else:
            scores = list(scores.cpu().numpy().astype(float))
            box_scores = scores
            mask_scores = scores

        masks = t[3][idx].view(-1, h, w).cpu().numpy()
        from pycocotools import mask
        from skimage import measure
        all_boxes = []
        all_segmentations = []
        for i in range(masks.shape[0]):
            # Make sure that the bounding box actually makes sense and a mask was produced
            if (boxes[i, 3] - boxes[i, 1]) * (boxes[i, 2] - boxes[i, 0]) > 0:
                # rle = pycocotools.mask.encode(np.asfortranarray(masks[i,:,:].astype(np.uint8)))
                # rle['counts'] = rle['counts'].decode('ascii') # json.dump doesn't like bytes strings
                ground_truth_binary_mask = masks[i,:,:].astype(np.uint8)
                fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
                encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
                ground_truth_area = mask.area(encoded_ground_truth)
                ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
                contours = measure.find_contours(ground_truth_binary_mask, 0.5)
                for contour in contours:
                    contour = np.flip(contour, axis=1)
                    segmentation = contour.ravel().tolist()
                # a=classes[i], boxes[i,:], box_scores[i]
                all_boxes.append(boxes[i,:])
                all_segmentations.append(segmentation)
    return all_boxes, all_segmentations, scores, classes


def prep_benchmark(dets_out, h, w):
    with timer.env('Postprocess'):
        t = postprocess(dets_out, w, h, crop_masks=args.crop, score_threshold=args.score_threshold)

    with timer.env('Copy'):
        classes, scores, boxes, masks = [x[:args.top_k] for x in t]
        if isinstance(scores, list):
            box_scores = scores[0].cpu().numpy()
            mask_scores = scores[1].cpu().numpy()
        else:
            scores = scores.cpu().numpy()
        classes = classes.cpu().numpy()
        boxes = boxes.cpu().numpy()
        masks = masks.cpu().numpy()
    
    with timer.env('Sync'):
        # Just in case
        torch.cuda.synchronize()

import json
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def save_json(dict_final, name):        
    f = open(name,"w+")
    f.write(json.dumps(dict_final, cls=NpEncoder))
    f.close()
    
def evalimage_cv2(net:Yolact, image:np.array):
    net.detect.use_fast_nms = args.fast_nms
    net.detect.use_cross_class_nms = args.cross_class_nms
    cfg.mask_proto_debug = args.mask_proto_debug
    try:
        frame = torch.from_numpy(image).cuda().float()
    except:
        return
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)

    with torch.no_grad():
        start = time.time()
        preds = net(batch)
        t1 = time.time()
        print("Time for image: ", t1 - start)

    all_boxes, all_segmentation, scores, classes = prep_display(preds, frame, None, None, undo_transform=False)

    return all_boxes, all_segmentation, scores, classes


class EvaluationYOLACT():

    def __init__(self, model_weight_path:str, name_config:str, top_k = 1000) -> None:
        parse_args()
        
        args.top_k = top_k
        args.trained_model = model_weight_path
        args.score_threshold = 0.4

        print(args)
        args.config = name_config

        if args.config is not None:
            set_cfg(args.config)
        
        with torch.no_grad():
            if not os.path.exists('results'):
                os.makedirs('results')

            if args.cuda:
                cudnn.fastest = True
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            else:
                torch.set_default_tensor_type('torch.FloatTensor')      

            print('Loading model...', end='')
            self.net = Yolact()
            self.net.load_weights(model_weight_path)
            self.net.eval()
        
    def eval(self, image):

        all_boxes, all_segmentation, scores, classes = evalimage_cv2(self.net, image)
        
        if args.cuda:
            self.net = self.net.cuda()

        return all_boxes, all_segmentation, scores, classes


#hyperparameters for evaluation
path = '/home/cvar-admin/Documents/yolact/data/copilot_1024_1376/images/test2023/'
weights = './weights/yolact_base_54_80000.pth'
name_config = 'yolact_base_config' #if resnet 101 use yolact_base_config elif resnet 50 use yolact_resnet50_config elif yolact plus use yolact_plus_base_config
#---------------------------------------------


yolact=EvaluationYOLACT(weights, name_config)
final_json = []

for filename in os.listdir(path):
    
    print(filename)
    image = cv2.imread(path+filename)
    bboxes, segmentations, scores, classes = yolact.eval(image)

    for i in range(len(bboxes)):
        ann = {
                'bbox': bboxes[i],
                'segmentation': [segmentations[i]],
                'category_id': classes[i],
                'score': scores[i]}
        
        final_json.append({'filename': filename, 'annotations': ann})
    

save_json(final_json, 'yolact_metrics.json')
