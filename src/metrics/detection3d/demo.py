#!/usr/bin/env python
# coding: utf-8

"""
This script allows you to obtain gt instance and prediction instance matches for the 3D mAP model evaluation. At the end, you can evaluate the mean average precision of your model based on the IoU metric. To do the evaluation, set evaluate to True (default).
"""

import time
import os, sys
import argparse
import numpy as np
import h5py

from vol3d_eval import VOL3Deval
from vol3d_util import seg_iou3d_sorted,heatmap_to_score,readh5


##### 1. I/O
def get_args():
    parser = argparse.ArgumentParser(description='Evaluate the mean average precision score (mAP) of 3D segmentation volumes')
    parser.add_argument('-gt','--gt-seg', type=str, default='~/my_ndarray.h5',
                       help='path to ground truth segmentation result')

    parser.add_argument('-p','--predict-seg', type=str, default='~/my_ndarray.h5',
                       help='path to predicted instance segmentation result')
    # either input the pre-compute prediction score
    parser.add_argument('-ps','--predict-score', type=str, default='',
                       help='path to confidence score for each prediction')
    # or avg input affinity/heatmap prediction
    parser.add_argument('-ph','--predict-heatmap', type=str, default='',
                       help='path to heatmap for all predictions')
    parser.add_argument('-phc','--predict-heatmap-channel', type=int, default=-1,
                       help='heatmap channel to use')
    parser.add_argument('-th','--threshold', type=str, default='5e3, 1.5e4',
                       help='get threshold for volume range [possible to have more than 4 ranges, c.f. cocoapi]')

    parser.add_argument('-o','--output-name', type=str, default='map_output',
                       help='output name prefix')
    parser.add_argument('-dt','--do-txt', type=int, default=1,
                       help='output txt for iou results')
    parser.add_argument('-de','--do-eval', type=int, default=1,
                       help='do evaluation')
    args = parser.parse_args()
    
    return args



def load_data(args):
    # load data arguments
    pred_seg = readh5(args.predict_seg)
    gt_seg = readh5(args.gt_seg)

    # check shape match
    sz_gt = np.array(gt_seg.shape)
    sz_pred = pred_seg.shape
    if np.abs((sz_gt-sz_pred)).max()>0:
        print('Warning: size mismatch. gt: ',sz_gt,', pred: ',sz_pred)
    sz = np.minimum(sz_gt,sz_pred)
    pred_seg = pred_seg[:sz[0],:sz[1],:sz[2]]
    gt_seg = gt_seg[:sz[0],:sz[1],:sz[2]]

    if args.predict_score != '':
        # Nx2: pred_id, pred_sc
        pred_score = readh5(args.predict_score)
    elif args.predict_heatmap!='':
        pred_heatmap = readh5(args.predict_heatmap)
        r_id, r_score, _ = heatmap_to_score(pred_seg, pred_heatmap, args.predict_heatmap_channel)
        pred_score = np.vstack([r_id, r_score]).T 
    else: # default: sort by size
        ui,uc = np.unique(pred_seg,return_counts=True)
        uc = uc[ui>0]
        ui = ui[ui>0]
        pred_score = np.ones([len(ui),2],int)
        pred_score[:,0] = ui
        pred_score[:,1] = uc

    thres = np.fromstring(args.threshold, sep = ",")
    areaRng = np.zeros((len(thres)+2,2),np.int64)
    areaRng[0,1] = 1e10
    areaRng[-1,1] = 1e10
    areaRng[2:,0] = thres
    areaRng[1:-1,1] = thres
    return gt_seg, pred_seg, pred_score, areaRng

def main():
    """ 
    Convert the grount truth segmentation and the corresponding predictions to a coco dataset
    to evaluate this dataset. The 3D volume is comparable to a video-type dataset and will therefore
    be converted as a video instance segmentation 
    input:
    output: coco_result_vid.json : This file will be written to your current directory and contains
                                    the metadata about the dataset. 
    """
    ## 1. Load data
    start_time = int(round(time.time() * 1000))
    print('\t1. Load data')
    args = get_args()
    gt_seg, pred_seg, pred_score, areaRng = load_data(args)
    
    ## 2. create complete mapping of ids for gt and pred:
    print('\t2. Compute IoU')
    result_p, result_fn, pred_score_sorted = seg_iou3d_sorted(pred_seg, gt_seg, pred_score, areaRng)
    
    stop_time = int(round(time.time() * 1000))
    print('\t-RUNTIME:\t{} [sec]\n'.format((stop_time-start_time)/1000) )

    ## 3. Evaluation script for 3D instance segmentation
    if args.output_name=='':
        args.output_name = args.predict_seg[:args.predict_seg.rfind('.')] 
    v3dEval = VOL3Deval(result_p, result_fn, pred_score_sorted, output_name=args.output_name)
    if args.do_txt > 0:
        v3dEval.save_match_p()
        v3dEval.save_match_fn()
    if args.do_eval > 0:
        print('start evaluation')        
        #Evaluation
        #v3dEval.params.areaRng = [[0, 1e10], [0, 1e5], [1e5, 5e5], [5e5, 1e10]]
        v3dEval.params.areaRng = areaRng
        v3dEval.accumulate()
        v3dEval.summarize()
        
if __name__ == '__main__':
    main()

