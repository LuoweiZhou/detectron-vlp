#!/bin/bash 

module load cuda/9.0 cudnn/v7.0-cuda.9.0 NCCL/2.2.13-cuda.9.0 openmpi/3.0.0/gcc.5.4.0

# python tools/extract_features_vg_flickr30k_xinlei.py --output-dir /checkpoint02/luoweizhou/dat/flickr30k/flickr30k_detection_vg_X-101-64x4d-FPN_2x_feature_det_coordinates --det-output-file /checkpoint02/luoweizhou/dat/flickr30k/flickr30k_detection_vg_X-101-64x4d-FPN_2x_feat_map_100prop_box_only_det_coordinates.h5 --max_bboxes 100 --min_bboxes 100 /checkpoint02/luoweizhou/dat/flickr30k/images | tee log/log_extract_features_vg_flickr30k_100dets

python tools/extract_features_vg_flickr30k_xinlei.py --output-dir /checkpoint02/luoweizhou/dat/flickr30k/flickr30k_detection_vg_X-101-64x4d-FPN_2x_feature_det_coordinates-nms_0.5 --det-output-file /checkpoint02/luoweizhou/dat/flickr30k/flickr30k_detection_vg_X-101-64x4d-FPN_2x_feat_map_100prop_box_only_det_coordinates-nms_0.5.h5 --cfg /private/home/luoweizhou/subsystem/detectron-xinlei/configs/visual_genome_trainval/e2e_faster_rcnn_X-101-64x4d-FPN_2x-NMS_0.5.yaml --max_bboxes 100 --min_bboxes 100 /checkpoint02/luoweizhou/dat/flickr30k/images | tee log/log_extract_features_vg_flickr30k_100dets_nms0.5
