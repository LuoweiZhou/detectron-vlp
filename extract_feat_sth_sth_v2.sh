#!/bin/bash 

python tools/extract_features_vg_sth_sth_v2_xinlei.py \
    --output-dir /z/dat/sth_sth_v2/feats/detectron_bbox \
    --max_bboxes 100 --min_bboxes 100 \
    --cfg /z/home/luozhou/subsystem/detectron-xinlei/configs/visual_genome_trainval/e2e_faster_rcnn_X-101-64x4d-FPN_2x.yaml \
    --wts /z/home/luozhou/subsystem/detectron-xinlei/checkpoint/visual_genome_train+visual_genome_val/e2e_faster_rcnn_X-101-64x4d-FPN_2x/RNG_SEED#1989_FAST_RCNN.MLP_HEAD_DIM#2048_FPN.DIM#512/model_final.pkl \
    --split $1 \
    /z/dat/sth_sth_v2/frames | tee log/log_extract_features_vg_100dets_sth_sth_v2_"$1"

    # --cfg /z/home/luozhou/subsystem/detectron-xinlei/configs/visual_genome/e2e_faster_rcnn_X-101-64x4d-FPN_1x_MLP_2048_FPN_512.yaml \
    # --wts /z/home/luozhou/subsystem/detectron-xinlei/checkpoint/visual_genome_train/FAST_RCNN_MLP_DIM2048_FPN_DIM512.pkl \
