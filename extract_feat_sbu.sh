#!/bin/bash 

DATA_ROOT=/z/dat/VLP/dat/SBU

python tools/extract_features_luowei.py \
    --output-dir $DATA_ROOT/region_feat_gvd_wo_bgd/feat_cls_1000_float16 \
    --det-output-file-prefix $DATA_ROOT/region_feat_gvd_wo_bgd/raw_bbox/sbu_detection_vg_100dets_vlp_checkpoint_trainval \
    --featcls-output-file-prefix $DATA_ROOT/region_feat_gvd_wo_bgd/feat_cls_1000_float16/sbu_detection_vg_100dets_vlp_checkpoint_trainval \
    --max_bboxes 100 --min_bboxes 100 \
    --cfg /z/home/luozhou/subsystem/detectron-xinlei/configs/visual_genome_trainval/e2e_faster_rcnn_X-101-64x4d-FPN_2x.yaml \
    --wts /z/home/luozhou/subsystem/detectron-xinlei/checkpoint/visual_genome_train+visual_genome_val/e2e_faster_rcnn_X-101-64x4d-FPN_2x/RNG_SEED#1989_FAST_RCNN.MLP_HEAD_DIM#2048_FPN.DIM#512/model_final.pkl \
    --split $1 --dataset SBU \
    $DATA_ROOT/images \
    | tee log/log_extract_features_vg_100dets_sbu_"$1"

    # --output-dir /home/luozhou/dat/VLP/SBU/region_feat_gvd_wo_bgd/feat_cls_1000_float16 \
    # /home/luozhou/dat/VLP/SBU/images \
    # --cfg /z/home/luozhou/subsystem/detectron-xinlei/configs/visual_genome/e2e_faster_rcnn_X-101-64x4d-FPN_1x_MLP_2048_FPN_512.yaml \
    # --wts /z/home/luozhou/subsystem/detectron-xinlei/checkpoint/visual_genome_train/FAST_RCNN_MLP_DIM2048_FPN_DIM512.pkl \
