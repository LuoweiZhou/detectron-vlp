#!/bin/bash 

DATA_ROOT=tmp/release/anet

python tools/extract_features_gvd_anet.py \
  --output-dir $DATA_ROOT/fc6_feat_100rois \
  --det-output-file $DATA_ROOT/anet_detection_vg_fc6_feat_100rois.h5 \
  --max_bboxes 100 --min_bboxes 100 \
  --list_of_ids $DATA_ROOT/split_ids_anet_entities.json $DATA_ROOT/frames_1_10frm \
  | tee log/log_extract_features_vg_100rois_gvd_anet
