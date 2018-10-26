module load cuda/9.0 cudnn/v7.0-cuda.9.0 NCCL/2.2.13-cuda.9.0 openmpi/3.0.0/gcc.5.4.0

id_prefix='/private/home/luoweizhou/subsystem/detectron-xinlei/tools/tmp_splits'
split_file='anet_split_'"$1"
list_of_ids=$id_prefix/$split_file'.json'

# python tools/extract_features_vg_anet.py --output-dir /checkpoint02/luoweizhou/dat/anet/fc6_feat --det-output-file /checkpoint02/luoweizhou/dat/anet/anet_detection_vg_fc6_feat"$1".h5 --max_bboxes 20 --min_bboxes 20 --list_of_ids $list_of_ids /checkpoint02/luoweizhou/dat/anet/frames_10frm | tee log/log_extract_features_vg_20rois_$split_file

# python tools/extract_features_vg_anet.py --output-dir /checkpoint02/luoweizhou/dat/anet/fc6_feat_50rois --det-output-file /checkpoint02/luoweizhou/dat/anet/anet_detection_vg_fc6_feat_50rois"$1".h5 --max_bboxes 50 --min_bboxes 50 --list_of_ids $list_of_ids /checkpoint02/luoweizhou/dat/anet/frames_10frm | tee log/log_extract_features_vg_50rois_$split_file

python tools/extract_features_vg_anet.py --output-dir /checkpoint02/luoweizhou/dat/anet/fc6_feat_100rois --det-output-file /checkpoint02/luoweizhou/dat/anet/anet_detection_vg_fc6_feat_100rois"$1".h5 --max_bboxes 100 --min_bboxes 100 --list_of_ids $list_of_ids /checkpoint02/luoweizhou/dat/anet/frames_10frm | tee log/log_extract_features_vg_100rois_$split_file

# python tools/extract_features_vg_anet.py --output-dir /checkpoint02/luoweizhou/dat/anet/fc6_feat_100rois_png --det-output-file /checkpoint02/luoweizhou/dat/anet/anet_detection_vg_fc6_feat_100rois_png"$1".h5 --max_bboxes 100 --min_bboxes 100 --list_of_ids $list_of_ids --image-ext '.png' /checkpoint02/luoweizhou/dat/anet/frames_10frm_png | tee log/log_extract_features_vg_100rois_png_$split_file

# python tools/extract_features_vg_anet.py --output-dir /checkpoint02/luoweizhou/dat/anet/fc6_feat_100rois_noresize --det-output-file /checkpoint02/luoweizhou/dat/anet/anet_detection_vg_fc6_feat_100rois_noresize"$1".h5 --max_bboxes 100 --min_bboxes 100 --list_of_ids $list_of_ids /checkpoint02/luoweizhou/dat/anet/frames_10frm_noresize | tee log/log_extract_features_vg_100rois_noresize_$split_file
