# for times in 1 2; do
#     ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/coco/rc_R-101-FPN_${times}x.yaml
#     ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/coco/rc_R-50-FPN_${times}x.yaml
#     ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/coco/rc_R-50-C4_${times}x.yaml
#     ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/coco/rc_X-101-32x8d-FPN_${times}x.yaml
#     ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/coco/rc_X-101-64x4d-FPN_${times}x.yaml
# done

# for data in ade coco; do
#     ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/${data}/rc_R-50-mem-FPN_1x.yaml MEM.AT_MIN False TRAIN.ASPECT_GROUPING False
#     ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/${data}/rc_R-50-mem-C4_1x.yaml
# done

# for fc_f in 2048 4096; do
#     for c in 256 512; do
#         for in_act in none; do
#             for pool in 7 14; do
#                 for ct_l in 1 3 5; do
#                     ~/devfair/starter.sh 8 1 16 learnfair vqa train_net --cfg configs/ade/rc_R-50-mem-C4_1x.yaml MEM.CT res MEM.CT_L $ct_l MEM.IN_ACT $in_act MEM.C $c MEM.FC_C $fc_f FAST_RCNN.ROI_XFORM_RESOLUTION $pool
#                     ~/devfair/starter.sh 8 1 16 learnfair vqa train_net --cfg configs/ade/rc_R-50-mem-FPN_1x.yaml MEM.CT res MEM.CT_L $ct_l MEM.IN_ACT $in_act MEM.C $c MEM.FC_C $fc_f FAST_RCNN.ROI_XFORM_RESOLUTION $pool
#                     ~/devfair/starter.sh 8 1 16 learnfair vqa train_net --cfg configs/coco/rc_R-50-mem-C4_1x.yaml MEM.CT res MEM.CT_L $ct_l MEM.IN_ACT $in_act MEM.C $c MEM.FC_C $fc_f FAST_RCNN.ROI_XFORM_RESOLUTION $pool
#                     ~/devfair/starter.sh 8 1 16 learnfair vqa train_net --cfg configs/coco/rc_R-50-mem-FPN_1x.yaml MEM.CT res MEM.CT_L $ct_l MEM.IN_ACT $in_act MEM.C $c MEM.FC_C $fc_f FAST_RCNN.ROI_XFORM_RESOLUTION $pool
#                 done
#             done
#         done
#     done
# done

# for rng_seed in 3; do
#     for attr in 0.1 0.3 0.7 0.9; do
#         for times in 1 2; do
#             ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/visual_genome_trainval/e2e_faster_rcnn_X-101-64x4d-FPN_${times}x.yaml RNG_SEED $rng_seed MODEL.LOSS_ATTR $attr 
#             ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/visual_genome_trainval/e2e_faster_rcnn_X-101-64x4d-FPN_${times}x.yaml RNG_SEED $rng_seed MODEL.LOSS_ATTR $attr
#         done
#     done
# done

#MODEL.CLS_EMBED False

for rng_seed in 3 227 1989; do
    for times in 1 2; do
        ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/visual_genome_trainval/e2e_faster_rcnn_R-101-C4_${times}x.yaml RNG_SEED $rng_seed FAST_RCNN.MLP_HEAD_DIM 2048 FPN.DIM 512
        ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/visual_genome_trainval/e2e_faster_rcnn_R-50-FPN_${times}x.yaml RNG_SEED $rng_seed FAST_RCNN.MLP_HEAD_DIM 2048 FPN.DIM 512
        ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/visual_genome_trainval/e2e_faster_rcnn_R-101-FPN_${times}x.yaml RNG_SEED $rng_seed FAST_RCNN.MLP_HEAD_DIM 2048 FPN.DIM 512
        ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/visual_genome_trainval/e2e_faster_rcnn_X-101-32x8d-FPN_${times}x.yaml RNG_SEED $rng_seed FAST_RCNN.MLP_HEAD_DIM 2048 FPN.DIM 512
        ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/visual_genome_trainval/e2e_faster_rcnn_X-101-64x4d-FPN_${times}x.yaml RNG_SEED $rng_seed FAST_RCNN.MLP_HEAD_DIM 2048 FPN.DIM 512
    done
done
