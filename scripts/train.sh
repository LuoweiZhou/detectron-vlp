for rng_seed in 3 227 1989; do
    ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/visual_genome/e2e_faster_rcnn_R-101-FPN_1x.yaml RNG_SEED $rng_seed
    ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/visual_genome/e2e_faster_rcnn_R-101-FPN_2x.yaml RNG_SEED $rng_seed
    ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/visual_genome/e2e_faster_rcnn_R-50-FPN_1x.yaml RNG_SEED $rng_seed
    ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/visual_genome/e2e_faster_rcnn_R-50-FPN_2x.yaml RNG_SEED $rng_seed
    ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/visual_genome/e2e_faster_rcnn_X-101-32x8d-FPN_1x.yaml RNG_SEED $rng_seed
    ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/visual_genome/e2e_faster_rcnn_X-101-32x8d-FPN_2x.yaml RNG_SEED $rng_seed
    ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/visual_genome/e2e_faster_rcnn_X-101-64x4d-FPN_1x.yaml RNG_SEED $rng_seed
    ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/visual_genome/e2e_faster_rcnn_X-101-64x4d-FPN_2x.yaml RNG_SEED $rng_seed
done

# for rng_seed in 3 227 1989; do
#     for attr in 0. 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.; do
#         for base_lr in 0.005 0.01; do
#             # ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/visual_genome/e2e_faster_rcnn_X-101-32x8d-FPN_1x.yaml RNG_SEED $rng_seed SOLVER.BASE_LR $base_lr MODEL.LOSS_ATTR $attr;
#             # ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/visual_genome/e2e_faster_rcnn_X-101-32x8d-FPN_2x.yaml RNG_SEED $rng_seed SOLVER.BASE_LR $base_lr MODEL.LOSS_ATTR $attr;
#             ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/visual_genome/e2e_faster_rcnn_X-101-64x4d-FPN_1x.yaml RNG_SEED $rng_seed SOLVER.BASE_LR $base_lr MODEL.LOSS_ATTR $attr MODEL.CLS_EMBED False;
#             ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/visual_genome/e2e_faster_rcnn_X-101-64x4d-FPN_2x.yaml RNG_SEED $rng_seed SOLVER.BASE_LR $base_lr MODEL.LOSS_ATTR $attr MODEL.CLS_EMBED False;
#         done
#     done
# done

# ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/visual_genome/e2e_faster_rcnn_X-101-64x4d-FPN_2x.yaml RNG_SEED 3 SOLVER.BASE_LR 0.005 MODEL.LOSS_ATTR 0.4
# ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/visual_genome/e2e_faster_rcnn_X-101-64x4d-FPN_2x.yaml RNG_SEED 227 SOLVER.BASE_LR 0.01 MODEL.LOSS_ATTR 0.6
# ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/visual_genome/e2e_faster_rcnn_X-101-64x4d-FPN_1x.yaml RNG_SEED 227 SOLVER.BASE_LR 0.005 MODEL.LOSS_ATTR 0.7
# ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/visual_genome/e2e_faster_rcnn_X-101-64x4d-FPN_1x.yaml RNG_SEED 227 SOLVER.BASE_LR 0.005 MODEL.LOSS_ATTR 0.8
# ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/visual_genome/e2e_faster_rcnn_X-101-64x4d-FPN_1x.yaml RNG_SEED 227 SOLVER.BASE_LR 0.01 MODEL.LOSS_ATTR 0.8
# ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/visual_genome/e2e_faster_rcnn_X-101-64x4d-FPN_2x.yaml RNG_SEED 227 SOLVER.BASE_LR 0.01 MODEL.LOSS_ATTR 0.8
# ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/visual_genome/e2e_faster_rcnn_X-101-64x4d-FPN_2x.yaml RNG_SEED 227 SOLVER.BASE_LR 0.005 MODEL.LOSS_ATTR 0.7
# ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/visual_genome/e2e_faster_rcnn_X-101-64x4d-FPN_1x.yaml RNG_SEED 227 SOLVER.BASE_LR 0.005 MODEL.LOSS_ATTR 0.9
# ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/visual_genome/e2e_faster_rcnn_X-101-64x4d-FPN_2x.yaml RNG_SEED 227 SOLVER.BASE_LR 0.005 MODEL.LOSS_ATTR 0.9
# ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/visual_genome/e2e_faster_rcnn_X-101-64x4d-FPN_1x.yaml RNG_SEED 227 SOLVER.BASE_LR 0.01 MODEL.LOSS_ATTR 0.9