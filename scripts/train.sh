for times in 0.5 1 2; do
    for rng_seed in 3 227 1989; do
        ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/ade/e2e_faster_rcnn_R-101-FPN_${times}x.yaml RNG_SEED $rng_seed
        ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/ade/e2e_faster_rcnn_R-50-FPN_${times}x.yaml RNG_SEED $rng_seed
        ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/ade/e2e_faster_rcnn_X-101-32x8d-FPN_${times}x.yaml RNG_SEED $rng_seed
        ~/devfair/starter.sh 8 1 72 learnfair vqa train_net --cfg configs/ade/e2e_faster_rcnn_X-101-64x4d-FPN_${times}x.yaml RNG_SEED $rng_seed
    done
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
