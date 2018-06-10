for rng_seed in 5 229 1991; do
    for times in 1 2; do
        # ~/devfair/starter.sh 8 1 72 priority,uninterrupted pyra train_net --cfg configs/coco/e2e_faster_rcnn_R-50-FPN_${times}x.yaml RNG_SEED $rng_seed 
        # ~/devfair/starter.sh 8 1 72 priority,uninterrupted pyra train_net --cfg configs/coco/e2e_faster_rcnn_R-50-FPN_${times}x.yaml RNG_SEED $rng_seed TRAIN.CPP_RPN proposals DATA_LOADER.NUM_THREADS 5
        ~/devfair/starter.sh 8 1 72 priority,uninterrupted pyra train_net --cfg configs/coco/e2e_faster_rcnn_R-50-FPN_${times}x.yaml RNG_SEED $rng_seed TRAIN.CPP_RPN all DATA_LOADER.NUM_THREADS 6 NUM_CPUS 36
    done
done
