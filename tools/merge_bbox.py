# merge multiple bbox .h5 files into one

import h5py

src_file_prefix = '/z/dat/VLP/dat/SBU/region_feat_gvd_wo_bgd/raw_bbox/sbu_detection_vg_100dets_vlp_checkpoint_trainval_bbox'
target_file = '/z/dat/VLP/dat/SBU/region_feat_gvd_wo_bgd/bbox/sbu_detection_vg_100dets_vlp_checkpoint_trainval_bbox.h5'

with h5py.File(target_file, 'w') as f_t:
    for i in range(1000):
      # print(i)
      try:
        with h5py.File(src_file_prefix+str(i).zfill(3)+'.h5', 'r') as f_s:
            for key, val in f_s.items():
                f_t.create_dataset(key, data=val)
      except IOError:
        print(i)
