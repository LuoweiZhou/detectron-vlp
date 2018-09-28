import h5py
import os
import numpy as np

# N' - total number of segments
# Output dets_labels size: N', 200, 7
# Output dets_num size: N', # useful because some segment does not have 10 frames
# Output nms_num size: N',

num_split = 16
proposal_per_frm = 20
num_frm = 10
data_root = '/checkpoint02/luoweizhou/dat/anet/'
file_prefix = 'others/anet_detection_vg_fc6_feat'
target_file = data_root+'anet_detection_vg_fc6_feat.h5'

dets_labels_lst = []
dets_num_lst = []
nms_num_lst = []

for i in range(num_split):
    h5_file = os.path.join(data_root, file_prefix+str(i)+'.h5')
    print('Loading h5 file: {}'.format(h5_file))

    h5_proposal_file = h5py.File(h5_file, 'r', driver='core')
    label_proposals = h5_proposal_file['dets_labels'][:]
    num_proposals = h5_proposal_file['dets_num'][:]
    num_nms = h5_proposal_file['nms_num'][:]
    h5_proposal_file.close()

    N = label_proposals.shape[0]
    flat_label_proposals = label_proposals[:, :, :proposal_per_frm, :].copy().reshape(N, proposal_per_frm*num_frm, 6)

    frm_idx = np.arange(num_frm)
    tile_frm_idx = np.tile(frm_idx.reshape(1,num_frm,1), (N,1,proposal_per_frm)).reshape(N,proposal_per_frm*num_frm, 1)

    flat_label_proposals = np.concatenate((flat_label_proposals[:, :, :4], tile_frm_idx, flat_label_proposals[:, :, 4:]), axis=2)

    dets_labels_lst.append(flat_label_proposals)
    dets_num_lst.append(np.sum(num_proposals, axis=1))
    nms_num_lst.append(np.sum(num_nms, axis=1))

dets_labels_lst = np.concatenate(dets_labels_lst)
dets_num_lst = np.concatenate(dets_num_lst)
nms_num_lst = np.concatenate(nms_num_lst)

print(dets_labels_lst.shape, dets_num_lst.shape, nms_num_lst.shape)
assert(np.mean(dets_num_lst) == np.mean(nms_num_lst))
print('On average, {} out of {} frames have proposals'.format(np.mean(dets_num_lst)/proposal_per_frm, num_frm))

f = h5py.File(target_file, "w")
f.create_dataset("dets_labels", data=dets_labels_lst)
f.create_dataset("dets_num", data=dets_num_lst)
f.create_dataset("nms_num", data=nms_num_lst)
f.close()
