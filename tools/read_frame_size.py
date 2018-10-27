
import json
import cv2
import os
import numpy as np

data_file = 'anet_data/dic_anet.json'
img_root = '/checkpoint02/luoweizhou/dat/anet/frames_10frm'
target_file = 'rhw.npy'

rhw = []
with open(data_file) as f:
    seg_lst = [i['id'] for i in json.load(f)['videos']]

for idx, seg in enumerate(seg_lst):
    if idx % 1000 == 0:
        print(idx)
    img_path = os.path.join(img_root, seg, '01.jpg')
    img = cv2.imread(img_path)
    try:
        rhw.append(img.shape[:2])
        assert(img.shape[1] == 720)
    except:
        rhw.append(np.zeros(2))
        print('no such file - {}'.format(seg))

print(len(rhw))
print(np.mean(rhw, axis=0))
rhw = np.stack(rhw)

with open(target_file, 'w') as f:
    np.save(f, rhw)
