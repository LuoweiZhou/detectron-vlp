import json

data_file = 'anet_data/dic_anet.json'
tmp_splits = 'tmp_splits/'
num_splits = 16

with open(data_file) as f:
    seg_lst = [i['id'] for i in json.load(f)['videos']]

print(len(seg_lst))

num_seg = len(seg_lst)
itv = num_seg*1./num_splits


for i in range(num_splits):
    start_ind = int(i*itv)
    end_ind = int((i+1)*itv)
    print(start_ind, end_ind)

    with open(tmp_splits+'anet_split_'+str(i)+'.json', 'w') as f:
        json.dump(seg_lst[start_ind:end_ind], f)
