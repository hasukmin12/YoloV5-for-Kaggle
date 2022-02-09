import numpy as np
from tqdm.notebook import tqdm
tqdm.pandas()
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import glob
import shutil
import sys
sys.path.append('../input/tensorflow-great-barrier-reef')
from joblib import Parallel, delayed
from IPython.display import display
import wandb
import sklearn

# try:
#     from kaggle_secrets import UserSecretsClient
#     user_secrets = UserSecretsClient()
#     api_key = user_secrets.get_secret("WANDB")
#     wandb.login(key=api_key)
#     anonymous = None
# except:
#     wandb.login(anonymous='must')
#     print('To use your W&B account,\nGo to Add-ons -> Secrets and provide your W&B access token. Use the Label name as WANDB. \nGet your W&B access token from here: https://wandb.ai/authorize')


FOLD      = 1 # which fold to train
DIM       = 3000
MODEL     = 'yolov5s6'
BATCH     = 16
EPOCHS    = 7
OPTMIZER  = 'Adam'

PROJECT   = 'great-barrier-reef-public' # w&b in yolov5
NAME      = f'{MODEL}-dim{DIM}-fold{FOLD}' # w&b for yolov5

REMOVE_NOBBOX = True # remove images with no bbox
ROOT_DIR  = '/data5/sukmin/kaggle/great_barrier_reef'
IMAGE_DIR = '/data5/sukmin/kaggle/images' # directory to save images
LABEL_DIR = '/data5/sukmin/kaggle/labels' # directory to save labels

if os.path.isdir(IMAGE_DIR)== False:
    os.mkdir(IMAGE_DIR)
if os.path.isdir(LABEL_DIR)== False:
    os.mkdir(LABEL_DIR)


# Train Data
df = pd.read_csv(f'{ROOT_DIR}/train.csv')
df['old_image_path'] = f'{ROOT_DIR}/train_images/video_'+df.video_id.astype(str)+'/'+df.video_frame.astype(str)+'.jpg'
df['image_path']  = f'{IMAGE_DIR}/'+df.image_id+'.jpg'
df['label_path']  = f'{LABEL_DIR}/'+df.image_id+'.txt'
df['annotations'] = df['annotations'].progress_apply(eval)
# display(df.head(2))
# print(df)

# number bbox
df['num_bbox'] = df['annotations'].progress_apply(lambda x: len(x))
data = (df.num_bbox>0).value_counts(normalize=True)*100
# print(f"No BBox: {data[0]:0.2f}% | With BBox: {data[1]:0.2f}%")


# erase Non-label data
if REMOVE_NOBBOX:
    df = df.query("num_bbox>0")


# # image copy 2 write access for YOLOv5
# def make_copy(row):
#     shutil.copyfile(row.old_image_path, row.image_path)
#     return
# image_paths = df.old_image_path.tolist()
# _ = Parallel(n_jobs=-1, backend='threading')(delayed(make_copy)(row) for _, row in tqdm(df.iterrows(), total=len(df)))


# check https://github.com/awsaf49/bbox for source code of following utility functions
from bbox.utils import coco2yolo, coco2voc, voc2yolo
from bbox.utils import draw_bboxes, load_image
from bbox.utils import clip_bbox, str2annot, annot2str

def get_bbox(annots):
    bboxes = [list(annot.values()) for annot in annots]
    return bboxes

# def get_imgsize(row):
#     row['width'], row['height'] = imagesize.get(row['image_path'])
#     return row

np.random.seed(32)
colors = [(np.random.randint(255), np.random.randint(255), np.random.randint(255))\
          for idx in range(1)]



# add bbox, width, height
df['bboxes'] = df.annotations.progress_apply(get_bbox)
df.head(2)
df['width']  = 1280
df['height'] = 720
display(df.head(2))




# create Labels for YOLOv5
# changes labels COCO to YOLO
cnt = 0
all_bboxes = []
bboxes_info = []
for row_idx in tqdm(range(df.shape[0])):
    row = df.iloc[row_idx]
    image_height = row.height
    image_width  = row.width
    bboxes_coco  = np.array(row.bboxes).astype(np.float32).copy()
    num_bbox     = len(bboxes_coco)
    names        = ['cots']*num_bbox
    labels       = np.array([0]*num_bbox)[..., None].astype(str)
    ## Create Annotation(YOLO)
    with open(row.label_path, 'w') as f:
        if num_bbox<1:
            annot = ''
            f.write(annot)
            cnt+=1
            continue
        bboxes_voc  = coco2voc(bboxes_coco, image_height, image_width)
        bboxes_voc  = clip_bbox(bboxes_voc, image_height, image_width)
        bboxes_yolo = voc2yolo(bboxes_voc, image_height, image_width).astype(str)
        all_bboxes.extend(bboxes_yolo.astype(float))
        bboxes_info.extend([[row.image_id, row.video_id, row.sequence]]*len(bboxes_yolo))
        annots = np.concatenate([labels, bboxes_yolo], axis=1)
        string = annot2str(annots)
        f.write(string)
print('Missing:',cnt)
print("")



# cross validation
from sklearn.model_selection import GroupKFold
kf = GroupKFold(n_splits = 3)
df = df.reset_index(drop=True)
df['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(kf.split(df, groups=df.video_id.tolist())):
    df.loc[val_idx, 'fold'] = fold
print("Cross validation results")
display(df.fold.value_counts())
print("")



# BBox distribution
bbox_df = pd.DataFrame(np.concatenate([bboxes_info, all_bboxes], axis=1),
             columns=['image_id','video_id','sequence',
                     'xmid','ymid','w','h'])
bbox_df[['xmid','ymid','w','h']] = bbox_df[['xmid','ymid','w','h']].astype(float)
bbox_df['area'] = bbox_df.w * bbox_df.h * 1280 * 720
bbox_df = bbox_df.merge(df[['image_id','fold']], on='image_id', how='left')
bbox_df.head(2)



# # show visualization
# df2 = df[(df.num_bbox>0)].sample(100) # takes samples with bbox
# y = 3; x = 2
# plt.figure(figsize=(12.8*x, 7.2*y))
# for idx in range(x*y):
#     row = df2.iloc[idx]
#     img           = load_image(row.image_path)
#     image_height  = row.height
#     image_width   = row.width
#     with open(row.label_path) as f:
#         annot = str2annot(f.read())
#     bboxes_yolo = annot[...,1:]
#     labels      = annot[..., 0].astype(int).tolist()
#     names         = ['cots']*len(bboxes_yolo)
#     plt.subplot(y, x, idx+1)
#     plt.imshow(draw_bboxes(img = img,
#                            bboxes = bboxes_yolo,
#                            classes = names,
#                            class_ids = labels,
#                            class_name = True,
#                            colors = colors,
#                            bbox_format = 'yolo',
#                            line_thickness = 2))
#     plt.axis('OFF')
# plt.tight_layout()
# plt.show()











# now we really start train

# Datasets
train_files = []
val_files   = []
train_df = df.query("fold!=@FOLD")
valid_df = df.query("fold==@FOLD")
train_files += list(train_df.image_path.unique())
val_files += list(valid_df.image_path.unique())
print(len(train_files), len(val_files))



#configuration
import yaml
cwd = '/data5/sukmin/kaggle/working/'
if os.path.isdir(cwd)== False:
    os.mkdir(cwd)

with open(os.path.join(cwd, 'train.txt'), 'w') as f:
    for path in train_df.image_path.tolist():
        f.write(path + '\n')

with open(os.path.join(cwd, 'val.txt'), 'w') as f:
    for path in valid_df.image_path.tolist():
        f.write(path + '\n')

data = dict(
    path='/kaggle/working',
    train=os.path.join(cwd, 'train.txt'),
    val=os.path.join(cwd, 'val.txt'),
    nc=1,
    names=['cots'],
)

with open(os.path.join(cwd, 'gbr.yaml'), 'w') as outfile:
    yaml.dump(data, outfile, default_flow_style=False)

f = open(os.path.join(cwd, 'gbr.yaml'), 'r')
print('\nyaml:')
print(f.read())


# from yolov5 import distutils
# display = utils.notebook_init()  # check