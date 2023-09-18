# Config Adpoted from https://github.com/cuiziteng/ICCV_MAET/blob/4e0a530473b88f8958737e7beec566696ac56309/configs/MAET_yolo/maet_yolo_ug2.py#L56

# Config for darkface (df)
df_dataset_type = 'UG2FaceDataset'
df_data_path = '/mnt/ccnas2/bdp/hf17/Datasets/Dark_face_2019/'
df_img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
df_pipeline_cfg = [
    dict(type='LoadImageFromFile'),
    dict(
      type='MultiScaleFlipAug',
      img_scale=(664, 664),
      flip=False,
      transforms=[
        dict(type='Resize', keep_ratio=True),
        dict(type='RandomFlip'),
        # dict(type='Normalize', **df_img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='Collect', keys=['img'])
      ])
]
df_data_cfg = dict(
  type = df_dataset_type,
  ann_file = df_data_path + 'main/train.txt',
  img_prefix = df_data_path,
  pipeline = df_pipeline_cfg)


