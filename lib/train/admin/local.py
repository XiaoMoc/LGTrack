class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/xiaoyang/Documents/PyProjects/LGTrack-main'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/xiaoyang/Documents/PyProjects/LGTrack-main/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/xiaoyang/Documents/PyProjects/LGTrack-main/pretrained_networks'
        self.lasot_dir = '/media/xiaoyang/new_ssd_volume/ampa_migra/F/data/lasot'
        self.got10k_dir = '/media/xiaoyang/new_ssd_volume/ampa_migra/F/data/got10k/train'
        self.got10k_val_dir = '/media/xiaoyang/new_ssd_volume/ampa_migra/F/data/got10k/val'
        self.lasot_lmdb_dir = '/home/xiaoyang/Documents/PyProjects/LGTrack-main/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/xiaoyang/Documents/PyProjects/LGTrack-main/data/got10k_lmdb'
        self.trackingnet_dir = '/media/xiaoyang/new_ssd_volume/ampa_migra/F/data/trackingnet'
        self.trackingnet_lmdb_dir = '/home/xiaoyang/Documents/PyProjects/LGTrack-main/data/trackingnet_lmdb'
        self.coco_dir = '/media/xiaoyang/new_ssd_volume/ampa_migra/F/data/coco'
        self.coco_lmdb_dir = '/home/xiaoyang/Documents/PyProjects/LGTrack-main/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/xiaoyang/Documents/PyProjects/LGTrack-main/data/vid'
        self.imagenet_lmdb_dir = '/home/xiaoyang/Documents/PyProjects/LGTrack-main/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
