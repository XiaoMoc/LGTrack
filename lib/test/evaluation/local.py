from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.biodrone_path = '/home/xiaoyang/Documents/PyProjects/LGTrack-main/data/biodrone'
    settings.davis_dir = ''
    settings.dtb70_path = '/media/xiaoyang/new_ssd_volume/ampa_migra/F/data/DTB70'
    settings.got10k_lmdb_path = '/home/xiaoyang/Documents/PyProjects/LGTrack-main/data/got10k_lmdb'
    settings.got10k_path = '/home/xiaoyang/Documents/PyProjects/LGTrack-main/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/xiaoyang/Documents/PyProjects/LGTrack-main/data/itb'
    settings.lasot_extension_subset_path_path = '/home/xiaoyang/Documents/PyProjects/LGTrack-main/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/xiaoyang/Documents/PyProjects/LGTrack-main/data/lasot_lmdb'
    settings.lasot_path = '/home/xiaoyang/Documents/PyProjects/LGTrack-main/data/lasot'
    settings.network_path = '/home/xiaoyang/Documents/PyProjects/LGTrack-main/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/xiaoyang/Documents/PyProjects/LGTrack-main/data/nfs'
    settings.otb_path = '/home/xiaoyang/Documents/PyProjects/LGTrack-main/data/otb'
    settings.prj_dir = '/home/xiaoyang/Documents/PyProjects/LGTrack-main'
    settings.result_plot_path = '/home/xiaoyang/Documents/PyProjects/LGTrack-main/output/test/result_plots'
    settings.results_path = '/home/xiaoyang/Documents/PyProjects/LGTrack-main/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/xiaoyang/Documents/PyProjects/LGTrack-main/output'
    settings.segmentation_path = '/home/xiaoyang/Documents/PyProjects/LGTrack-main/output/test/segmentation_results'
    settings.tc128_path = '/home/xiaoyang/Documents/PyProjects/LGTrack-main/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/xiaoyang/Documents/PyProjects/LGTrack-main/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/xiaoyang/Documents/PyProjects/LGTrack-main/data/trackingnet'
    settings.uav123_path = '/media/xiaoyang/new_ssd_volume/ampa_migra/F/data/uav123'
    settings.uav_path = '/home/xiaoyang/Documents/PyProjects/LGTrack-main/data/uav'
    settings.uavdt_path = '/media/xiaoyang/new_ssd_volume/ampa_migra/F/data/uavdt'
    settings.visdrone2018_path = '/media/xiaoyang/new_ssd_volume/ampa_migra/F/data/visdrone2018'
    settings.vot18_path = '/home/xiaoyang/Documents/PyProjects/LGTrack-main/data/vot2018'
    settings.vot22_path = '/home/xiaoyang/Documents/PyProjects/LGTrack-main/data/vot2022'
    settings.vot_path = '/home/xiaoyang/Documents/PyProjects/LGTrack-main/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

