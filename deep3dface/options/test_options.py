"""This script contains the test options for Deep3DFaceRecon_pytorch
"""

from .base_options import BaseOptions


# class TestOptions(BaseOptions):
#     """This class includes test options.

#     It also includes shared options defined in BaseOptions.
#     """

#     def initialize(self, parser):
#         parser = BaseOptions.initialize(self, parser)  # define shared options
#         parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
#         parser.add_argument('--dataset_mode', type=str, default=None, help='chooses how datasets are loaded. [None | flist]')
#         parser.add_argument('--img_folder', type=str, default='examples', help='folder for test images.')

#         # Dropout and Batchnorm has different behavior during training and test.
#         self.isTrain = False
#         return parser

class TestOptions:
    """Class to define and manage test options."""

    def __init__(self):
        self.add_image = True
        self.bfm_folder = "BFM"
        self.bfm_model = "BFM_model_front.mat"
        self.camera_d = 10.0
        self.center = 112.0
        self.checkpoints_dir = "./checkpoints"
        self.dataset_mode = None
        self.ddp_port = "12355"
        self.display_per_batch = True
        self.epoch = "20"
        self.eval_batch_nums = float("inf")
        self.focal = 1015.0
        self.gpu_ids = "0"
        self.img_folder = "examples"
        self.init_path = "checkpoints/init_model/resnet50-0676ba61.pth"
        self.isTrain = False  # Default is False for testing
        self.model = "facerecon"
        self.name = ""
        self.net_recon = "resnet50"
        self.phase = "test"
        self.suffix = ""
        self.use_ddp = False  # Default is False for single GPU
        self.use_last_fc = False
        self.use_opengl = True
        self.verbose = False
        self.vis_batch_nums = 1
        self.world_size = 1
        self.z_far = 15.0
        self.z_near = 5.0

    def print_options(self):
        """Print all options."""
        print("----------------- Options ---------------")
        for key, value in vars(self).items():
            print(f"{key:>25}: {value}")
        print("----------------- End -------------------")

  