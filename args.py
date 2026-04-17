import argparse


def str2bool(value):
    """Convert string to boolean for argparse"""
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


class Args:
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Segmentation')
        # model and dataset 
        parser.add_argument('--n_class', type=int, default=2, help='segmentation classes')
        parser.add_argument('--data_path', type=str, help='path to dataset where images store')
        parser.add_argument('--model_path', type=str, help='path to store trained model files, no need to include task specific name')
        parser.add_argument('--log_path', type=str, help='path to store tensorboard log files, no need to include task specific name')
        parser.add_argument('--task_name', type=str, help='task name for naming saved model files and log files')
        parser.add_argument('--experiment', type=str, help='experiment for naming saved model files and log files')
        parser.add_argument('--input_mode', type=int, default=3, choices=[1, 2, 3], help='Input mode for MultiScaleSegFormer: 1=single input (local only), 2=dual input (local+medium), 3=triple input (local+medium+large). Default: 3')
        parser.add_argument('--use_window', type=str2bool, default=False, help='Use SAMHAWindow for MultiScaleSegFormer. Default: False. Accepts: true/false, yes/no, 1/0')

        # Distance prior ablation settings (for SAMHA)
        parser.add_argument('--distance_prior', type=str, default='log', choices=['log', 'exp', 'inv', 'gaussian', 'raw', 'none'], help='Distance prior function: log (default), exp, inv, gaussian, raw, or none (no distance bias)')
        parser.add_argument('--distance_sigma', type=float, default=1.0, help='Distance scaling parameter (sigma) for distance map computation. Default: 1.0')
        parser.add_argument('--lambda_dist_init', type=float, default=0.1, help='Initial value for lambda_dist (distance bias strength). Default: 0.1')
        parser.add_argument('--lambda_dist_trainable', type=str2bool, default=True, help='Whether lambda_dist is trainable (True) or fixed (False). Default: True')

        # dataset
        parser.add_argument('--dataset', type=int, default=2, choices=[1, 2], help='dataset for training procedure. 1=Dataset1 2=Dataset2')
        parser.add_argument('--train', action='store_true', default=False, help='train')
        parser.add_argument('--val', action='store_true', default=False, help='val')

        # context size
        parser.add_argument('--context_M', type=int, default=2, help='Medium context multiplier (e.g., 2 means 224×2 = 448)')
        parser.add_argument('--context_L', type=int, default=3, help='Large context multiplier (e.g., 3 means 224×3 = 672)')

        # model paths
        parser.add_argument('--pre_path', type=str, default="", help='name for pre model path')

        # image size and patch overlap
        parser.add_argument('--batch_size', type=int, default=3, help='batch size for origin global image (without downsampling)')
        parser.add_argument('--sub_batch_size', type=int, default=6, help='batch size for using local image patches')
        parser.add_argument('--size_p', type=int, default=508, help='size (in pixel) for cropped local image')
        parser.add_argument('--size_g', type=int, default=508, help='size (in pixel) for resized global image')
        parser.add_argument('--patch_overlap', type=float, default=0.20, help='patch overlap percentage (0.0-1.0, e.g., 0.20 = 20%% overlap)')

        # training parameters
        parser.add_argument('--num_epochs', type=int, default=50, help='number of training epochs')
        parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
        parser.add_argument('--start', type=int, default=1, help='start epoch')
        parser.add_argument('--lens', type=int, default=50, help='lens parameter')

        # wandb and WSI options
        parser.add_argument('--use_wandb', action='store_true', default=False, help='use Weights & Biases for logging')
        parser.add_argument('--wsi_level', type=int, default=3, help='WSI pyramid level for Dataset2 (3=recommended, 4=faster, 5=debug)')
        parser.add_argument('--gpu', type=str, default=0, help='GPU device ID(s) to use (e.g., "0" or "0,1" or "2"). If not set, uses all available GPUs.')

        self.parser = parser

    def parse(self):
        return self.parser.parse_args()