import argparse
import os
import sys
import inspect
import torch
from utils.misc import bool_flag
from train_private_model import train

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


def check_parameters(params):
    assert params.dump_path is not None
    os.makedirs(params.dump_path, exist_ok=True)


def get_parser():
    parser = argparse.ArgumentParser(description='Train/evaluate image classification models')

    # attack parameters
    parser.add_argument("--dump_path", type=str, default=None)
    parser.add_argument("--architecture", choices=["smallnet", "linear", "wrn28_10", "densenet121", "mlp", "vgg16"],default="smallnet")
    parser.add_argument("--private_train_split", type=float, default=0.25)
    parser.add_argument("--private_heldout_split", type=float, default=0.25)

    # Data parameters
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100", "credit", "adult", "cinic10"],default="cifar10")
    parser.add_argument("--mask_path", type=str, required=True)
    parser.add_argument('--n_data', type=int, default=500)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--data_num_dimensions', type=int, default=75)
    parser.add_argument('--random_seed', type=int, default=10)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--in_channels", type=int, default=3)

    # training parameters
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--optimizer", default="sgd,lr=0.001,momentum=0.9")
    parser.add_argument("--aug", type=bool_flag, default=False)  # data augmentation flag
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--log_gradients", type=bool_flag, default=False)
    parser.add_argument("--log_batch_models", type=bool_flag, default=False)  # save model for each batch of data
    parser.add_argument("--log_epoch_models", type=bool_flag, default=False)  # save model for each training epoch
    parser.add_argument('--print_freq', type=int, default=50)  # training printing frequency
    parser.add_argument("--save_periodic", type=int, default=0)  # training saving frequency

    # privacy parameters
    parser.add_argument("--private", type=bool_flag, default=False)  # privacy flag
    parser.add_argument("--noise_multiplier", type=float, default=None)
    parser.add_argument("--privacy_epsilon", type=float, default=None)
    parser.add_argument("--privacy_delta", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    return parser


if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    path = "data/"

    known_masks, hidden_masks = {}, {}
    hidden_masks['public'], hidden_masks['private'] = {}, {}
    hidden_masks['shadow1'] = {}
    hidden_masks['shadow2'] = {}
    hidden_masks['shadow3'] = {}

    known_masks['public'] = torch.load(params.mask_path + params.dataset + "public.pth")
    known_masks['private'] = torch.load(params.mask_path + params.dataset + "private.pth")
    hidden_masks['private']['train'] = torch.load(params.mask_path + "hidden/"+ params.dataset + "train.pth")
    hidden_masks['private']['heldout'] = torch.load(params.mask_path + "hidden/"+ params.dataset + "heldout.pth")
    hidden_masks['public']['train'] = torch.load(params.mask_path + "hidden/"+ params.dataset + "public_train.pth")

    hidden_masks['shadow1']['train'] = torch.load(params.mask_path + "hidden/"+ params.dataset +"shadow1_train.pth")
    hidden_masks['shadow1']['heldout'] = torch.load(params.mask_path + "hidden/"+ params.dataset +"shadow1_heldout.pth")

    train(params, hidden_masks['shadow1']['train'])