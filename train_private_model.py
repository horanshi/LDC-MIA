import argparse
import json
import os
import sys
import inspect
import torch
from models import build_model
from datasets import get_dataset
from utils.evaluator import Evaluator
from utils.logger import create_logger
from utils.misc import bool_flag
from utils.trainer import Trainer
from utils.masks import generate_masks

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
    parser.add_argument("--data_root", type=str, default="data") #path to store the training dataset
    parser.add_argument("--dataset", type=str,choices=["cifar10", "cifar100",  "credit", "adult", "cinic10"], default="cifar10")
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


def train(params, mask):
    # Create logger and print params
    logger = create_logger(params)

    trainloader, n_data = get_dataset(params=params, is_train=True, mask=mask)
    validloader, _ = get_dataset(params=params, is_train=False)

    model = build_model(params)
    model.cuda()

    trainer = Trainer(model, params, n_data=n_data)
    trainer.reload_checkpoint()

    evaluator = Evaluator(model, params)

    # training
    for epoch in range(trainer.epoch, params.epochs):

        # update epoch / sampler / learning rate
        trainer.epoch = epoch
        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        # train
        for (idx, images, targets) in trainloader:
            trainer.classif_step(idx, images, targets)
            trainer.end_step()

        logger.info("============ End of epoch %i ============" % trainer.epoch)

        # evaluate classification accuracy
        scores = evaluator.run_all_evals(evals=['classif'], data_loader=validloader)
        for name, val in trainer.get_scores().items():
            scores[name] = val

        # print / JSON log
        for k, v in scores.items():
            logger.info('%s -> %.6f' % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))

        # end of epoch
        trainer.end_epoch(scores)

    return model


if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    check_parameters(params)

    if params.dataset=='cinic10':
        n_data=90000
    elif params.dataset=='credit':
        n_data=800
    elif params.dataset=='adult':
        n_data=32561
    else:
        n_data=50000

    if params.mask_path=="none":
        split_config = {"public": {"train": 0.2},
                        "private": {"train": params.private_train_split,"heldout": params.private_heldout_split},
                        "shadow1": {"train":0.15, "heldout":0.15},
                        }

        # Randomly split the data according to the configuration
        known_masks, hidden_masks = generate_masks(n_data, split_config)

        path = "data/"
        torch.save(known_masks['public'], path + params.dataset + "public.pth")
        torch.save(known_masks['private'], path + params.dataset + "private.pth")
        torch.save(known_masks['shadow1'], path + params.dataset + "private.pth")

        torch.save(hidden_masks['private']['train'], path + "hidden/" + params.dataset + "train.pth")
        torch.save(hidden_masks['private']['heldout'], path + "hidden/" + params.dataset + "heldout.pth")
        torch.save(hidden_masks['public']['train'], path + "hidden/" + params.dataset + "public_train.pth")

        torch.save(hidden_masks['shadow1']['train'], path + "hidden/" + params.dataset + "shadow1_train.pth")
        torch.save(hidden_masks['shadow1']['heldout'], path + "hidden/" + params.dataset + "shadow1_heldout.pth")

        mask = hidden_masks['private']['train']
    else:
        mask = torch.load(params.mask_path)

    train(params, mask)


