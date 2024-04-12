import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn import preprocessing
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from torchvision import transforms
from datasets import get_transform, IdxDataset, MaskDataset
from models import build_model
from train_private_model import train
from utils.evaluator import Evaluator
from utils.logger import create_logger
from utils.masks import generate_masks, to_mask, merge_mask
from utils.misc import bool_flag
from utils.trainer import Trainer
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset


def get_parser():
    """
    Generate a parameters parser.
    """
    parser = argparse.ArgumentParser(description='Privacy attack parameters')

    # attack parameters
    parser.add_argument("--architecture", choices=["smallnet", "linear", "wrn28_10", "densenet121", "mlp", "vgg16"],default="smallnet")
    parser.add_argument("--private_architecture",choices=["smallnet", "linear", "wrn28_10", "densenet121", "mlp", "vgg16"], default="smallnet")
    parser.add_argument("--attack_architecture",choices=["smallnet", "linear", "wrn28_10", "densenet121", "mlp", "vgg16"], default="smallnet")
    parser.add_argument("--dump_path", type=str, default="attack_model")  # path to the reference model
    parser.add_argument("--model_path", type=str, default="model")  # path to the private model
    parser.add_argument("--mia_model_path", type=str, default="model")  # path to the mia classifier
    parser.add_argument("--shadow_path", type=str, default="shadow_1")  # path to the shadow model
    parser.add_argument("--cos_threshold", type=float, default=0)

    # data parameters
    parser.add_argument("--data_root", type=str, default="data")  # path to the data
    parser.add_argument("--dataset", type=str,choices=["cifar10", "cifar100",  "credit", "adult", "cinic10"], default="cifar10")
    parser.add_argument("--mask_path", type=str, required=True)  # path to the data mask
    parser.add_argument('--data_num_dimensions', type=int, default=75)  # number of features for non-image data
    parser.add_argument('--classifier_num_dimensions', type=int, default=75)  # number of features for non-image data
    parser.add_argument("--num_classes", type=int, default=10)  # number of classes for classification task
    parser.add_argument("--in_channels", type=int, default=3)  # number of input channels for image data

    # training parameters
    parser.add_argument("--aug", type=bool_flag, default=False)  # data augmentation flag
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)  # training epochs of reference model
    parser.add_argument("--classifier_epochs", type=int, default=50)  # training epochs of MIA classifier
    parser.add_argument("--optimizer", default="sgd,lr=0.001,momentum=0.9")
    parser.add_argument("--num_workers", type=int, default=2)
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


def adult_data_transform(df):
    binary_data = pd.get_dummies(df)
    feature_cols = binary_data[binary_data.columns[:-2]]
    scaler = preprocessing.StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(feature_cols), columns=feature_cols.columns)
    return data


def get_dataset(params):
    if params.dataset == 'cifar10':
        if params.aug == True:
            augmentations = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
            normalize = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            model_transform = transforms.Compose(augmentations + normalize)
        else:
            normalize = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            model_transform = transforms.Compose(normalize)
        return torchvision.datasets.CIFAR10(root=params.data_root, train=True, download=True, transform=model_transform)

    elif params.dataset == 'cifar100':
        if params.aug:
            augmentations = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
            normalize = [transforms.ToTensor(), transforms.Normalize(mean=[n / 255 for n in [129.3, 124.1, 112.4]],std=[n / 255 for n in [68.2, 65.4, 70.4]])]
            transform = transforms.Compose(augmentations + normalize)

        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean=[n / 255 for n in [129.3, 124.1, 112.4]], std=[n / 255 for n in [68.2, 65.4, 70.4]])])

        dataset = torchvision.datasets.CIFAR100(root=params.data_root, train=True, download=True, transform=transform)
        return dataset

    elif params.dataset == 'cinic10':
        if params.aug == True:
            augmentations = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
            normalize = [transforms.ToTensor(), transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404],std=[0.24205776, 0.23828046, 0.25874835])]
            transform = transforms.Compose(augmentations + normalize)
        else:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404],std=[0.24205776, 0.23828046, 0.25874835])])

        dataset = torchvision.datasets.ImageFolder(root=params.data_root + '/train', transform=transform)
        return dataset

    elif params.dataset == 'credit':
        cred = fetch_openml('credit-g', version='1', as_frame=False, parser='liac-arff')

        data = SimpleImputer(missing_values=np.nan, strategy='mean', copy=True).fit(cred.data).transform(cred.data)
        target = preprocessing.LabelEncoder().fit(cred.target).transform(cred.target)
        X = data
        norm = np.max(np.concatenate((-1 * X.min(axis=0)[np.newaxis], X.max(axis=0)[np.newaxis]), axis=0).T,axis=1).astype('float32')
        data = np.divide(data, norm)

        data = torch.tensor(data).float()
        target = torch.tensor(target).long()

        ids = np.arange(1000)[:800]

        final_data = []
        for i in ids:
            final_data.append([data[i], target[i]])
        params.num_classes = 2
        return final_data

    elif params.dataset == 'adult':
        columns = ["age", "workClass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                   "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",
                   "income"]
        train_data = pd.read_csv(params.data_root + '/adult.data', names=columns, sep=' *, *', na_values='?',engine='python')
        test_data = pd.read_csv(params.data_root + '/adult.test', names=columns, sep=' *, *', skiprows=1, na_values='?',engine='python')

        original_train = train_data
        original_test = test_data
        num_train = len(original_train)
        original = pd.concat([original_train, original_test])
        labels = original['income']
        labels = labels.replace('<=50K', 0).replace('>50K', 1)
        labels = labels.replace('<=50K.', 0).replace('>50K.', 1)

        # Remove target
        del original["income"]

        data = adult_data_transform(original)
        train_data = data[:num_train]
        train_labels = labels[:num_train]
        test_data = data[num_train:]
        test_labels = labels[num_train:]

        test_data = torch.tensor(test_data.to_numpy()).float()
        train_data = torch.tensor(train_data.to_numpy()).float()
        test_labels = torch.tensor(test_labels.to_numpy(dtype='int64')).long()
        train_labels = torch.tensor(train_labels.to_numpy(dtype='int64')).long()

        final_data = []
        for i in np.arange(len(train_data)):
            final_data.append([train_data[i], train_labels[i]])
        return final_data


def one_hot(x, params):
    return torch.eye(params.num_classes)[x,:]


def get_label(params, mask):
    label=[]
    ids = (mask == True).nonzero().flatten().numpy()
    dataset = get_dataset(params)
    for id in ids:
        target = torch.tensor(dataset[id][1]).unsqueeze(0)
        target = target.cpu()
        label.append(target[0])
    return label


def inference(private_model, loader):
    losses = []
    private_model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            output = private_model(images)
            loss = -F.cross_entropy(output, targets, reduction='none')
            losses.extend(loss.tolist())

    return torch.tensor(losses)


def collect_outputs(model, loader):
    model.eval()
    outputs = []

    with torch.no_grad():
        for images in loader:
            images = images[0].cuda(non_blocking=True)
            model_output = model(images)
            outputs.extend(model_output.detach().cpu())

    return outputs


# Get the membership score feature of LDC-MIA
def get_losses(params, model_path, mask_in, mask_out):
    params.architecture = params.private_architecture

    # get the final model parameters
    private_model = build_model(params)
    private_model_path = os.path.join(model_path, "checkpoint.pth")
    state_dict_private = torch.load(private_model_path, map_location='cuda:0')
    private_model.load_state_dict(state_dict_private['model'])
    private_model = private_model.cuda()
    private_model.eval()

    # get the appropriate ids to dot product
    private_train_ids = (mask_in == True).nonzero().flatten().numpy()
    private_heldout_ids = (mask_out == True).nonzero().flatten().numpy()
    # load the dataset
    dataset = get_dataset(params)

    train_dataset = torch.utils.data.Subset(dataset, private_train_ids)
    heldout_dataset = torch.utils.data.Subset(dataset, private_heldout_ids)

    # Create DataLoaders for batching
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, pin_memory=True)
    heldout_loader = DataLoader(heldout_dataset, batch_size=64, shuffle=False, pin_memory=True)

    # Now perform the inference
    train_losses = inference(private_model, train_loader)
    heldout_losses = inference(private_model, heldout_loader)
    return train_losses, heldout_losses


# Get the calibrated membership score
def get_calibrated_losses(params, private_model, attack_model, ids):
    dataset = get_dataset(params)

    sub_dataset = Subset(dataset, ids)
    data_loader = DataLoader(sub_dataset, batch_size=64, shuffle=False, pin_memory=True)

    private_model.eval()
    attack_model.eval()
    all_losses = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            # Get the losses from both models
            output = private_model(images)
            private_loss = -F.cross_entropy(output, targets, reduction='none')

            attack_output = attack_model(images)
            attack_loss = -F.cross_entropy(attack_output, targets, reduction='none')

            # Calculate the calibrated loss for each item in the batch
            calibrated_losses = private_loss - attack_loss
            all_losses.extend(calibrated_losses.tolist())

    return torch.tensor(all_losses)


# Get the calibrated membership score of LDC-MIA
def calibrated_loss_attack(params, shadow_path, mask_in, mask_out):
    # load the masks
    known_masks, hidden_masks = {}, {}
    hidden_masks['public'], hidden_masks['private'] = {}, {}
    known_masks['public'] = torch.load(params.mask_path + params.dataset+ "public.pth")
    hidden_masks['public']['train'] = torch.load(params.mask_path + "hidden/" +params.dataset + "public_train.pth")

    # train the reference model
    params.architecture = params.attack_architecture
    attack_model = train(params, hidden_masks['public']['train'])
    attack_model = attack_model.cuda()

    # load the private model
    params.architecture = params.private_architecture
    private_model = build_model(params)
    private_model_path = os.path.join(shadow_path, "checkpoint.pth")
    state_dict_private = torch.load(private_model_path)
    private_model.load_state_dict(state_dict_private['model'])
    private_model = private_model.cuda()

    # get the appropriate ids to dot product
    private_train_ids = (mask_in == True).nonzero().flatten().numpy()
    private_heldout_ids = (mask_out == True).nonzero().flatten().numpy()

    private_model.eval()
    attack_model.eval()
    train_losses = get_calibrated_losses(params, private_model, attack_model, private_train_ids)
    heldout_losses = get_calibrated_losses(params, private_model, attack_model, private_heldout_ids)

    return train_losses, heldout_losses


# Get the logit vector for calculating neighborhood information
def get_neighbor(params, mask_in, mask_out, mask_reference):
    # get the reference model parameters
    params.architecture = params.attack_architecture
    private_model = build_model(params)
    private_model_path = os.path.join(params.dump_path, "checkpoint.pth")
    state_dict_private = torch.load(private_model_path)
    private_model.load_state_dict(state_dict_private['model'])
    private_model = private_model.cuda()
    private_model.eval()

    # get the appropriate ids to dot product
    private_train_ids = (mask_in == True).nonzero().flatten().numpy()
    private_heldout_ids = (mask_out == True).nonzero().flatten().numpy()
    private_reference_ids = (mask_reference == True).nonzero().flatten().numpy()

    # load the dataset
    dataset = get_dataset(params)
    batch_size = 64

    # Create DataLoader for each set of IDs
    train_loader = DataLoader(Subset(dataset, private_train_ids), batch_size=batch_size, shuffle=False,pin_memory=True)
    heldout_loader = DataLoader(Subset(dataset, private_heldout_ids), batch_size=batch_size, shuffle=False,pin_memory=True)
    reference_loader = DataLoader(Subset(dataset, private_reference_ids), batch_size=batch_size, shuffle=False,pin_memory=True)

    # Collect outputs for each dataset
    train_neighbor = collect_outputs(private_model, train_loader)
    heldout_neighbor = collect_outputs(private_model, heldout_loader)
    reference_neighbor = collect_outputs(private_model, reference_loader)

    return train_neighbor, heldout_neighbor, reference_neighbor


# Get the neighborhood information of LDC-MIA
def neighbor_cos_similarity(params, train_neighbor, heldout_neighbor, reference_neighbor):
    train_neighbor = torch.stack(train_neighbor)
    heldout_neighbor = torch.stack(heldout_neighbor)
    reference_neighbor = torch.stack(reference_neighbor)
    cos_tensor = torch.cat((train_neighbor, heldout_neighbor, reference_neighbor))

    # Calculate the cosine similarity matrix
    cos_tensor = cos_tensor.cuda()
    cos_tensor = cos_tensor / torch.norm(cos_tensor, dim=-1, keepdim=True)
    similarity = torch.mm(cos_tensor, cos_tensor.T)

    # Calculate the number of neighbors
    similarity = similarity[:, len(train_neighbor) + len(heldout_neighbor):]
    train_neighbor_result = []  # Store number of each member's neighbors whose cos similarity are higher than the threshold
    heldout_neighbor_result = []  # Store number of each non-member's neighbors whose cos similarity are higher than the threshold

    for i in range(0, len(train_neighbor)):
        neighbor = (similarity[i] >= params.cos_threshold).float()
        sum = torch.sum(neighbor) / len(reference_neighbor)
        train_neighbor_result.append(sum)

    for i in range(len(train_neighbor), len(train_neighbor) + len(heldout_neighbor)):
        neighbor = (similarity[i] >= params.cos_threshold).float()
        sum = torch.sum(neighbor) / len(reference_neighbor)
        heldout_neighbor_result.append(sum)

    train_neighbor_result = torch.tensor(train_neighbor_result)
    heldout_neighbor_result = torch.tensor(heldout_neighbor_result)
    return train_neighbor_result, heldout_neighbor_result


# Generate the training data for mia classifier, assemble the attack information for shadow dataset
def generate_classifer_train_data(params, shadow_path, mask_in, mask_out):
    in_label = get_label(params=params, mask=mask_in)
    out_label = get_label(params=params, mask=mask_out)
    in_label = one_hot(in_label, params)
    out_label = one_hot(out_label, params)

    finaldata_train = []
    finaldata_test = []

    shadow_in_conf, shadow_out_conf = calibrated_loss_attack(params, shadow_path, mask_in, mask_out)

    with torch.no_grad():
        private_in_conf, private_out_conf = get_losses(params, shadow_path, mask_in, mask_out)
        ref_mask = merge_mask(mask_in, mask_out)
        private_in_neighbor, private_out_neighbor, reference_neighbor = get_neighbor(params, mask_in, mask_out, ref_mask)
        train_cos, heldout_cos = neighbor_cos_similarity(params, private_in_neighbor, private_out_neighbor, reference_neighbor)

    # Get the enhanced calibrated membership score
    shadow_in_conf = torch.div(shadow_in_conf, train_cos)
    shadow_out_conf = torch.div(shadow_out_conf, heldout_cos)

    for i in range(0,len(in_label)):
        sample= []
        for data in in_label[i]:
            sample.append(data)
        sample.append(shadow_in_conf[i])
        sample.append(private_in_conf[i])
        sample = torch.tensor(sample).float()
        finaldata_train.append([i, sample, 1])

    for i in range(0,len(out_label)):
        sample = []
        for data in out_label[i]:
            sample.append(data)
        sample.append(shadow_out_conf[i])
        sample.append(private_out_conf[i])
        sample = torch.tensor(sample).float()
        finaldata_test.append([len(in_label)+i, sample, 0])

    return finaldata_train, finaldata_test


if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()

    known_masks, hidden_masks = {}, {}
    hidden_masks['public'], hidden_masks['private'], hidden_masks['shadow'] = {}, {}, {}
    known_masks['public'] = torch.load(params.mask_path + params.dataset + "public.pth")
    known_masks['private'] = torch.load(params.mask_path + params.dataset + "private.pth")
    hidden_masks['private']['train'] = torch.load(params.mask_path + "hidden/" + params.dataset + "train.pth")
    hidden_masks['private']['heldout'] = torch.load(params.mask_path + "hidden/" + params.dataset + "heldout.pth")
    hidden_masks['public']['train'] = torch.load(params.mask_path + "hidden/" + params.dataset + "public_train.pth")
    hidden_masks['shadow']['train'] = torch.load(params.mask_path + "hidden/" + params.dataset + "shadow1_train.pth")
    hidden_masks['shadow']['heldout'] = torch.load(params.mask_path + "hidden/" + params.dataset + "shadow1_heldout.pth")

    # Processing the training data for mia classifier
    print("####### Processing the training data for mia classifier...")
    shadow_dataset_train, shadow_dataset_test = generate_classifer_train_data(params, params.shadow_path, hidden_masks['shadow']['train'], hidden_masks['shadow']['heldout'])
    final_data = []
    for data in shadow_dataset_train:
        final_data.append(data)
    for data in shadow_dataset_test:
        final_data.append(data)

    n_data = len(final_data)
    split_config = {"public": 0.1, "private": {"train": 0.9}}
    known_masks, hidden_masks = generate_masks(n_data, split_config)

    train_data = MaskDataset(final_data, known_masks['private'])
    valid_data = MaskDataset(final_data, known_masks['public'])
    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=params.batch_size)
    validloader = torch.utils.data.DataLoader(valid_data, shuffle=True, batch_size=params.batch_size)

    print("####### Training the mia classifier...")
    # Create logger and print params
    logger = create_logger(params)
    # train the MIA classifier
    params.optimizer = "sgd,lr=0.004,momentum=0.9"
    params.dump_path = params.mia_model_path
    params.architecture = "linear"
    params.num_classes = 2
    params.data_num_dimensions = params.classifier_num_dimensions
    params.epochs = params.classifier_epochs

    model = build_model(params)
    model.cuda()

    trainer = Trainer(model, params, n_data=len(train_data))
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


