import os
import sys
import inspect
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import roc_curve, auc
import sklearn.metrics as metrics
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torch.nn import functional as F
from models import build_model
from utils.misc import bool_flag
from utils.masks import evaluate_masks, merge_mask
from MIA_classifier import get_label
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


def get_parser():
    """
    Generate a parameters parser.
    """
    parser = argparse.ArgumentParser(description='Privacy attack parameters')

    # Attack parameters
    parser.add_argument("--architecture", choices=["smallnet", "linear", "wrn28_10", "densenet121", "mlp", "vgg16"],default="smallnet")
    parser.add_argument("--private_architecture",choices=["smallnet", "linear", "wrn28_10", "densenet121", "mlp", "vgg16"], default="smallnet")
    parser.add_argument("--attack_architecture",choices=["smallnet", "linear", "wrn28_10", "densenet121", "mlp", "vgg16"], default="smallnet")
    parser.add_argument("--dump_path", type=str, default=None)  # reference model path
    parser.add_argument("--model_path", type=str, default="model")  # private model path
    parser.add_argument("--classifier_path", type=str, default="mia_model") # mia classifier path
    parser.add_argument('--classifier_num_dimensions', type=int, default=3)  # number of features for mia classifier
    parser.add_argument("--threshold_low", type=float, default=-10000)  # select the evaluation threshold
    parser.add_argument("--threshold_high", type=float, default=10000)  # select the evaluation threshold
    parser.add_argument("--accurate", type=int, default=10000)  # select the precision of evaluation threshold
    parser.add_argument("--cos_threshold", type=float, default=0)  # select the cosine similarity threshold

    # data parameters
    parser.add_argument("--data_root", type=str, default="data")  # path to the data
    parser.add_argument("--dataset", type=str,choices=["cifar10", "cifar100",  "credit", "adult", "cinic10"], default="cifar10")
    parser.add_argument("--mask_path", type=str, required=True)  # path to the data mask
    parser.add_argument('--data_num_dimensions', type=int, default=3)  # number of features for data
    parser.add_argument("--num_classes", type=int, default=10)  # number of classes for classification task
    parser.add_argument("--in_channels", type=int, default=3)  # number of input channels for image data

    # training parameters
    parser.add_argument("--aug", type=bool_flag, default=False)  # data augmentation flag
    parser.add_argument("--batch_size", type=int, default=32)
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


# Get the membership score feature of LDC-MIA
def get_losses(params, mask_in, mask_out):
    params.architecture = params.private_architecture

    # get the final model parameters
    private_model = build_model(params)
    private_model_path = os.path.join(params.model_path, "checkpoint.pth")
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
    # load the dataset
    dataset = get_dataset(params)

    sub_dataset = Subset(dataset, ids)
    data_loader = DataLoader(sub_dataset, batch_size=64, shuffle=False, pin_memory=True)

    private_model.eval()  # Set the models to evaluation mode
    attack_model.eval()
    all_losses = []

    with torch.no_grad():  # No need to compute gradients
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
            all_losses.extend(calibrated_losses.tolist())  # Convert to list and store

    return torch.tensor(all_losses)


# Get the calibrated membership score of LDC-MIA
def calibrated_loss_attack(params, mask_in, mask_out):
    # load the masks
    known_masks, hidden_masks = {}, {}
    hidden_masks['public'], hidden_masks['private'], hidden_masks['shadow1'] = {}, {}, {}
    known_masks['public'] = torch.load(params.mask_path + params.dataset + "public.pth")
    known_masks['private'] = torch.load(params.mask_path + params.dataset + "private.pth")
    hidden_masks['private']['train'] = torch.load(params.mask_path + "hidden/" + params.dataset + "train.pth")
    hidden_masks['private']['heldout'] = torch.load(params.mask_path + "hidden/" + params.dataset + "heldout.pth")
    hidden_masks['public']['train'] = torch.load(params.mask_path + "hidden/" + params.dataset + "public_train.pth")
    hidden_masks['shadow1']['train'] = torch.load(params.mask_path + "hidden/" + params.dataset + "shadow1_train.pth")
    hidden_masks['shadow1']['heldout'] = torch.load(params.mask_path + "hidden/" + params.dataset + "shadow1_heldout.pth")

    # get the reference model parameters
    params.architecture = params.attack_architecture
    attack_model = build_model(params)
    attack_model_path = os.path.join(params.dump_path, "checkpoint.pth")
    state_dict_attack = torch.load(attack_model_path)
    attack_model.load_state_dict(state_dict_attack['model'])
    attack_model = attack_model.cuda()

    # get the private model parameters
    params.architecture = params.private_architecture
    private_model = build_model(params)
    private_model_path = os.path.join(params.model_path, "checkpoint.pth")
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
    similarity = similarity[:, len(train_neighbor) + len(heldout_neighbor):]  # Can't use the unknown members and non-members to calculate the neighbor
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


# For each target sample(x_target, y_target),
# before inference the target sample on mia classifier,
# integrate all the features [label, enhanced calibrated membership score, membership score] for each target sample
def generate_classifier_test_data(params, cal_loss, loss, mask):
    """
        get the input for MIA_classifier
        generate [label, calibrated membership score, membership score]
    """
    finaldata = []
    label = get_label(params=params, mask=mask)
    label = one_hot(label, params)

    for i in range(0, len(label)):
        sample = []
        for data in label[i]:
            sample.append(data)
        sample.append(cal_loss[i])
        sample.append(loss[i])
        sample = torch.tensor(sample).float()
        finaldata.append(sample)

    return finaldata


# Get the probability that whether the target sample(x_target, y_target) is member(training data of target model)
def get_classifier_confidence(params, model_path, dataset):
    device = torch.device('cpu')

    # get the final model parameters
    params.architecture = "linear"
    params.num_classes = 2
    params.data_num_dimensions = params.classifier_num_dimensions
    private_model = build_model(params)
    private_model_path = os.path.join(model_path, "checkpoint.pth")
    state_dict_private = torch.load(private_model_path, map_location=device)
    private_model.load_state_dict(state_dict_private['model'])
    private_model = private_model.cpu()

    train_images = torch.stack([data for data in dataset])
    train_images = train_images.cpu()

    softmax = torch.nn.Softmax(dim=1)

    train_output = private_model(train_images)
    train_output = softmax(train_output)

    confidence = []
    for i in train_output:
        if torch.argmax(i, dim=0) == 0:
            confidence.append(-i[0])
        else:
            confidence.append(i[1])

    final_confidence = torch.Tensor(confidence)

    return final_confidence


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


def one_hot(x, params):
    return torch.eye(params.num_classes)[x, :]


# Get MIA result(FPR, TPR, ACCURACY, AUC, PRECISION, RECALL)
def get_result(train_vals, heldout_vals, params):
    tpr_list = []
    fpr_list = []
    precision_list = []
    recall_list = []
    acc = 0
    for threshold_test in range(params.threshold_low, params.threshold_high + 1, 1):
        threshold_test = threshold_test / params.accurate
        tpr, fpr, precision, recall, accuracy = evaluate_masks(train_vals, heldout_vals, threshold=threshold_test)
        tpr = tpr.cpu().data.numpy()
        fpr = fpr.cpu().data.numpy()
        precision = precision.cpu().data.numpy()
        recall = recall.cpu().data.numpy()
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        precision_list.append(precision)
        recall_list.append(recall)
        if accuracy >= acc:
            acc = accuracy
    get_auc = auc(fpr_list, tpr_list)
    return fpr_list, tpr_list, acc, get_auc, precision_list, recall_list


if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()

    # load the masks
    known_masks, hidden_masks = {}, {}
    hidden_masks['public'], hidden_masks['private'], hidden_masks['shadow'] = {}, {}, {}
    known_masks['public'] = torch.load(params.mask_path + params.dataset + "public.pth")
    known_masks['private'] = torch.load(params.mask_path + params.dataset + "private.pth")
    hidden_masks['private']['train'] = torch.load(params.mask_path + "hidden/" + params.dataset + "train.pth")
    hidden_masks['private']['heldout'] = torch.load(params.mask_path + "hidden/" + params.dataset + "heldout.pth")
    hidden_masks['public']['train'] = torch.load(params.mask_path + "hidden/" + params.dataset + "public_train.pth")
    hidden_masks['shadow']['train'] = torch.load(params.mask_path + "hidden/" + params.dataset + "shadow1_train.pth")
    hidden_masks['shadow']['heldout'] = torch.load(params.mask_path + "hidden/" + params.dataset + "shadow1_heldout.pth")

    # get the membership score for LDC-MIA
    print("####### Getting the Membership score...")
    train_loss, heldout_loss = get_losses(params, hidden_masks['private']['train'],hidden_masks['private']['heldout'])

    # LDC-MIA
    print("####### Getting the original Calibrated Membership score...")
    train_cals, heldout_cals = calibrated_loss_attack(params, hidden_masks['private']['train'],hidden_masks['private']['heldout'])
    with torch.no_grad():
        # ref_mask = hidden_masks['shadow']['train']
        ref_mask = merge_mask(hidden_masks['shadow']['heldout'], hidden_masks['shadow']['train'])
        print("####### Getting the neighborhood vector...")
        private_in_neighbor, private_out_neighbor, reference_neighbor = get_neighbor(params,hidden_masks['private']['train'],hidden_masks['private']['heldout'], ref_mask)
        print("####### Calculating the neighborhood information...")
        train_cos, heldout_cos = neighbor_cos_similarity(params, private_in_neighbor, private_out_neighbor,reference_neighbor)

    # calculate the enhanced calibrated membership score
    print("####### Calculating the Enhanced Calibrated membership score...")
    train_cals = torch.div(train_cals, train_cos)
    heldout_cals = torch.div(heldout_cals, heldout_cos)

    # get the data of target model that integrating with three features needed by LDC-MIA for mia classifier
    print("####### Assembling the attack information of target sample...")
    in_data = generate_classifier_test_data(params, train_cals, train_loss, hidden_masks['private']['train'])
    out_data = generate_classifier_test_data(params, heldout_cals, heldout_loss, hidden_masks['private']['heldout'])

    # get the results of LOSS ATTACK
    params.threshold_high = int(torch.max(train_loss).item() * 100) + 1
    params.threshold_low = int(torch.min(heldout_loss).item() * 100) - 1
    params.accurate = 100
    _, _, acc_ls, is_auc_ls, _, _ = get_result(train_loss, heldout_loss, params)
    # calculate TPR@FPR
    score = torch.cat((train_loss, heldout_loss), -1)
    zeros_array = np.zeros(len(train_loss), dtype=int)
    ones_array = np.ones(len(heldout_loss), dtype=int)
    y = np.concatenate((zeros_array, ones_array))
    fpr_ls, tpr_ls, _ = metrics.roc_curve(y, score, pos_label=0)
    fpr_ls = np.array(fpr_ls)
    low_ls_001 = tpr_ls[np.where(fpr_ls <= 0.0001)[0][-1]]
    low_ls_01 = tpr_ls[np.where(fpr_ls <= 0.001)[0][-1]]
    low_ls_1 = tpr_ls[np.where(fpr_ls <= 0.01)[0][-1]]

    # get the results of LDC-MIA
    print("####### Inferencing the target samples on MIA classifier...")
    in_prob = get_classifier_confidence(params, params.classifier_path, in_data)
    out_prob = get_classifier_confidence(params, params.classifier_path, out_data)
    params.threshold_high = int(torch.max(in_prob).item() * 10000) + 1
    params.threshold_low = int(torch.min(out_prob).item() * 10000) - 1
    params.accurate = 10000
    _, _, acc, is_auc, _, _ = get_result(in_prob, out_prob, params)
    # calculate TPR@FPR
    score = torch.cat((in_prob, out_prob), -1)
    zeros_array = np.zeros(len(in_prob), dtype=int)
    ones_array = np.ones(len(out_prob), dtype=int)
    y = np.concatenate((zeros_array, ones_array))
    fpr_new, tpr_new, _ = metrics.roc_curve(y, score, pos_label=0)
    fpr_new = np.array(fpr_new)
    low_001 = tpr_new[np.where(fpr_new <= 0.0001)[0][-1]]
    low_01 = tpr_new[np.where(fpr_new <= 0.001)[0][-1]]
    low_1 = tpr_new[np.where(fpr_new <= 0.01)[0][-1]]

    # plot the ROC curve
    print("####### Saving the ROC curve 'ROC.png'...")
    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-4, 1)
    plt.ylim(1e-4, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls='--', color='gray', label='Random guess')
    plt.plot(fpr_new, tpr_new, color='m', label='LDC-MIA')
    plt.plot(fpr_ls, tpr_ls, color='green', label='Loss Attack')
    plt.legend(loc='best')
    plt.savefig("ROC", bbox_inches='tight')
    plt.clf()

    # plot the Precision-Recall curve
    print("####### Saving the Precision-Recall curve 'PR.png'...")
    precision_new = tpr_new / (tpr_new + fpr_new)
    recall_new = tpr_new
    precision_ls = tpr_ls / (tpr_ls + fpr_ls)
    recall_ls = tpr_ls
    plt.xlim(0, 1)
    plt.ylim(0.5, 1)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.plot(recall_new, precision_new, color='m', label='LDC-MIA')
    plt.plot(recall_ls, precision_ls, color='green', label='Loss Attack')
    plt.legend(loc='best')
    plt.savefig("PR", bbox_inches='tight')

    print("LDC-MIA TPR @1%FPR: {}, TPR @0.1%FPR: {}, TPR @0.01%FPR: {} | LOSS ATTACK TPR @1%FPR: {}, TPR @0.1%FPR: {}, TPR @0.01%FPR: {}".format(
            low_1, low_01, low_001, low_ls_1, low_ls_01, low_ls_001
        ))
    print("LDC-MIA accuracy: {} | LOSS ATTACK accuracy: {}".format(acc, acc_ls))
    print("LDC-MIA AUC: {} | LOSS ATTACK AUC: {}".format(is_auc, is_auc_ls))










