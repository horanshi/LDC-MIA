import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, TensorDataset
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import pandas as pd
from .text_data import TextIterator


class IdxDataset(Dataset):
    """
    Wraps a dataset so that with each element is also returned its index
    """
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __getitem__(self, i: int):
        sample = self.dataset[i]
        if type(sample) is tuple:
            sample = list(sample)
            sample.insert(0, i)
            return tuple(sample)
        else:
            return i, sample

    def __len__(self):
        return len(self.dataset)


class MaskDataset(Dataset):
    def __init__(self, dataset: Dataset, mask: torch.Tensor):
        """
        example:
        mask: [0, 1, 1]
        cumul: [-1, 0, 1]
        remap: {0: 1, 1: 2}
        """
        assert mask.dim() == 1
        assert mask.size(0) == len(dataset)
        assert mask.dtype == torch.bool

        mask = mask.long()
        cumul = torch.cumsum(mask, dim=0) - 1
        self.remap = {}
        for i in range(mask.size(0)):
            if mask[i] == 1:
                self.remap[cumul[i].item()] = i
            assert mask[i] in [0, 1]

        self.dataset = dataset
        self.mask = mask
        self.length = cumul[-1].item() + 1

    def __getitem__(self, i: int):
        return self.dataset[self.remap[i]]

    def __len__(self):
        return self.length


def adult_data_transform(df):
    binary_data = pd.get_dummies(df)
    feature_cols = binary_data[binary_data.columns[:-2]]
    scaler = preprocessing.StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(feature_cols), columns=feature_cols.columns)
    return data


def get_transform(dataset, aug, is_train):
    if dataset == "cifar10":
        if aug and is_train:
            augmentations = [transforms.RandomCrop(32, padding=4),transforms.RandomRotation(30)]
            normalize = [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            transform = transforms.Compose(augmentations + normalize)
        else:
            transform = transforms.Compose( [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif dataset=='cifar100':
        if aug and is_train:
            augmentations = [transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip()]
            normalize = [transforms.ToTensor(),transforms.Normalize(mean=[n/255 for n in [129.3, 124.1, 112.4]], std=[n/255 for n in [68.2,  65.4,  70.4]])]
            transform = transforms.Compose(augmentations + normalize)
        else:
            transform = transforms.Compose( [transforms.ToTensor(), transforms.Resize((224, 224)), transforms.Normalize(mean=[n/255 for n in [129.3, 124.1, 112.4]], std=[n/255 for n in [68.2,  65.4,  70.4]])])
    elif dataset=='cinic10':
        if aug and is_train:
            augmentations = [transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip()]
            normalize = [transforms.ToTensor(),transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404], std=[0.24205776, 0.23828046, 0.25874835])]
            transform = transforms.Compose(augmentations + normalize)
        else:
            transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404], std=[0.24205776, 0.23828046, 0.25874835])])

    return transform


def get_dataset(*, params, is_train, mask=None):
    if is_train:
        assert mask is not None

    if params.dataset == "cifar10":
        if is_train:
            transform = get_transform(params.dataset, params.aug, True)
        else:
            transform = get_transform(params.dataset, params.aug, False)

        dataset = torchvision.datasets.CIFAR10(root=params.data_root, train=is_train, download=True, transform=transform)
        dataset = IdxDataset(dataset)
        if mask is not None:
            dataset = MaskDataset(dataset, mask)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
        n_data = len(dataset)
        params.num_classes = 10
        return dataloader, n_data

    elif params.dataset=="cinic10":
        if is_train:
            transform = get_transform(params.dataset, params.aug, True)
        else:
            transform = get_transform(params.dataset, params.aug, False)
        if is_train:
            dataset = torchvision.datasets.ImageFolder(root=params.data_root+'/train',transform=transform)
        else:
            dataset = torchvision.datasets.ImageFolder(root=params.data_root+'/valid',transform=transform)
        dataset = IdxDataset(dataset)
        if mask is not None:
            dataset = MaskDataset(dataset, mask)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
        n_data = len(dataset)

        params.num_classes = 10

        return dataloader, n_data

    elif params.dataset=='cifar100':

        if is_train:
            transform = get_transform(params.dataset, params.aug, True)
        else:
            transform = get_transform(params.dataset, params.aug, False)

        dataset = torchvision.datasets.CIFAR100(root=params.data_root, train=is_train, download=True, transform=transform)
        dataset = IdxDataset(dataset)
        
        if mask is not None:
            dataset = MaskDataset(dataset, mask)
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
        
        n_data = len(dataset)

        params.num_classes = 100

        return dataloader, n_data

    elif params.dataset=='credit':
        
        cred=fetch_openml('credit-g', version='1', as_frame=False, parser='liac-arff')
    
        data = SimpleImputer(missing_values=np.nan, strategy='mean', copy=True).fit(cred.data).transform(cred.data)
        target = preprocessing.LabelEncoder().fit(cred.target).transform(cred.target)   
        X=data
        norm = np.max(np.concatenate((-1*X.min(axis=0)[np.newaxis], X.max(axis=0)[np.newaxis]),axis=0).T, axis=1).astype('float32')
        data=np.divide(data,norm)

        data=torch.tensor(data).float()
        target=torch.tensor(target).long()
        if is_train:
            ids=np.arange(1000)[:800]
        else:
            ids=np.arange(1000)[800:]
        
        final_data = []
        for i in ids:
            final_data.append([i,data[i], target[i]])

        params.num_classes = 2
        
        if mask is not None:
            final_data = MaskDataset(final_data, mask)
        dataloader = torch.utils.data.DataLoader(final_data, shuffle=True, batch_size=params.batch_size)
        
        n_data=len(final_data)
        print('Datasize', n_data)

        return dataloader, n_data
    
    elif params.dataset == 'adult':

        columns = ["age", "workClass", "fnlwgt", "education", "education-num","marital-status", "occupation", "relationship","race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
        train_data = pd.read_csv(params.data_root+'/adult.data', names=columns, sep=' *, *', na_values='?', engine='python')
        test_data  = pd.read_csv(params.data_root+'/adult.test', names=columns, sep=' *, *', skiprows=1, na_values='?', engine='python')

        original_train=train_data
        original_test=test_data
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

        test_data=torch.tensor(test_data.to_numpy()).float()
        train_data=torch.tensor(train_data.to_numpy()).float()
        test_labels=torch.tensor(test_labels.to_numpy(dtype='int64')).long()
        train_labels=torch.tensor(train_labels.to_numpy(dtype='int64')).long()
        
        if is_train:
            final_data = []
            for i in np.arange(len(train_data)):
                final_data.append([i,train_data[i], train_labels[i]])
                        
            if mask is not None:
                final_data = MaskDataset(final_data, mask)
            
            dataloader = torch.utils.data.DataLoader(final_data, shuffle=True, batch_size=params.batch_size)
            
            n_data=len(final_data)
        else:
            final_data = []
            for i in np.arange(len(test_data)):
                final_data.append([i,test_data[i], test_labels[i]])
            
            dataloader = torch.utils.data.DataLoader(final_data, batch_size=params.batch_size)
            
            n_data=len(final_data)

            print('Datasize', n_data)

        return dataloader,n_data