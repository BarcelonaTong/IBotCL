from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
from loaders.CUB200 import CUB_200
from loaders.ImageNet import ImageNet
from loaders.COVID19 import COVID19
from loaders.matplob import Matplot, MakeImage
import numpy as np
from PIL import Image
import torch
import os


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_train_transformations(args, norm_value):
    aug_list = [
                transforms.Resize((256, 256), Image.BILINEAR),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_value[0], norm_value[1])
                ]
    return transforms.Compose(aug_list)


def get_val_transformations(args, norm_value):
    aug_list = [
                transforms.Resize((256, 256), Image.BILINEAR),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(norm_value[0], norm_value[1])
                ]
    return transforms.Compose(aug_list)


def get_transformations_synthetic():
    aug_list = [
                transforms.Resize((224, 224), Image.BILINEAR),
                transforms.ToTensor(),
                ]
    return transforms.Compose(aug_list)


def get_train_transformations_covid19(args, norm_value):
    aug_list = [
                transforms.Resize((299, 299)),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop((int(299 * 0.8), int(299 * 0.8))),
                transforms.ColorJitter(),
                transforms.ToTensor(),
                transforms.Normalize(norm_value[0], norm_value[1])
                ]
    return transforms.Compose(aug_list)


def get_val_transformations_covid19(args, norm_value):
    aug_list = [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(norm_value[0], norm_value[1])
                ]
    return transforms.Compose(aug_list)


def get_transform(args):
    if args.dataset == "CUB200" or args.dataset == "ImageNet" or args.dataset == "imagenet":
        transform_train = get_train_transformations(args, [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
        transform_val = get_val_transformations(args, [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
        return {"train": transform_train, "val": transform_val}
    elif args.dataset == "matplot":
        transform_train = get_transformations_synthetic()
        transform_val = get_transformations_synthetic()
        return {"train": transform_train, "val": transform_val}
    # COVID-19
    elif args.dataset == "COVID-19":
        transform_train = get_train_transformations_covid19(args, [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
        transform_val = get_val_transformations_covid19(args, [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
        return {"train": transform_train, "val": transform_val}
    raise ValueError(f'unknown {args.dataset}')


def select_dataset(args, transform):
    if args.dataset == "CUB200":
        dataset_train = CUB_200(args, train=True, transform=transform["train"])
        dataset_val = CUB_200(args, train=False, transform=transform["val"])
        return dataset_train, dataset_val
    elif args.dataset == "ImageNet" or args.dataset == "imagenet":
        dataset_train = ImageNet(args, "train", transform=transform["train"])
        dataset_val = ImageNet(args, "val", transform=transform["val"])
        return dataset_train, dataset_val
    elif args.dataset == "matplot":
        data_ = MakeImage().get_img()
        dataset_train = Matplot(data_, "train", transform=transform["train"])
        dataset_val = Matplot(data_, "val", transform=transform["val"])
        return dataset_train, dataset_val
    # COVID-19
    elif args.dataset == "COVID-19":
        dataset_train = COVID19(args, "train", transform=transform["train"])
        dataset_val = COVID19(args, "val", transform=transform["val"])
        return dataset_train, dataset_val
    raise ValueError(f'unknown {args.dataset}')


def loader_generation(args):
    transform = get_transform(args)
    train_set, val_set = select_dataset(args, transform)
    print('Train samples %d - Val samples %d' % (len(train_set), len(val_set)))

    train_loader1 = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=False, drop_last=True)
    train_loader2 = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=False, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                           shuffle=False,
                           num_workers=args.num_workers,
                           pin_memory=False, drop_last=False)
    return train_loader1, train_loader2, val_loader


def load_all_imgs(args):
    def filter(data):
        imgs = []
        labels = []
        for i in range(len(data)):
            root = data[i][0]
            if args.dataset == "matplot":
                ll = data[i][1]
            else:
                ll = int(data[i][1])
            if args.dataset == "CUB200":
                ll -= 1
                root = os.path.join(os.path.join(args.dataset_dir, "CUB_200_2011"), 'images', root)
            imgs.append(root)
            labels.append(ll)
        return imgs, labels

    if args.dataset == "ImageNet" or args.dataset == "imagenet":
        train = ImageNet(args, "train", transform=None).train
        val = ImageNet(args, "train", transform=None).val
        cat = ImageNet(args, "train", transform=None).category
        train_imgs, train_labels = filter(train)
        val_imgs, val_labels = filter(val)
        return train_imgs, train_labels, val_imgs, val_labels, cat
    elif args.dataset == "CUB200":
        train = CUB_200(args)._train_path_label
        val = CUB_200(args)._test_path_label
        cat = np.arange(1, args.num_classes+1, 1)
        train_imgs, train_labels = filter(train)
        val_imgs, val_labels = filter(val)
        return train_imgs, train_labels, val_imgs, val_labels, cat
    elif args.dataset == "matplot":
        data_ = MakeImage().get_img()
        train = data_[0]
        val = data_[1]
        train_imgs, train_labels = filter(train)
        val_imgs, val_labels = filter(val)
        return train_imgs, train_labels, val_imgs, val_labels
    # COVID-19
    elif args.dataset == "COVID-19":
        train = COVID19(args, "train", transform=None).train
        val = COVID19(args, "train", transform=None).val
        cat = COVID19(args, "train", transform=None).category
        train_imgs, train_labels = filter(train)
        val_imgs, val_labels = filter(val)
        return train_imgs, train_labels, val_imgs, val_labels, cat