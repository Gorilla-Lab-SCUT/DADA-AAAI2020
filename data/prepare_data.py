import os
import shutil
import torch
import scipy.io as scio
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def generate_dataloader(args):
    # Data loading code
    traindir_s = os.path.join(args.source_train_data_path, args.source_domain)
    traindir_t = os.path.join(args.target_train_data_path, args.target_domain)
    valdir_t = os.path.join(args.target_test_data_path, args.target_domain)
    
    if not os.path.isdir(traindir_s):
        raise ValueError('The required data path is not exist, please download the dataset!')

    if args.no_da:
        # transformation on the training data during training
        data_transform_train = transforms.Compose([
      			transforms.Resize((224, 224)), 
      			transforms.ToTensor(),
      			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      	])
        # transformation on the test data during test
        data_transform_test = transforms.Compose([
      			transforms.Resize((224, 224)), 
      			transforms.ToTensor(),
      			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      	])
    else:
        # transformation on the training data during training
        data_transform_train = transforms.Compose([
            transforms.Resize(256),
      			transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
      			transforms.ToTensor(),
      			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      	])
        # transformation on the test data during test
        data_transform_test = transforms.Compose([
      			transforms.Resize(256), 
      			transforms.CenterCrop(224),
      			transforms.ToTensor(),
      			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      	])
       
    source_train_dataset = datasets.ImageFolder(root=traindir_s, transform=data_transform_train)
    source_test_dataset = datasets.ImageFolder(root=traindir_s, transform=data_transform_test)
    target_train_dataset = datasets.ImageFolder(root=traindir_t, transform=data_transform_train)
    target_test_dataset = datasets.ImageFolder(root=valdir_t, transform=data_transform_test)
    
    source_train_loader = torch.utils.data.DataLoader(
        source_train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True
    )
    source_test_loader = torch.utils.data.DataLoader(
        source_test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    target_train_loader = torch.utils.data.DataLoader(
        target_train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True
    )
    target_test_loader = torch.utils.data.DataLoader(
        target_test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )
    
    return source_train_loader, target_train_loader, source_test_loader, target_test_loader

