import argparse
import collections
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, collater, Resizer, \
    AspectRatioBasedSampler, Augmenter, Normalizer
from retinanet.eval import Evaluation
    
from torch.utils.data import DataLoader
import multiprocessing
import sys
import logging
import os
import shutil
import datetime

def main(args=None):
    dt_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logging.basicConfig(filename=dt_str+".log")
    log_file = open(dt_str+'.log', 'w')
    sys.stdout = log_file

    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--coco_path', help='Path to COCO directory', default='./data')
    parser.add_argument('--output_path', help='Path to output directory to save checkpoints', default='./output')
    parser.add_argument('--depth', help='ResNet depth, must be one of 18, 34, 50, 101, 152', type=int, default=101)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=72)

    parser.add_argument('--train',  type=str, default='train')
    parser.add_argument('--val', type=str,default='val')

    parser.add_argument('--b_size',type=int,default = 4)
    parser.add_argument('--lr_plan',type=int,default = 1)
    parser.add_argument('--lr_ini',type=float,default = 1e-5)
    parser.add_argument('--lr_decay',type=float,default = 0.1)


    parser = parser.parse_args(args)

    print(f'train_depth{parser.depth}_b_size{parser.b_size}_lr_plan{parser.lr_plan}_lr_decay_{parser.lr_decay}_lr_ini_{parser.lr_ini}')


    if not os.path.exists(parser.output_path):
        os.mkdir(parser.output_path)

    if parser.coco_path is None:
        raise ValueError('Must provide --coco_path when training on COCO.')




    # Train and val data sets 
    dataset_train = CocoDataset(parser.coco_path, set_name=parser.train,
                                transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    dataset_val = CocoDataset(parser.coco_path, set_name=parser.val,
                                transform=transforms.Compose([Normalizer(), Resizer()]))

    # Train sampler & data loader, AspectRatioBasedSampler batch_size=2
    
    # print(dataset_train,dataset_val)
    # quit()
    n_cores = multiprocessing.cpu_count() - 1
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=int(parser.b_size), drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=n_cores, collate_fn=collater, batch_sampler=sampler)

    # Create the model
    if parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    # Set the model to training mode and create an optimizer
    retinanet.training = True
    
    parser.lr_plan = int(parser.lr_plan)
    parser.lr_ini = float(parser.lr_ini)
    parser.lr_decay = float(parser.lr_decay)
    
    # IR - iteration rate - dacay later in lr_scheduler
    optimizer = optim.Adam(retinanet.parameters(), lr=parser.lr_ini)

    # Create a learning rate scheduler
    if parser.lr_plan ==0:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[48, 64])
    elif parser.lr_plan ==1:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor = parser.lr_decay)

    # Initialize loss history and epoch loss list
    loss_hist = collections.deque(maxlen=500)
    epoch_loss_list = []

    print('Num training images: {}'.format(len(dataset_train)))
    for epoch_num in range(parser.epochs):
        
        # Set the model to training mode and freeze batch normalization layers
        retinanet.training = True
        retinanet.train()
        retinanet.freeze_bn()

        epoch_loss = []

        for iter_num, data in tqdm(enumerate(dataloader_train)):
            
            # Zero the gradients of the optimizer
            optimizer.zero_grad()


            # Forward pass through the RetinaNet model to calculate losses
            if torch.cuda.is_available():
                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot'].cuda()])
            else:
                classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])


            # Calculate the total loss  = classification_loss + regression_loss        
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            
            loss = classification_loss + regression_loss

            if bool(loss == 0):
                continue


            # Perform backward pass
            loss.backward()
            
            # Clip the gradients to avoid gradient explosion
            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

            # Optimize the parameters
            optimizer.step()


            # Append the loss to the loss history and epoch loss
            loss_hist.append(float(loss))

            epoch_loss.append(float(loss))

            # Print the loss information every 100 iterations
            if iter_num % 100 == 0:
                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

            del classification_loss
            del regression_loss

        # Update the learning rate
        if parser.lr_plan ==0:
            scheduler.step()
        elif parser.lr_plan ==1:
            scheduler.step(loss)


        # Save the total loss after every epoch
        epoch_loss_list.append(np.mean(epoch_loss))

        if (epoch_num + 1) % 10 == 0 or epoch_num + 1 == parser.epochs:
            print('Evaluating dataset')
            retinanet.eval()
            retinanet.training = False
            eval = Evaluation()
            eval.evaluate(dataset_val, retinanet)
            # The final model is saved.
            torch.save(retinanet, os.path.join(parser.output_path, 'retinanet_epoch{}.pt'.format(epoch_num + 1)))

    print(epoch_loss_list)
    print(parser.output_path)
    torch.save(retinanet, os.path.join(parser.output_path, 'model_final.pt'))
    log_file.close()

if __name__ == '__main__':
    main()
