import os
import argparse
import sys

import numpy as np


from torchvision.utils import save_image

from model.model import PrivateDecoder
from utils.utils import count_params, postprocessing
from tensorboardX import SummaryWriter
import random
import dataset.transform as trans
import dataset.np_transform as np_trans
from torchvision.transforms import Compose
import torchvision.transforms as tr
from dataset.fundus import Fundus_Multi3
from dataset.prostate import Prostate_Multi
import torch.backends.cudnn as cudnn

from torch.nn import BCELoss, CrossEntropyLoss, DataParallel, Upsample
import torch

from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import make_grid
from utils.utils import decode_seg_map_sequence
import shutil
from validation_func import test_fundus, test_prostate, test_fundus_rec
from PIL import Image
import warnings
from network2 import deeplabv3_resnet50

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss


def create_bitmask(mask, bbox):
    per_im_bitmasks_full = []
    w, h = mask.shape[2], mask.shape[3]
    for i in range(mask.shape[0]):
        bitmask_full = torch.zeros((2, mask.shape[2], mask.shape[3])).cuda().float()
        for j in range(mask.shape[1]):
            bitmask_full[j][
            int((bbox[j]["ch"][i] - bbox[j]["h"][i] / 2) * h):int((bbox[j]["ch"][i] + bbox[j]["h"][i] / 2) * h + 1),
            int((bbox[j]["cw"][i] - bbox[j]["w"][i] / 2) * w):int(
                (bbox[j]["cw"][i] + bbox[j]["w"][i] / 2) * w + 1)] = 1.0
        per_im_bitmasks_full.append(bitmask_full)

    gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)
    return gt_bitmasks_full


def create_bitmask_fundus(mask, bbox):
    per_im_bitmasks_full = []
    for i in range(mask.shape[0]):
        bitmask_full = torch.zeros((mask.shape[1], mask.shape[2])).cuda().float()
        bitmask_full[bbox[1][i]:bbox[1][i] + bbox[3][i] + 1,
        bbox[0][i]: bbox[0][i] + bbox[2][i] + 1] = 1.0
        bitmask_full = bitmask_full.unsqueeze(0)
        per_im_bitmasks_full.append(bitmask_full)
    gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)
    return gt_bitmasks_full


def create_bitmask_prostate(mask, bbox):
    per_im_bitmasks_full = []
    for i in range(mask.shape[0]):
        bitmask_full = torch.zeros((mask.shape[1], mask.shape[2])).cuda().float()
        bitmask_full[int(bbox[1][i]):int(bbox[1][i]) + int(bbox[3][i]) + 1,
        int(bbox[0][i]): int(bbox[0][i]) + int(bbox[2][i]) + 1] = 1.0
        bitmask_full = bitmask_full.unsqueeze(0)
        per_im_bitmasks_full.append(bitmask_full)
    gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)
    return gt_bitmasks_full


def compute_project_term(mask_scores, gt_bitmasks):
    ms = torch.empty(0).cuda()
    gt = torch.empty(0).cuda()
    for i in range(mask_scores.shape[1]):
        ms = torch.cat(([ms, mask_scores[:, i, :, :]]))
        gt = torch.cat(([gt, gt_bitmasks[:, i, :, :]]))
    ms = ms.unsqueeze(1)
    gt = gt.unsqueeze(1)

    mask_losses_y = dice_coefficient(
        ms.max(dim=2, keepdim=True)[0],
        gt.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = dice_coefficient(
        ms.max(dim=3, keepdim=True)[0],
        gt.max(dim=3, keepdim=True)[0]
    )
    return (mask_losses_x + mask_losses_y).mean()


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Train')
    # basic settings
    parser.add_argument('--data_root', type=str, default='/data/yhz/', help='root path of training dataset')
    parser.add_argument('--dataset', type=str, default='fundus', choices=['fundus', 'prostate'],
                        help='training dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size of training')
    parser.add_argument('--test_batch_size', type=int, default=8, help='batch size of testing')
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--epochs', type=int, default=None, help='training epochs')
    parser.add_argument('--domain_idxs', type=str, default='0,1,2', help='training epochs')
    parser.add_argument('--test_domain_idx', type=int, default=3, help='training epochs')
    parser.add_argument('--image_size', type=int, default=256, help='cropping size of training samples')
    parser.add_argument('--backbone', type=str, default='mobilenet', help='backbone of semantic segmentation model')
    parser.add_argument('--model', type=str, default=None, help='head of semantic segmentation model')
    parser.add_argument('--num_classes', type=int, default=None, help='number of classes')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pretrained backbone')
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--output_stride', type=int, default=16, help='output stride of deeplab')
    parser.add_argument('--deterministic', action='store_true', help='whether use deterministic training')
    parser.add_argument('--save_path', type=str, default=None, required=True, help='path of saved checkpoints')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

    args = parser.parse_args()
    return args


def train_fundus(trainloader, model, writer, args, optimizer, testloader=None):
    criterion = BCELoss()
    L2Loss = torch.nn.MSELoss()
    model = DataParallel(model).cuda()
    dec = PrivateDecoder(2048, 3, False)
    dec_optim = Adam(dec.parameters(), args.lr, betas=(0.5, 0.999))
    dec = DataParallel(dec).cuda()
    L1Loss = torch.nn.L1Loss()
    total_iters = len(trainloader) * args.epochs
    upsample = Upsample(size=[256, 256], mode='bilinear', align_corners=True)

    previous_best = 0.0
    iter_num = 0
    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.6f" %
              (epoch, optimizer.param_groups[0]["lr"]))
        model.train()
        total_loss = 0.0
        strength = 0.3#*(epoch+1)/args.epochs
        aug = tr.ColorJitter(brightness=strength, contrast=strength, saturation=strength)
        tbar = tqdm(trainloader, ncols=150)

        for i, (img, mask, bbox, idx) in enumerate(tbar):
            img, mask = img.cuda(), mask.cuda()
            gt_mask = create_bitmask(mask, bbox)

            out = aug(img)
            inn = aug(img)
            out2 = aug(img)
            inn2 = aug(img)
            for c in range(3):
                inn[:, c, :, :] = inn[:, c, :, :] * gt_mask[:, 1, :, :]
                out[:, c, :, :] = out[:, c, :, :] * (1 - gt_mask[:, 1, :, :])
                out[:, c, :, :] += inn[:, c, :, :]
                inn2[:, c, :, :] = inn2[:, c, :, :] * gt_mask[:, 1, :, :]
                out2[:, c, :, :] = out2[:, c, :, :] * (1 - gt_mask[:, 1, :, :])
                out2[:, c, :, :] += inn2[:, c, :, :]
            # save_image(img,'1.png')
            # save_image(out,'2.png')
            # save_image(out2,'3.png')
            # sys.exit(0)
            features, pred_hard = model(torch.cat((img, out, out2)))
            _, rec = dec(features['out'])
            pred_soft = torch.sigmoid(pred_hard)
            loss_project = compute_project_term(pred_soft[:8, 1:2, :, :], gt_mask[:, 1:2, :, :])
            loss_project_aug1 = compute_project_term(pred_soft[8:16, 1:2, :, :], gt_mask[:, 1:2, :, :])
            loss_project_aug2 = compute_project_term(pred_soft[16:, 1:2, :, :], gt_mask[:, 1:2, :, :])
            loss_pro = loss_project + (loss_project_aug1 + loss_project_aug2) / 2
            loss_consis = L1Loss(pred_soft[:8, 1:2, :, :], pred_soft[8:16, 1:2, :, :]) + L1Loss(
                pred_soft[:8, 1:2, :, :], pred_soft[16:, 1:2, :, :])

            loss_rec = L1Loss(rec,torch.cat((img,img,img)))
            loss = loss_pro + loss_consis* min(0.5, iter_num / 500) + loss_rec
            if iter_num == 4000:
                save_image(img,'1.png')
                save_image(rec,'2.png')

            dec_optim.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            dec_optim.step()


            # del feature
            lr = args.lr * (1 - iter_num / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            dec_optim.param_groups[0]["lr"] = lr

            iter_num = iter_num + 1

            writer.add_scalar('lr', lr, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)

            if iter_num % 100 == 0:
                image = img[0:3, 0:3, ...]
                grid_image = make_grid(image, 3, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                grid_image = make_grid(pred_soft[0:3, 0, ...].unsqueeze(1), 3, normalize=True)
                writer.add_image('train/Soft_Predicted_OC', grid_image, iter_num)

                grid_image = make_grid(pred_soft[0:3, 1, ...].unsqueeze(1), 3, normalize=True)
                writer.add_image('train/Soft_Predicted_OD', grid_image, iter_num)

                grid_image = make_grid(mask[0:3, 0, ...].unsqueeze(1), 3, normalize=False)
                writer.add_image('train/GT_OC', grid_image, iter_num)

                grid_image = make_grid(mask[0:3, 1, ...].unsqueeze(1), 3, normalize=False)
                writer.add_image('train/GT_OD', grid_image, iter_num)

            tbar.set_description('Loss:%.3f,Pro:%.3f,Consistency:%.3f,Reconstruction:%.3f' % (
                total_loss, loss_pro, loss_consis, loss_rec))

            if iter_num % 500 == 0:
                save_mode_path = os.path.join(args.save_path, 'checkpoints', 'iter_' + str(iter_num) + '.pth')
                torch.save(model.module.state_dict(), save_mode_path)

        if (epoch + 1) % 1 == 0:
            print("Test on target domain {}".format(args.test_domain_idx))
            test_fundus_rec(model, epoch, args.data_root, args.test_domain_idx, args.save_path, args.test_batch_size,
                        dataset=args.dataset)

    save_mode_path = os.path.join(args.save_path, 'final_model.pth')
    torch.save(model.module.state_dict(), save_mode_path)
    print('\nSave Final Model to {}'.format(args.save_path))


def train_prostate(trainloader, model, writer, args, optimizer, testloader=None):
    criterion = CrossEntropyLoss()
    upsample = Upsample(size=[256, 256], mode='bilinear', align_corners=True)
    model = DataParallel(model).cuda()
    dec = PrivateDecoder(2048, 3, False)
    dec_optim = Adam(dec.parameters(), args.lr, betas=(0.5, 0.999))
    dec = DataParallel(dec).cuda()
    total_iters = len(trainloader) * args.epochs
    previous_best = 0.0
    iter_num = 0
    L1Loss = torch.nn.L1Loss()
    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.6f" %
              (epoch, optimizer.param_groups[0]["lr"]))

        model.train()
        total_loss = 0.0
        strength = 0.4 * (epoch + 1) / args.epochs
        aug = tr.ColorJitter(brightness=strength, contrast=strength, saturation=strength)
        tbar = tqdm(trainloader, ncols=150)
        for i, (img, mask, bbox, onehot_label, idx) in enumerate(tbar):
            img,mask = img.cuda(),mask.cuda()
            img += 1
            img /= 2

            gt_mask = create_bitmask_prostate(mask, bbox)

            out = aug(img)
            inn = aug(img)
            out2 = aug(img)
            inn2 = aug(img)
            for c in range(3):
                inn[:, c, :, :] = inn[:, c, :, :] * gt_mask[:, 0, :, :]
                out[:, c, :, :] = out[:, c, :, :] * (1 - gt_mask[:, 0, :, :])
                out[:, c, :, :] += inn[:, c, :, :]
                inn2[:, c, :, :] = inn2[:, c, :, :] * gt_mask[:, 0, :, :]
                out2[:, c, :, :] = out2[:, c, :, :] * (1 - gt_mask[:, 0, :, :])
                out2[:, c, :, :] += inn2[:, c, :, :]
            # save_image(img,'1.png')
            # save_image(out,'2.png')
            # save_image(out2,'3.png')
            # sys.exit(0)
            features, pred_hard = model(torch.cat((img, out, out2)))
            _, rec = dec(features['out'])
            pred_soft = torch.sigmoid(pred_hard)
            loss_project = compute_project_term(pred_soft[:8, 1:2, :, :], gt_mask)
            loss_project_aug1 = compute_project_term(pred_soft[8:16, 1:2, :, :], gt_mask)
            loss_project_aug2 = compute_project_term(pred_soft[16:, 1:2, :, :], gt_mask)
            loss_pro = loss_project + (loss_project_aug1 + loss_project_aug2) / 2
            loss_consis = L1Loss(pred_soft[:8, 1:2, :, :], pred_soft[8:16, 1:2, :, :]) + L1Loss(
                pred_soft[:8, 1:2, :, :], pred_soft[16:, 1:2, :, :])
            loss_rec = L1Loss(rec, torch.cat((img, img, img)))
            loss = loss_pro + loss_consis * min(1.0, iter_num / 500)+loss_rec
            total_loss = loss.item()

            if iter_num == 4000:
                save_image(img, '1.png')
                save_image(rec, '2.png')

            dec_optim.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            dec_optim.step()

            # del feature
            lr = args.lr * (1 - iter_num / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            dec_optim.param_groups[0]["lr"] = lr

            iter_num = iter_num + 1

            writer.add_scalar('lr', lr, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)

            if iter_num % 100 == 0:
                image = img[0:3, 1, ...].unsqueeze(1)
                grid_image = make_grid(image, 3, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = torch.max(pred_soft[0:3, ...], 1)[1].detach().data.cpu().numpy()
                image = decode_seg_map_sequence(image)
                grid_image = make_grid(image, 3, normalize=False)
                writer.add_image('train/Predicted', grid_image, iter_num)

                image = mask[0:3, ...].detach().data.cpu().numpy()
                image = decode_seg_map_sequence(image)
                grid_image = make_grid(image, 3, normalize=False)
                writer.add_image('train/GT', grid_image, iter_num)

            tbar.set_description('Loss:%.3f,Pro:%.3f,Consistency:%.3f,Reconstruction:%.3f' % (
                total_loss, loss_pro, loss_consis, loss_rec))

            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(args.save_path, 'checkpoints', 'iter_' + str(iter_num) + '.pth')
                torch.save(model.module.state_dict(), save_mode_path)

        # if (epoch + 1) % 1 == 0:
        #     print("Test on target domain {}".format(args.test_domain_idx))
        #     test_prostate(model, epoch, args.data_root, args.test_domain_idx, args.save_path, args.test_batch_size, dataset=args.dataset)

    save_mode_path = os.path.join(args.save_path, 'final_model.pth')
    torch.save(model.module.state_dict(), save_mode_path)
    print('\nSave Final Model to {}'.format(args.save_path))


def main(args):
    data_root = os.path.join(args.data_root, args.dataset)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(os.path.join(args.save_path, 'checkpoints')):
        os.mkdir(os.path.join(args.save_path, 'checkpoints'))
    # if os.path.exists(args.save_path + '/code'):
    #     shutil.rmtree(args.save_path + '/code')
    # shutil.copytree('.', args.save_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    writer = SummaryWriter(args.save_path + '/log')

    dataset_zoo = {'fundus': Fundus_Multi3, 'prostate': Prostate_Multi}
    transform = {'fundus': Compose([trans.Resize((256, 256)), trans.Normalize()]),  # trans.RandomScaleCrop((256, 256)),
                 'prostate': Compose([np_trans.CreateOnehotLabel(args.num_classes)])}

    domain_idx_list = args.domain_idxs.split(',')
    domain_idx_list = [int(item) for item in domain_idx_list]

    trainset = dataset_zoo[args.dataset](base_dir=data_root, split='train',
                                         domain_idx_list=domain_idx_list, transform=transform[args.dataset])

    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=8,
                             shuffle=True, drop_last=True, pin_memory=True)

    # model = Unet2D(num_classes=args.num_classes)
    # model = ResNetN(in_dim=3, n_class=args.num_classes, n=50).cuda()
    model = deeplabv3_resnet50(2)

    # model = ResNetN(in_dim=3, n_class=2, n=50)
    optimizer = Adam(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    print('\nParams: %.1fM' % count_params(model))

    if args.dataset == 'fundus':
        train_fundus(trainloader, model, writer, args, optimizer)
    elif args.dataset == 'prostate':
        train_prostate(trainloader, model, writer, args, optimizer)
    else:
        raise ValueError('Not support Dataset {}'.format(args.dataset))


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    if args.epochs is None:
        args.epochs = {'fundus': 80, 'prostate': 30}[args.dataset]
    if args.lr is None:
        args.lr = {'fundus': 1e-4, 'prostate': 1e-4}[args.dataset]
    if args.num_classes is None:
        args.num_classes = {'fundus': 2, 'prostate': 2}[args.dataset]

    print(args)

    main(args)
