import argparse
import torch
from sympy.stats.sampling.sample_numpy import numpy
from torchvision import datasets
import torchvision.transforms as transforms

from src.gamma_diffusion import GammaDiffusion
# from src.gaussian_diffusion import GaussianDiffusion
# from src.gaussian_nb import GaussianDiffusion
from src.gaussian_diffusion_v2 import GaussianDiffusion
from src.nb_diffusion import NBDiffusion
from src.old_nb_diffusion import OLDNBDiffusion
from src.unet import UNetModel
from src.utils import import_config, plot_images
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO)

####################################################################################################
# 训练
####################################################################################################

def train(config):
    """
    Train
    :param config:
    :return:
    """
    logging.info("Start Train...")

    root_dir = config['root_dir']
    batch_size = config['model_config']['batch_size']
    timesteps = config['model_config']['timesteps']
    beta_schedule = config['model_config']['beta_schedule']
    datasets_type = config['datasets_type']
    epochs = config['epochs']
    group_epoch = config['group_epoch']  # 多少个epoch 为一组 ，用于记录
    RESUME = True if config['resume']=="True" else False
    diffusion_type=config['diffusion_type']

    if diffusion_type == "NB":
        diffusion = NBDiffusion(timesteps=timesteps, beta_schedule=beta_schedule)
    elif diffusion_type == "Gamma":
        logging.debug("Gamma Diffusion Model")
        diffusion = GammaDiffusion(timesteps=timesteps, beta_schedule=beta_schedule)
    elif diffusion_type == "Gaussian":
        diffusion = GaussianDiffusion(timesteps=timesteps, beta_schedule=beta_schedule)
    elif diffusion_type == "OldNB":
        logging.info("OldNB Diffusion Model")
        diffusion = OLDNBDiffusion(timesteps=timesteps, beta_schedule=beta_schedule)
    else:
        diffusion = None
        ValueError("没有这个diffusion类型")

    # 设置训练设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.debug(f"Already used <{device}>")

    # 加载数据集
    if datasets_type == "cifar10":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        datasets.CIFAR10.url = "https://ai-studio-online.bj.bcebos.com/v1/8cf77ffb4c584eaaa716edb69eb0af6541eb532ddc0f4d00bfd7a06b113a2441?responseContentDisposition=attachment%3Bfilename%3Dcifar-10-python.tar.gz&authorization=bce-auth-v1%2F5cfe9a5e1454405eb2a975c43eace6ec%2F2025-01-23T15%3A41%3A37Z%2F21600%2F%2F8ba5a4006db020fa30e061cb18f8f7e93d5d5fce2492c17ac37c4d0f9fd7dcb2"
        dataset = datasets.CIFAR10(rf"{root_dir}/data", train=True, download=True, transform=transform)
        dataset_channel = 3
        dataset_image_size = 32
    elif datasets_type == "mnist":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        dataset = datasets.MNIST(rf"{root_dir}/data", train=True, download=True, transform=transform)
        dataset_channel = 1
        dataset_image_size = 32
    else:
        dataset = None
        dataset_channel = None
        dataset_image_size = None
        ValueError('没有这个数据集类型')

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 创建unet_model
    unet_model = UNetModel(
        in_channels=dataset_channel,
        model_channels=128,
        out_channels=dataset_channel,
        channel_mult=(1, 2, 2, 2),
        attention_resolutions=(2,),
        dropout=0.1
    ).float()

    # 设置优化器
    optimizer = torch.optim.Adam(unet_model.parameters(), lr=2e-4)


    # 是否继续训练
    start_epoch = -1

    if RESUME:
        checkpoint_path = config['checkpoint_path']
        check_point = torch.load(checkpoint_path)
        unet_model.load_state_dict(check_point["model_state_dict"])
        optimizer.load_state_dict(check_point["optimizer_state_dict"])
        start_epoch = check_point["epoch"]

    # 统一张量的device
    unet_model = unet_model.to(device)

    # Unet 训练 nb 噪声
    group_epoch_total_loss = 0
    for epoch in range(start_epoch + 1, epochs):
        per_epoch_total_loss = 0
        for step, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            batch_size = images.shape[0]
            images = images.to(device)

            # sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = diffusion.train_losses(unet_model, images, t)
            per_epoch_total_loss += loss.item()

            # tensorboard记录每个batch的 loss
            writer.add_scalar('Batch Loss/train', loss.item(), epoch * len(train_loader) + step)

            # loss 逆向传播
            loss.backward()
            # 进行梯度下降
            optimizer.step()

        group_epoch_total_loss += per_epoch_total_loss

        # 记录每个epoch的平均loss
        per_epoch_avg_loss = per_epoch_total_loss / len(train_loader)
        writer.add_scalar(rf'Per Epoch Avg Loss/train', per_epoch_avg_loss, epoch)

        if epoch % group_epoch == 0:

            group_epoch_avg_loss = group_epoch_total_loss / group_epoch * len(train_loader)
            writer.add_scalar(rf' Per {group_epoch} Epoch Avg Loss/train', group_epoch_avg_loss, epoch)
            print(f"Epoch {epoch}, Loss {group_epoch_avg_loss}")

            check_point = {
                "epoch": epoch,
                "model_state_dict": unet_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(check_point, rf"{root_dir}/checkpoints/checkpoint_{experiment_name}_{diffusion_type}_{RESUME}_{epoch}.pth")

            # 绘图,采样4张图
            generated_images = torch.tensor(diffusion.sample(unet_model, dataset_image_size, batch_size=64, channels=dataset_channel))
            img_num = 8
            fig = plt.figure(figsize=(img_num * 2, img_num + 2), constrained_layout=True)
            gs = fig.add_gridspec(img_num * 2, img_num + 2)
            t_idx = [x * (timesteps // img_num) for x in range(img_num)]
            t_idx[-1] = timesteps - 21
            t_idx.append(timesteps - 11)
            t_idx.append(timesteps - 1)
            for n_row in range(img_num):
                for n_col in range(img_num + 2):
                    f_ax = fig.add_subplot(gs[n_row, n_col])
                    img = torch.tensor(generated_images[t_idx[n_col], n_row]).permute([1, 2, 0])
                    # img = numpy.array((img - img.min()) / (img.max() - img.min()), dtype=numpy.uint8)
                    img = numpy.array((img + 1.0) * 255 / 2, dtype=numpy.uint8)
                    f_ax.imshow(img)
                    f_ax.axis("off")
            for n_row in range(img_num, img_num * 2):
                for n_col in range(img_num + 2):
                    f_ax = fig.add_subplot(gs[n_row, n_col])
                    img = torch.tensor(generated_images[t_idx[n_col], n_row - img_num]).permute([1, 2, 0])
                    img = numpy.array(((img - img.min()) / (img.max() - img.min())) * 255, dtype=numpy.uint8)
                    # img = numpy.array((img + 1.0) * 255 / 2, dtype=numpy.uint8)
                    f_ax.imshow(img)
                    f_ax.axis("off")
            writer.add_figure(rf"{diffusion_type}_sample_{epoch}.png", fig)
            plt.close()

if __name__ == '__main__':
    # 获取命令行参数
    parser = argparse.ArgumentParser(description="这是一个训练脚本，用于获取训练。")
    parser.add_argument('--config', type=str, required=True, help='输入配置文件的路径')

    args = parser.parse_args()

    if args.config:
        config_path = args.config
    else:
        config_path = rf"C:\Users\31409\PycharmProjects\NB_Diffusioon\configs\train_config.json"

    train_config = import_config(config_path)
    logs_dir = train_config['logs_dir']
    experiment_name = train_config["experiment_name"]
    # tensorboar 记录
    writer = SummaryWriter(rf'{logs_dir}/experiment_{experiment_name}')
    # 开始训练
    train(train_config)
    # 结尾工作
    writer.close()