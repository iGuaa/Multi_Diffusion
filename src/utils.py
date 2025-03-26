import json
import math

import numpy
import torch

####################################################################################################
# 时间变化曲线函数
####################################################################################################

# beta schedule
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
    # return torch.linspace(1, timesteps,timesteps, dtype=torch.float32)

def nb_linear_beta_schedule(timesteps):
    # scale = 1000 / timesteps
    # beta_start = scale * 0.0001
    # beta_end = scale * 0.02
    # return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
    # return torch.linspace(1, timesteps,timesteps, dtype=torch.float32)
    scale = 1000 / timesteps
    beta_start = scale * 0.9
    beta_end = scale * 0.000996
    # return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

def gamma_linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def gaussian_linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.1
    beta_end = scale * 0.2
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def gaussian_v2_linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.9999
    beta_end = scale * 0.98
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)



def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sqrt_beta_schedule(timesteps, s=0.0001):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2205.14217
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = 1 - torch.sqrt(t + s)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


####################################################################################################
# config文件的导入导出函数
####################################################################################################
def export_config(dictionary, filename):
    """
    将字典保存为JSON文件。

    :param dictionary: 要保存的字典
    :param filename: 保存的文件名
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dictionary, f, ensure_ascii=False, indent=4)
        print(f"字典已成功保存为 {filename}")
    except Exception as e:
        print(f"保存文件时出错: {e}")

def import_config(filename):
    """
    将JSON文件导入为字典。

    :param filename: 要导入的JSON文件名
    :return: 导入的字典
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"JSON文件 {filename} 已成功导入为字典")
        return data
    except Exception as e:
        print(f"导入文件时出错: {e}")
        return None

####################################################################################################
# paint 绘图
####################################################################################################


import matplotlib.pyplot as plt
import numpy as np

def plot_images(imgs,img_num=8,time_steps=1000):
    # 画图
    fig = plt.figure(figsize=(img_num*2, img_num+2), constrained_layout=True)
    gs = fig.add_gridspec(img_num*2, img_num+2)
    t_idx = [x*(time_steps//img_num) for x in range(img_num)]
    t_idx[-1] = time_steps - 21
    t_idx.append(time_steps-11)
    t_idx.append(time_steps-1)
    for n_row in range(img_num):
        for n_col in range(img_num+2):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            img = torch.tensor(imgs[t_idx[n_col],n_row]).permute([1,2,0])
            # img = numpy.array((img - img.min()) / (img.max() - img.min()), dtype=numpy.uint8)
            img = numpy.array((img + 1.0) * 255 / 2, dtype=numpy.uint8)
            f_ax.imshow(img)
            f_ax.axis("off")
    for n_row in range(img_num,img_num*2):
        for n_col in range(img_num+2):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            img = torch.tensor(imgs[t_idx[n_col],n_row-img_num]).permute([1,2,0])
            img = numpy.array(((img - img.min()) / (img.max() - img.min()))*255, dtype=numpy.uint8)
            # img = numpy.array((img + 1.0) * 255 / 2, dtype=numpy.uint8)
            f_ax.imshow(img)
            f_ax.axis("off")
    return fig