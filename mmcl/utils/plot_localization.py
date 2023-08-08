import torch
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from matplotlib import animation

import seaborn as sns
import wandb


def plot_ecg_attention(original_ecg, attentation_map, idx):
    """
    :input:
    original_ecg (B, C, C_sig, T_sig)
    attention_map (B, Heads, C_sig*N_(C_sig), C_sig*N_(C_sig))
    """
    B, C, C_sig, T_sig = original_ecg.shape
    B, Heads, N, N = attentation_map.shape

    NpC = int(N / C_sig) # N_(C_sig)

    # only for nice visualization 
    original_ecg = (original_ecg+0.5*abs(original_ecg.min()))

    # (B, Heads, N_(C_sig), N_(C_sig)), attention map of the first ecg lead
    attentation_map = attentation_map[:, :, 1:(1+NpC), 1:(1+NpC)] # leave the cls token out
    # (B, Heads, N_(C_sig))
    attentation_map = attentation_map.mean(dim=2)
    attentation_map = F.normalize(attentation_map, dim=-1)
    attentation_map = attentation_map.softmax(dim=-1)
    # (B, Heads, T_sig)
    attentation_map = F.interpolate(attentation_map, size=T_sig, mode='linear')

    # (T_sig)
    original_ecg = original_ecg[idx, 0, 0].cpu()
    # (Heads, T_sig)
    attentation_map = attentation_map[idx].cpu()

    fig, axes = plt.subplots(nrows=Heads, sharex=True)

    for head in range(0, Heads):
        axes[head].plot(range(0, original_ecg.shape[-1], 1), original_ecg, zorder=2) # (2500)
        sns.heatmap(attentation_map[head, :].unsqueeze(dim=0).repeat(15, 1), linewidth=0.0, # (1, 2500)
                    alpha=0.3,
                    zorder=1,
                    ax=axes[head])
        axes[head].set_ylim(original_ecg.min(), original_ecg.max())

    # remove y labels of all subplots
    [ax.yaxis.set_visible(False) for ax in axes.ravel()]
    plt.tight_layout()

    wandb.log({"attn_ecg": wandb.Image(fig)})
    plt.close('all')

def plot_image_localization(original_img, importance_img, idx):
    """
    :input: 
    original_img (B, C, H_img, H_img)
    importance_img (B, H'_img, W'_img)
    """
    B, _, H_img, W_img = original_img.shape

    original_img = original_img - original_img.min()
    original_img = original_img / original_img.max()
    original_img = original_img.cpu()

    importance_img = F.interpolate(importance_img.unsqueeze(1), size=(H_img, W_img), mode='bilinear').squeeze(1)
    importance_img = importance_img.cpu()
    img_idx = int(torch.rand(1).item()*32)
    original_img = original_img[idx][1].unsqueeze(dim=0)
    importance_img = importance_img[idx]

    fig, axes = plt.subplots(ncols=2, figsize=(6, 3))

    axes[0].imshow(original_img.permute(1, 2, 0),
                cmap='gray',
                aspect="auto",
                zorder=1)

    axes[1].imshow(original_img.permute(1, 2, 0),
                cmap='gray',
                aspect="auto",
                zorder=1)
    sns.heatmap(importance_img, linewidth=0.0, vmin=-1.0, vmax=1.0, cmap="RdBu_r",
                alpha=0.5,
                zorder=2,
                ax=axes[1])

    plt.tight_layout()
    wandb.log({"loc_img": wandb.Image(fig)})
    plt.close('all')

def plot_ecg_localization(original_ecg, importance_ecg, idx):
    """
    :input: 
    original_ecg (B, C, C_sig, T_sig)
    importance_ecg (B, C_sig, N'_(C_sig))
    """
    B, _, C_sig, T_sig = original_ecg.shape

    # only for nice visualization 
    original_ecg = (original_ecg+0.5*abs(original_ecg.min())).cpu()

    # (B, C_sig, T_sig)
    importance_ecg = F.interpolate(importance_ecg, size=T_sig, mode='linear')
    importance_ecg = importance_ecg.cpu()
    original_ecg = original_ecg[idx]
    importance_ecg = importance_ecg[idx]

    fig, axes = plt.subplots(nrows=13, ncols=2, sharex=True, gridspec_kw={'width_ratios': [40, 1]}, figsize=(8, 12))
    
    for x in range(12):
        axes[x, 0].plot(range(0, original_ecg.shape[-1], 1), original_ecg[0, x, :], zorder=2) # (2500)
        sns.heatmap(importance_ecg[x, :].unsqueeze(dim=0).repeat(15, 1), linewidth=0.0, vmin=-1.0, vmax=1.0, cmap="RdBu_r", # (1, 2500) cmap="RdBu_r"
                    alpha=0.2,
                    zorder=1,
                    ax=axes[x, 0])
        axes[x, 0].set_ylim(original_ecg.min(), original_ecg.max())

    # last row
    axes[12, 0].plot(range(0, original_ecg.shape[-1], 1), torch.zeros(size=(original_ecg.shape[-1],)), zorder=1) # (2500)
    sns.heatmap(importance_ecg[:, :].mean(dim=0, keepdim=True).repeat(15, 1), linewidth=0.0, vmin=-1.0, vmax=1.0, cmap="RdBu_r", # (1, 2500)
                alpha=0.5,
                zorder=2,
                ax=axes[12, 0])
    axes[12, 0].set_ylim(original_ecg.min(), original_ecg.max())

    # last column
    # get the grid of the last column
    gs = axes[0, -1].get_gridspec()
    # remove the underlying axes of the last column
    for ax in axes[:, -1]:
        ax.remove()
    axbig = fig.add_subplot(gs[:-1, -1])

    sns.heatmap(importance_ecg[:, :].mean(dim=-1, keepdim=True), linewidth=0.0, vmin=-1.0, vmax=1.0, cmap="RdBu_r", # (1, 12)
                alpha=0.75,
                ax=axbig)

    # remove y labels of all subplots
    [ax.yaxis.set_visible(False) for ax in axes.ravel()]
    plt.tight_layout()

    wandb.log({"loc_ecg": wandb.Image(fig)})
    plt.close('all')

def plot_pairwise_localization(original_img, original_ecg, importance_pairwise, idx, num_frames: int = 50):
    """
    :input:
    original_img (B, C, H_img, W_img)
    original_ecg (B, C, C_sig, T_sig)
    importance_pairwise (B, N'_(C_sig), H', W')
    
    """
    B, _, C_sig, T_sig = original_ecg.shape
    B, _, H_img, W_img = original_img.shape
    B, NpC, H, W = importance_pairwise.shape # NpC = N_(C_sig)
    # (B, H'*W', N'_(C_sig))
    importance_pairwise = importance_pairwise.flatten(2).permute(0, 2, 1)
    # (B, H'*W', num_frames)
    importance_pairwise = F.interpolate(importance_pairwise, size=num_frames, mode='linear')
    # (B, num_frames, H', W')
    importance_pairwise = importance_pairwise.permute(0, 2, 1).view(B, num_frames, H, W)

    original_img = original_img - original_img.min()
    original_img = original_img / original_img.max()

    # (num_frames, H', W')
    importance_pairwise = importance_pairwise[0].cpu()
    original_img = original_img[idx].cpu()
    original_ecg = original_ecg[idx].cpu()

    fig, (ax_ecg, ax_img) = plt.subplots(nrows=2)
    ecg_2_plot = original_ecg[0, 0, :]
    ax_ecg.set_xlim(0, T_sig)
    ax_ecg.set_ylim(ecg_2_plot.min(), ecg_2_plot.max())
    ax_ecg.plot(range(0, T_sig, 1), ecg_2_plot)
    box_width = T_sig / num_frames
    rect = mpatch.Rectangle((0, ecg_2_plot.min()), box_width, (ecg_2_plot.max()-ecg_2_plot.min()), alpha=0.7, facecolor='gray', zorder=2)
    ax_ecg.add_patch(rect)

    ax_img.imshow(original_img.permute(1, 2, 0),
                    cmap='gray',
                    aspect=1.0,
                    zorder=1)

    importance_data = ax_img.imshow(np.zeros((H_img, W_img)), cmap='inferno', alpha=0.7,
                    zorder=2, vmin=importance_pairwise.min(), vmax=importance_pairwise.max())

    def animate(frame):
        rect.set_xy(((frame / num_frames) * (T_sig - box_width), ecg_2_plot.min()))

        # (H', W')
        importance_frame = importance_pairwise[frame]
        # (H_img, W_img)
        importance_frame = F.interpolate(importance_frame[None, None, :, :], size=(H_img, W_img), mode='bilinear').squeeze()
        importance_frame = importance_frame.cpu()
        
        importance_data.set_data(importance_frame)

        return importance_data

    anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=5, repeat=True)

    path = '/tmp/animation.gif'
    anim.save(path, writer='imagemagick', fps=int(num_frames/15))
    wandb.log({'pairwise_loc': wandb.Video(path, fps=int(num_frames/10), format="gif")})
    plt.close('all')