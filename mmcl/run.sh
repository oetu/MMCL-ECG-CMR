#!/usr/bin/sh

# run the program
# # UNet Imaging
# python run.py version=503 run_simclr=True paths=server_cardiac models=unet max_epochs=300 lr=1e-4 temperature=0.1 batch_size=256 lambda_0=0.5 embedding_dim=128 projection_dim=64

# # UNet MM
# python run.py version=538 run_multimodal=True paths=server_cardiac models=unet max_epochs=300 lr=3e-2 temperature=0.1 batch_size=256 lambda_0=0.25 embedding_dim=128 projection_dim=64 augmentations=default attention_pool=False global_pool=True imaging_pretrain_checkpoint=/vol/aimspace/users/tuo/ECGMultimodalContrastiveLearning/runs/imaging/version_501/checkpoint_best_loss.ckpt ecg_pretrain_checkpoint=/vol/aimspace/users/tuo/ECGMultimodalContrastiveLearning/pretrained_checkpoints/tiny/v1/checkpoint-399.pth
# python run.py version=521 run_multimodal=True paths=server_cardiac models=unet max_epochs=300 lr=3e-4 temperature=0.1 batch_size=256 lambda_0=0.5 embedding_dim=128 projection_dim=64 augmentations=default attention_pool=False global_pool=True imaging_pretrain_checkpoint=/vol/aimspace/users/tuo/ECGMultimodalContrastiveLearning/runs/imaging/version_501/checkpoint_best_loss.ckpt
# python run.py version=510 run_multimodal=True paths=server_cardiac models=unet max_epochs=300 lr=3e-4 temperature=0.1 batch_size=256 lambda_0=0.25 embedding_dim=128 projection_dim=64 augmentations=default attention_pool=False global_pool=True

# # CLOCS ECG
# python run.py version=151 run_simclr_ecg=True paths=server_cardiac max_epochs=300 lr=3e-4 temperature=0.1 batch_size=768 embedding_dim=384 projection_dim=128 augmentations=heavy time_steps=2500 attention_pool=False global_pool=True

# # Multimodal ECG Imaging
# python run.py version=1001 run_multimodal=True paths=server_cardiac save_embeddings=False max_epochs=300 use_softmax=False temp=0.8 lr=1e-4 augmentations=default lambda_0=0.75 embedding_dim=512 projection_dim=128 imaging_pretrain_checkpoint=/vol/aimspace/users/tuo/ECGMultimodalContrastiveLearning/runs/imaging/version_220/checkpoint_best_loss.ckpt attention_pool=False global_pool=True model_size=tiny ecg_pretrain_checkpoint=/vol/aimspace/users/tuo/ECGMultimodalContrastiveLearning/pretrained_checkpoints/tiny/v1/checkpoint-399.pth

# BYOL ECG
python run.py version=401 run_ecg=True loss=byol model=vit paths=tower_cardiac save_embeddings=False max_epochs=300 use_softmax=False temp=0.8 lr=3e-5 augmentations=default attention_pool=False global_pool=True model_size=tiny 

# python run.py version=420 max_epochs=300 use_softmax=False temp=0.8 lr=1e-4 augmentations=default lambda_0=0.75 embedding_dim=512 projection_dim=128 imaging_pretrain_checkpoint=/vol/aimspace/users/tuo/ECGMultimodalContrastiveLearning/runs/imaging/version_220/checkpoint_best_loss.ckpt attention_pool=False global_pool=True model_size=tiny ecg_pretrain_checkpoint=/vol/aimspace/users/tuo/ECGMultimodalContrastiveLearning/pretrained_checkpoints/tiny/v1/checkpoint-399.pth
# python run.py version=422 max_epochs=300 use_softmax=False temp=0.8 lr=3e-4 augmentations=default lambda_0=0.75 embedding_dim=512 projection_dim=128 imaging_pretrain_checkpoint=/vol/aimspace/users/tuo/ECGMultimodalContrastiveLearning/runs/imaging/version_220/checkpoint_best_loss.ckpt attention_pool=False global_pool=True model_size=tiny
# python run.py version=408 max_epochs=300 use_softmax=False temp=0.5 lr=1e-4 augmentations=default lambda_0=0.75 embedding_dim=512 projection_dim=128 imaging_pretrain_checkpoint=/vol/aimspace/users/tuo/ECGMultimodalContrastiveLearning/runs/imaging/version_220/checkpoint_best_loss.ckpt attention_pool=False global_pool=True model_size=tiny
# python run.py version=416 max_epochs=300 use_softmax=False temp=1.0 lr=3e-4 augmentations=default lambda_0=0.75 embedding_dim=512 projection_dim=128 imaging_pretrain_checkpoint=/vol/aimspace/users/tuo/ECGMultimodalContrastiveLearning/runs/imaging/version_220/checkpoint_best_loss.ckpt attention_pool=False global_pool=True model_size=tiny

# python run.py version=277 lr=1e-4 augmentations=heavy lambda_0=0.75 embedding_dim=512 projection_dim=128 imaging_pretrain_checkpoint=/vol/aimspace/users/tuo/ECGMultimodalContrastiveLearning/runs/imaging/version_220/checkpoint_best_loss.ckpt attention_pool=False global_pool=True model_size=tiny ecg_pretrain_checkpoint=/vol/aimspace/users/tuo/ECGMultimodalContrastiveLearning/pretrained_checkpoints/tiny/v1/checkpoint-399.pth
# python run.py version=278 lr=1e-4 augmentations=heavy lambda_0=0.75 embedding_dim=512 projection_dim=128 imaging_pretrain_checkpoint=/vol/aimspace/users/tuo/ECGMultimodalContrastiveLearning/runs/imaging/version_220/checkpoint_best_loss.ckpt attention_pool=True global_pool=False model_size=tiny ecg_pretrain_checkpoint=/vol/aimspace/users/tuo/ECGMultimodalContrastiveLearning/pretrained_checkpoints/tiny/v1/checkpoint-399.pth

# python run.py version=221 batch_size=256 lr=3e-5 lambda_0=0.75 embedding_dim=512 projection_dim=128 global_pool=True model_size=small ecg_pretrain_checkpoint=/vol/aimspace/users/tuo/ECGMultimodalContrastiveLearning/pretrained_checkpoints/small/v1/checkpoint-399.pth
# python run.py version=222 batch_size=256 lr=3e-5 lambda_0=0.75 embedding_dim=512 projection_dim=256 global_pool=True model_size=small ecg_pretrain_checkpoint=/vol/aimspace/users/tuo/ECGMultimodalContrastiveLearning/pretrained_checkpoints/small/v1/checkpoint-399.pth

# python run.py version=224 batch_size=224 lr=3e-5 lambda_0=0.75 embedding_dim=512 projection_dim=128 global_pool=True model_size=medium ecg_pretrain_checkpoint=/vol/aimspace/users/tuo/ECGMultimodalContrastiveLearning/pretrained_checkpoints/medium/v1/checkpoint-399.pth
# python run.py version=225 batch_size=224 lr=3e-5 lambda_0=0.75 embedding_dim=512 projection_dim=256 global_pool=True model_size=medium ecg_pretrain_checkpoint=/vol/aimspace/users/tuo/ECGMultimodalContrastiveLearning/pretrained_checkpoints/medium/v1/checkpoint-399.pth

# python run.py version=220 lr=1e-4 max_epochs=500 warmup_epochs=50 run_multimodal=False run_simclr=True embedding_dim=1024 projection_dim=256

# temperature=(0.3)
# blr=(1e-5 3e-5 1e-4 3e-4 1e-3)
# global_pool=(True)
# version=100

# for temp in "${temperature[@]}"
# do
#     for lr in "${blr[@]}"
#     do
#         for gap in "${global_pool[@]}"
#         do
#             cmd="python run.py version=$version temperature=$temp lr=$lr lambda_0=0.75 augmentations=heavy embedding_dim=512 projection_dim=128 global_pool=$gap imaging_pretrain_checkpoint=/vol/aimspace/projects/ukbb/cardiac/cardiac_segmentations/projects/ecg/weights/weights_imaging.ckpt model_size=tiny ecg_pretrain_checkpoint=/vol/aimspace/users/tuo/ECGMultimodalContrastiveLearning/pretrained_checkpoints/tiny/checkpoint-399.pth"
#             echo $cmd && $cmd

#             # increment version number
#             ((version=version+1))
#         done
#     done
# done