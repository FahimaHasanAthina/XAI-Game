import sys
sys.path.append('/home/fa578s/Desktop/CSC790_project/atari/')

from atari.play import AtariNet
from atari.ale_env import ALEModern, ALEClassic
import torch
from pathlib import Path
from gzip import GzipFile
from pytorchgradcammaster.pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise, KPCA_CAM
)
from pytorchgradcammaster.pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorchgradcammaster.pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import gym
from gym.envs.classic_control.rendering import SimpleImageViewer
from Tennis_master.play import wrap_atari, wrap_deepmind, QNetwork


def _load_checkpoint(fpath, device="cpu"):
    fpath = Path(fpath)
    with fpath.open("rb") as file:
        with GzipFile(fileobj=file) as inflated:
            return torch.load(inflated, map_location=device)
    #return torch.load(fpath, map_location='cpu')

checkpoint = "/home/fa578s/Desktop/CSC790_project/atari/Pong/DQN_modern/atari_DQN_modern_Pong_2_model_49750000.gz"
#checkpoint = "/home/fa578s/Desktop/CSC790_project/atari/Assault/DQN_modern/atari_DQN_modern_Assault_2_model_49750000.gz"
#checkpoint = "/home/fa578s/Desktop/CSC790_project/atari/Tennis/DQN_modern/atari_DQN_modern_Tennis_1_model_43000000.gz"
#checkpoint = "/home/fa578s/Desktop/CSC790_project/Tennis_master/runs/best/model-9100000.pth"
#checkpoint = "/home/fa578s/Desktop/CSC790_project/atari/Breakout/DQN_modern/atari_DQN_modern_Breakout_2_model_49250000.gz"
#checkpoint = "/home/fa578s/Desktop/CSC790_project/atari/SpaceInvaders/DQN_modern/atari_DQN_modern_SpaceInvaders_2_model_49500000.gz"

ckp_path = Path(checkpoint) 
game = ckp_path.parts[-3]
print(game)

 # set env
ALE = ALEModern if "_modern/" in checkpoint else ALEClassic
env = ALE(
    game,
    torch.randint(100_000, (1,)).item(),
    sdl=True,
    device="cpu",
    clip_rewards_val=False,
    
    )

# device = torch.device('cpu')
# env = gym.make("TennisNoFrameskip-v4", render_mode='rgb_array')
# env = wrap_atari(env)
# env = gym.wrappers.RecordEpisodeStatistics(env) # records episode reward in `info['episode']['r']`
# env = wrap_deepmind(env,clip_rewards=True,frame_stack=True,scale=False)


#destination path
des_path = '/home/fa578s/Desktop/CSC790_project/gradcam_exp/pong/'

# init model
model = AtariNet(env.action_space.n, distributional="C51_" in checkpoint)
#model = QNetwork(env).to(device)

# sanity check
print(env)
frame_path = '/home/fa578s/Desktop/CSC790_project/frames_pong/'
frames = os.listdir(frame_path)
sorted(frames)

# load state
ckpt = _load_checkpoint(checkpoint)
model.load_state_dict(ckpt["estimator_state"])
#model.load_state_dict(ckpt)
model.eval()
target_layer = [model._AtariNet__features[-2]]
#target_layer = [model.network[-6]]
print(target_layer)
targets = [ClassifierOutputTarget(5)]
i = 0

# Added
for fr in range(0, len(frames)-4, 4):
    input_images = []
    for i in range(4):
        input_img = os.path.join(frame_path, f'_frame{fr + i}.jpg')
        rgb_img = cv2.imread(input_img)
        rgb_img = cv2.resize(rgb_img, (84, 84))
        grayscale_image = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        input_images.append(grayscale_image)

    if len(input_images) != 4:
        break
    
    input_img = np.stack(input_images, axis=0)
    img_rgba = np.float32(input_img)
    img_rgba = np.transpose(img_rgba, (1, 2, 0))

    input_tensor = preprocess_image(img_rgba)
    #input_tensor = torch.tensor(img_rgba).unsqueeze(0)
    input_tensor = input_tensor.to(torch.float32)


    
    methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "gradcamelementwise": GradCAMElementWise,
        'kpcacam': KPCA_CAM
    }
    cam_algorithm = methods['gradcam']

    with cam_algorithm(model=model,target_layers=target_layer) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img/255, grayscale_cam, use_rgb=True)
        model_outputs = cam.outputs
        
    cam = np.uint8(255*grayscale_cam)
    cam = cv2.merge([cam, cam, cam])

    images = np.hstack((np.uint8(rgb_img), cam, visualization))

    # # Perform CAM visualization
    # with cam_algorithm(model=model, target_layers=target_layer) as cam:
    #     grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    #     grayscale_cam = grayscale_cam[0, :]  # Extract the first grayscale CAM

    # # Normalize CAM to range [0, 1]
    # normalized_cam = (grayscale_cam - np.min(grayscale_cam)) / (np.max(grayscale_cam) - np.min(grayscale_cam))

    # # Convert the normalized CAM to a 3-channel mask
    # mask = np.expand_dims(normalized_cam, axis=-1)  # Shape: [H, W, 1]
    # mask = np.repeat(mask, 3, axis=-1)  # Shape: [H, W, 3]

    # # Apply the mask to the original image (keep highlighted parts only)
    # highlighted_image = (rgb_img / 255) * mask  # Scale rgb_img to [0, 1] before multiplying
    # highlighted_image = (highlighted_image * 255).astype(np.uint8)
  
    #images = cv2.resize(images, (224,224))
    img_name = os.path.join(des_path, f'explanation_{fr // 4}.png')
    img = Image.fromarray(visualization)
    img.save(img_name)