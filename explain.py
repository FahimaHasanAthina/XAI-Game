import sys
sys.path.append('/home/fa578s/Desktop/CSC790_project/atari/')

from atari.play import AtariNet
from atari.ale_env import ALEModern, ALEClassic
import torch
from pathlib import Path
from gzip import GzipFile
from pytorchgradcammaster.pytorch_grad_cam import GradCAM
from pytorchgradcammaster.pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorchgradcammaster.pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os



def _load_checkpoint(fpath, device="cpu"):
    fpath = Path(fpath)
    with fpath.open("rb") as file:
        with GzipFile(fileobj=file) as inflated:
            return torch.load(inflated, map_location=device)

#checkpoint = "/home/fa578s/Desktop/CSC790_project/atari/Pong/DQN_modern/atari_DQN_modern_Pong_2_model_49750000.gz"
#checkpoint = "/home/fa578s/Desktop/CSC790_project/atari/Assault/DQN_modern/atari_DQN_modern_Assault_2_model_49750000.gz"
#checkpoint = "/home/fa578s/Desktop/CSC790_project/atari/Tennis/DQN_modern/atari_DQN_modern_Tennis_1_model_43000000.gz"
checkpoint = "/home/fa578s/Desktop/CSC790_project/atari/Breakout/DQN_modern/atari_DQN_modern_Breakout_2_model_49250000.gz"

ckp_path = Path(checkpoint) 
game = ckp_path.parts[-3]
# print(game)

 # set env
ALE = ALEModern if "_modern/" in checkpoint else ALEClassic
env = ALE(
    game,
    torch.randint(100_000, (1,)).item(),
    sdl=True,
    device="cpu",
    clip_rewards_val=False,
    
    )

#destination path
des_path = '/home/fa578s/Desktop/CSC790_project/pong_explanation/'

# init model
model = AtariNet(env.action_space.n, distributional="C51_" in checkpoint)
# sanity check
print(env)
frame_path = '/home/fa578s/Desktop/CSC790_project/frames_pong/'
frames = os.listdir(frame_path)
sorted(frames)

# load state
ckpt = _load_checkpoint(checkpoint)
model.load_state_dict(ckpt["estimator_state"])
model.eval()
target_layer = [model._AtariNet__features[-2]]
targets = [ClassifierOutputTarget(5)]
i = 0


for fr in range(0, len(frames)):
    name = f'_frame{fr}.jpg'
    rgb_img = cv2.imread(os.path.join(frame_path, name))
    rgb_img = cv2.resize(rgb_img, (84, 84))
    img_rgba = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2RGBA)
    img_rgba = np.float32(img_rgba)/255
    rgb_channel = img_rgba[:, :, :3]
    alpha_channel = torch.tensor(img_rgba[:, :, 3:]).permute(2, 0, 1).unsqueeze(0)
    input_tensor = preprocess_image(rgb_channel,
                                    mean=[0.485, 0.456, 0.406],
                                    std = [0.229, 0.224, 0.255])
    alpha_channel = alpha_channel.permute(1, 0, 2, 3)
    input_tensor = input_tensor.permute(1, 0, 2, 3)
    input_tensor = torch.cat([input_tensor, alpha_channel], dim=0)
    input_tensor = input_tensor.permute(1, 0, 2, 3)
    input_tensor = input_tensor.to(torch.uint8)


    # # #print(model)
    with GradCAM(model=model, target_layers=target_layer) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        print(grayscale_cam)
        visualization = show_cam_on_image(img_rgba, grayscale_cam, use_rgb=True)
        model_outputs = cam.outputs
        

    cam = np.uint8(255*grayscale_cam)
    cam = cv2.merge([cam, cam, cam])
    cam = cv2.cvtColor(cam, cv2.COLOR_RGB2RGBA)

    images = np.hstack((np.uint8(225*img_rgba), cam, visualization))
    ##img = Image.fromarray(images)
    img_name = os.path.join(des_path, f'explanation_{i}.png')
    img = Image.fromarray(visualization)
    img.save(img_name)
    i = i+1


# Added
# for fr in range(0, len(frames), 4):
#     input_images = []
#     for i in range(4):
#         input_img = f'/home/fa578s/Desktop/CSC790_project/frames_pong/_frame{fr + i}.jpg'
#         rgb_img = cv2.imread(input_img)
#         rgb_img = cv2.resize(rgb_img, (84, 84))
#         grayscale_image = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)/255
#         input_images.append(grayscale_image)

#     if len(input_images) != 4:
#         continue

#     input_img = np.stack(input_images, axis=0)
#     input_tensor = torch.tensor(input_img)
#     input_tensor = input_tensor.unsqueeze(0)
#     input_tensor = input_tensor.to(torch.float32)

#     input_tensor = input_tensor.squeeze(0)
#     input = input_tensor.mean(dim=0)

#     with GradCAM(model=model, target_layers=target_layer) as cam:
#         grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
#         grayscale_cam = grayscale_cam[0, :]
#         visualization = show_cam_on_image(img_rgba, grayscale_cam, use_rgb=True)
#         model_outputs = cam.outputs
        
#     cam = np.uint8(255*grayscale_cam)
#     # cam = cv2.merge([cam, cam, cam])
#     cam = cv2.cvtColor(cam, cv2.COLOR_RGB2RGBA)

#     images = np.hstack((np.uint8(225*rgb_img), cam, visualization))
#     img_name = os.path.join(des_path, f'explanation_{fr // 4}.png')
#     img = Image.fromarray(visualization)
#     img.save(img_name)