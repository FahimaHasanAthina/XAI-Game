import random
import numpy as np
import gym

from Pong_master.dqn.agent import DQNAgent
from Pong_master.dqn.replay_buffer import ReplayBuffer
from Pong_master.dqn.wrappers import *
from Pong_master.dqn.model_neurips import DQN
import torch
import os
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

np.random.seed(42)

pong_checkpoint = "/home/fa578s/Desktop/CSC790_project/Pong_master/checkpoint_dqn_neurips.pth"

env = 'PongNoFrameskip-v4'

env = gym.make(env)
env.seed(42)

env = NoopResetEnv(env, noop_max=30)
env = MaxAndSkipEnv(env, skip=4)
env = EpisodicLifeEnv(env)
env = FireResetEnv(env)
env = WarpFrame(env)
env = PyTorchFrame(env)
env = ClipRewardEnv(env)
env = FrameStack(env, 4)
env = gym.wrappers.Monitor(
env, './video/', video_callable=lambda episode_id: episode_id % 50 == 0, force=True)

replay_buffer = ReplayBuffer(int(5e3))

model = DQN(env.observation_space,
            env.action_space)

model.load_state_dict(torch.load(pong_checkpoint))
model.eval()

des_path = '/home/fa578s/Desktop/CSC790_project/gradcam_exp/pong_only/'
frame_path = '/home/fa578s/Desktop/CSC790_project/frames_pong_only/' 
frames = os.listdir(frame_path)

target_layer = [model.conv[-2]]
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
  
    #images = cv2.resize(images, (224,224))
    img_name = os.path.join(des_path, f'explanation_{fr // 4}.png')
    img = Image.fromarray(visualization)
    img.save(img_name)