# from breakout_master.breakout.dqn import NeuralNetwork
from DQN_PyTorch_Breakout_master.Breakout.DQN_model import DQN
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
import gym
from DQN_PyTorch_Breakout_master.Breakout.transforms import Transforms


# model = NeuralNetwork()
# model = torch.load('breakout_master/breakout/trained_model/current_model_420000.pth', map_location='cpu')
# model.eval()
# Specify environment location
env_name = 'Breakout-v0'

def init_gym_env(env_path):

    env = gym.make(env_path)

    state_space = env.reset().shape
    state_space = (state_space[0], state_space[1], state_space[2])
    state_raw = np.zeros(state_space, dtype=np.uint8)
    processed_state = Transforms.to_gray(state_raw)
    state_space = processed_state.shape
    action_space = env.action_space.n

    return env, state_space, action_space

# Initialize Gym Environment
env, state_space, action_space = init_gym_env(env_name)

env = gym.make('BreakoutDeterministic-v4')

model = DQN(state_space, action_space, filename='breakout_model')
model.load_state_dict(torch.load('/home/fa578s/Desktop/CSC790_project/DQN_PyTorch_Breakout_master/Breakout/models/breakout_model.pth'))
model.eval()
des_path = '/home/fa578s/Desktop/CSC790_project/gradcam_exp/breakout_only/'
frame_path = '/home/fa578s/Desktop/CSC790_project/frames_breakout_only/' 
frames = os.listdir(frame_path)

target_layer = [model.l1[-2]]
print(target_layer)
targets = [ClassifierOutputTarget(1)]
i = 0

# Added
for fr in range(0, len(frames)):
    input_images = []
    
    input_img = os.path.join(frame_path, f'_frame{fr + i}.jpg')
    rgb_img = cv2.imread(input_img)
    rgb_img = cv2.resize(rgb_img, (84, 84))
    grayscale_image = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
   
  
    # img_rgba = np.float32(grayscale_image).unsqueeze(0)
    # print(img_rgba.shape)
    # img_rgba = np.transpose(img_rgba, (1, 2, 0))

    # input_tensor = preprocess_image(img_rgba)
    input_tensor = torch.tensor(grayscale_image).unsqueeze(0)
    input = np.float32(input_tensor)
    input = np.transpose(input, (1, 2, 0))
    input = preprocess_image(input)
    input_tensor = input.to(torch.float32)


    
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
    img_name = os.path.join(des_path, f'explanation_{fr}.png')
    img = Image.fromarray(visualization)
    img.save(img_name)