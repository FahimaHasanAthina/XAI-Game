import os
import cv2

frames_path = '/home/fa578s/Desktop/CSC790_project/gradcam_exp/tennis/'
frames = os.listdir(frames_path)
video_name = '/home/fa578s/Desktop/CSC790_project/gradcam_exp/videos/tennis_explained.mp4'
frame = cv2.imread(os.path.join(frames_path, frames[0]))
frame = cv2.resize(frame, (256, 256))
height, width, layers = frame.shape

fc = cv2.VideoWriter_fourcc(*"mp4v")

video = cv2.VideoWriter(video_name, fc, 1.5, (height, width))

for i in range(len(frames)):
    img_path = os.path.join(frames_path, frames[i])  
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    video.write(img)


cv2.destroyAllWindows()
video.release()


