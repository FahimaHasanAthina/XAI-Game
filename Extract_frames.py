import cv2

def FrameCapture(path): 
  
    vidObj = cv2.VideoCapture(path) 
    fps = vidObj.get(cv2.CAP_PROP_FPS)
    frames = 100
    frame_int = int(fps/frames)
    print(fps)
    n = 0 
    path = '/home/fa578s/Desktop/CSC790_project/frames_breakout/'
    #Used as counter variable 
    count = 0
    # checks whether frames were extracted 
    success = 1

    while success: 

        success, image = vidObj.read() 
        if success==False:
            break
        if fps < frames:
            cv2.imwrite("{}_frame{}.jpg".format(path, count), image)
            count += 1
        else:
            if n%frame_int==0:
                # Saves t100 frames per second
                cv2.imwrite("{}_frame{}.jpg".format(path, count), image) 
                count += 1
        n = n+1

#FrameCapture('/home/fa578s/Videos/Screencasts/tennis.mp4')
#FrameCapture('/home/fa578s/Videos/Screencasts/assault.mp4')
FrameCapture('/home/fa578s/Videos/Screencasts/breakout.mp4')