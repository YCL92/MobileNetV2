#!/usr/bin/env python
# coding: utf-8

# # Prototype

# ## Includes

# In[ ]:


# mass includes
import time
import pickle
import cv2
import numpy as np


# ## Initialization

# In[ ]:


# setup camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# load pretrained model
model = cv2.dnn.readNet('MobileNetV2.xml', 'MobileNetV2.bin')

# specify target device (VPU)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# load class labels
with open('./labels.pkl', 'rb') as pkl:
    labels = pickle.load(pkl)


# ## Main loop

# In[ ]:


def main():
    fps = 0.0

    # infinit loop
    while True:
        # start time
        start_time = time.time()

        # read a frame
        ret, frame = camera.read()

        # preprocessing
        input_blob = cv2.dnn.blobFromImage(
            frame, size=(224, 224), swapRB=True, crop=True, ddepth=cv2.CV_8U)

        # perform an inference
        model.setInput(input_blob)
        result = model.forward()
        top3 = result.argsort()[0, -3:]

        # mark label to frame
        cv2.putText(frame, 'FPS: %.2f' % fps, (0, 475),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (128, 128, 128), 2)
        for index in range(0, 3):
            label = labels[top3[index]]
            prob = result[0, top3[index]]
            if prob > 0.6:
                color = (0, 255, 0)
            elif prob > 0.3:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            cv2.putText(frame, '(%.2f) %s' % (prob * 100, label),
                        (0, (index + 1) * 30), cv2.FONT_HERSHEY_TRIPLEX, 1,
                        color, 2)

        # show result on screen
        cv2.imshow('Demo (press Q to exit)', frame)

        # conditional exit (press Q)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # update FPS
        fps = 1 / (time.time() - start_time)

    # release the capture
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

