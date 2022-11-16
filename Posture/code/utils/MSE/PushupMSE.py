import numpy as np
import sys
import os
import cv2
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Const import const

def PushupMSE(image,input_val):
  true_val = np.load("utils/MSE/FMPushupCoord.npy")

  MSE = np.square(np.subtract(true_val,input_val)).mean()
  if (MSE <= 0.02):
    cv2.putText(image, "Pushup : Perfect" , (20,270), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0), 2, cv2.LINE_AA)
    return const.GOOD
  elif (0.02<MSE<=0.05):
    cv2.putText(image, "Pushup : Normal" , (20,270), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    return const.NORMAL
  else:
    cv2.putText(image, "Pushup : Bad" , (20,270), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    return const.BAD