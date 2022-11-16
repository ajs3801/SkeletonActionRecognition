import os
import sys

#from ..EvaluatePose import EvaluateSquatPose as esp
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from Const import const

def initDict():
  dict = {const.SQUAT_STRING:0,const.PUSHUP_STRING:0,const.LUNGE_STRING:0}
  return dict