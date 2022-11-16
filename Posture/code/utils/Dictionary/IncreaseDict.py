import os
import sys

#from ..EvaluatePose import EvaluateSquatPose as esp
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from Const import const

def IncreaseDict(dict,newaction):
  dict[newaction] += 1