import os
import sys

#from ..EvaluatePose import EvaluateSquatPose as esp
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from Const import const

def EvaluateDictAction(dict,cur):
  max_key = max(dict,key=dict.get)
  
  if not dict[max_key]:
    return const.NONE_STRING
  
  if (dict[max_key] >= const.THRESHOLD_ACTION):
    if (cur == const.STAND_STRING and (max_key==const.SQUAT_STRING or max_key==const.LUNGE_STRING)):
      return max_key
    elif (cur == const.LYINGE_STRING and max_key==const.PUSHUP_STRING):
      return max_key
  
  else:
    return const.NONE_STRING