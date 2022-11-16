# import EvalAnswer as EA
import cv2
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import Engine

def EvalAnswer(name):
  strs = name.split('_')
  target = strs[1]

  strs = target.split('.')
  target = strs[0]

  return target

# input으로 test_list를 넣어주면 됨
def RunRandomTest(test_list,MODEL,folder_name):
  notcorrect = []
  correct_list = []
  correct = 0
  for i in test_list:
    print("===========================================================")
    VIDEO_PATH = os.path.join(folder_name,i)
    print("CURRENT VIDEO PATH : {}".format(VIDEO_PATH))
    print("===========================================================")
    cap = cv2.VideoCapture(VIDEO_PATH)
    predict = Engine.TestEngine(cap,MODEL)
    print("Answer  is {}".format(EvalAnswer(i)))
    print("Predict is {}".format(predict))

    if predict == EvalAnswer(i):
      correct+=1
      correct_list.append((i,EvalAnswer(i),predict))
    else:
      notcorrect.append((i,EvalAnswer(i),predict))
  
  print("===========================================================")
  print("### TEST END ###")
  print("# Test Accuracy -> {}%".format(correct/len(test_list)*100))
  print("# Total -> {}/{}".format(correct,len(test_list)))
  print("# Model -> {}".format(MODEL))
  # print("### ANALYSIS correct###")
  # for index,i in enumerate(correct_list):
  #   print("#{} CASE".format(index+1))
  #   print(" -> Videopath : {}".format(i[0]))
  #   print(" -> Answer : {}".format(i[1]))
  #   print(" -> Predict : {}".format(i[2]))
  # print("### ANALYSIS not correct###")
  for index,i in enumerate(notcorrect):
    print("#{} CASE".format(index+1))
    print(" -> Videopath : {}".format(i[0]))
    print(" -> Answer : {}".format(i[1]))
    print(" -> Predict : {}".format(i[2]))
  print("===========================================================")