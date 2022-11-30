# import EvalAnswer as EA
import cv2
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import Engine
def EvalAnswer(name):
  if ("SLP") in name:
    return  "SSLLPP"
  elif ("SPL") in name:
    return "SSPPLL"
  elif ("PSL") in name:
    return "PPSSLL"
  elif ("PLS") in name:
    return "PPLLSS"
  elif ("LSP") in name:
    return "LLSSPP"
  elif ("LPS") in name:
    return "LLPPSS"

# input으로 test_list를 넣어주면 됨
def RunTest(test_list):
  notcorrect = []
  correct = 0
  for i in test_list:
    print("===========================================================")
    VIDEO_PATH = os.path.join("TestVideo",i)
    print("CURRENT VIDEO PATH : {}".format(VIDEO_PATH))
    print("===========================================================")
    cap = cv2.VideoCapture(VIDEO_PATH)
    predict = Engine.ODEngine(cap)
    print("Answer  is {}".format(EvalAnswer(i)))
    print("Predict is {}".format(predict))

    if predict == EvalAnswer(i):
      correct+=1
    else:
      notcorrect.append((i,EvalAnswer(i),predict))
  
  print("===========================================================")
  print("### TEST END ###")
  print("# Test Accuracy -> {}%".format(correct/len(test_list)*100))
  print("# Total -> {}/{}".format(correct,len(test_list)))
  print("### ANALYSIS ###")
  for index,i in enumerate(notcorrect):
    print("#{} CASE".format(index+1))
    print(" -> Videopath : {}".format(i[0]))
    print(" -> Answer : {}".format(i[1]))
    print(" -> Predict : {}".format(i[2]))
  print("===========================================================")