import utils.Engine as Engine
import utils.Test as Test
import cv2 # Import opencv
import os

if __name__ == "__main__":
  cur_path = os.getcwd()
  test_path = Test.ReturnTestPath("TestVideo")
  test_list = Test.ReturnTestList(test_path)
  Test.RunTest(test_list)