import utils.Engine as Engine
import cv2 # Import opencv

if __name__ == "__main__":
  MODEL = "model/ActionV7_rf.pkl"
  VIDEO_PATH = 0

  # TESTCAM
  cap = cv2.VideoCapture(VIDEO_PATH)

  Engine.StartEngine(cap)
  Engine.InferenceEngine(cap,MODEL)