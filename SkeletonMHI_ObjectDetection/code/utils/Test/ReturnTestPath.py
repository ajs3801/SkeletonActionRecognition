import os

def ReturnTestPath(TestVideo):
  cur_path = os.getcwd()
  test_path = os.path.join(cur_path,TestVideo)

  return test_path