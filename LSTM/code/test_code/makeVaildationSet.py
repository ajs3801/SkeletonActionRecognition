import os
import shutil

actions = ['squat-down', 'squat-up', 'pushup-down',
           'pushup-up', 'lunge-down', 'lunge-up']

src = '../test1'
dest = '../test2'

for action in actions:
    for filename in os.listdir(os.path.join(src, action)):
        print(os.path.join(src, action))
        print(action, filename)
        src_path = os.path.join(src, action)
        dest_path = os.path.join(dest, action)
        shutil.move(os.path.join(src_path, filename),
                    os.path.join(dest_path, filename))
