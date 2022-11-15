import os

from numpy import percentile

actions = [
    "squat-down",
    "squat-up",
    "pushup-down",
    "pushup-up",
    "lunge-down",
    "lunge-up",
    "stand",
    "stand2push",
    "push2stand",
]

# print("model test dataset")
# for action in actions:
#     cnt = 0
#     for filename in os.listdir("../videos/model_test_videos/{}".format(action)):
#         if filename.endswith(".DS_Store"):
#             continue
#         cnt += 1
#     print(action, cnt)

# train = []
# print("training dataset")
for action in actions:
    cnt = 0
    for filename in os.listdir("../videos/original_videos/{}".format(action)):
        if filename.endswith(".DS_Store"):
            continue
        cnt += 1
    # train.append(cnt)
    print(action, cnt)

# valid = []
# print("\nvalidation dataset")
# for action in actions:
#     cnt = 0
#     for filename in os.listdir("../videos/{}".format(action)):
#         if filename.endswith(".DS_Store"):
#             continue
#         cnt += 1
#     valid.append(cnt)
#     print(action, cnt)

# all = []
# print("\nall dataset")
# for action in actions:
#     cnt = 0
#     for filename in os.listdir("../valid_videos/{}".format(action)):
#         if filename.endswith(".DS_Store"):
#             continue
#         cnt += 1
#     for filename in os.listdir("../videos/{}".format(action)):
#         if filename.endswith(".DS_Store"):
#             continue
#         cnt += 1
#     all.append(cnt)
#     print(action, cnt)

# print("\nvalidation percentage")
# for i, action in enumerate(actions):
#     percentage = int((valid[i] / all[i]) * 100)
#     print("{}%".format(percentage), action)
