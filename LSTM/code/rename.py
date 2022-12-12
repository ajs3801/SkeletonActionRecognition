import os
from configs import actions

for action_idx, action in enumerate(actions):
    video_file_path = "./videos/original_videos/{0}".format(action)
    videos = os.listdir(video_file_path)
    for video_idx, video in enumerate(videos):
        src = os.path.join(video_file_path, video)
        new_video_name = "{}-{}.avi".format(action_idx, video_idx)
        dst = os.path.join(video_file_path, new_video_name)
        # print(src)
        # print(dst)
        os.rename(src, dst)
