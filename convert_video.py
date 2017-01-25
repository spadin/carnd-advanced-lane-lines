from moviepy.editor import VideoFileClip
from pipeline import Pipeline
import numpy as np

if __name__ == "__main__":
    np.seterr(all='ignore')
    pipeline = Pipeline()

    clip1 = VideoFileClip("./video/project_video.mp4")
    white_clip = clip1.fl_image(pipeline.process_image)
    white_clip.write_videofile("./output/project_video.mp4", audio=False)

