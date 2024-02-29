from decord import VideoReader
from decord import cpu
from moviepy.editor import *


def read_video_decord(file_path: str = ""):
    with open(file_path, 'rb') as f:
        vr = VideoReader(f, ctx=cpu(0))
    frames = vr[:].asnumpy()
    return frames


def read_video_moviepy(file_path: str = "", fps: int = None):
    # loading video dsa gfg intro video
    clip = VideoFileClip(file_path, audio_fps=16000)
    # new clip with new fps
    if fps is not None:
        clip = clip.set_fps(fps)
    # displaying new clip
    # new_clip.ipython_display(width=360)
    return clip


if __name__ == '__main__':
    read_video_moviepy(file_path=r"D:\These\data\Audio-Visual\RAVDESS\Ravdess-visual\Actor_01\01-01-01-01-01-01-01.mp4",
                       fps=25)
