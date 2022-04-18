import os
import sys

import imageio
import numpy as np

import core.utils as utils

class VideoRecorder(object):
    def __init__(self, root_dir, height=256, width=256, camera_id=0, fps=30):
        self.save_dir = utils.make_dir(root_dir, 'video') if root_dir else None
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def new_recorder_init(self, file_name, enabled=True):
        print("init called")
         # create a video writer with imageio
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.timesteps = 0

        if self.enabled:
            path = os.path.join(self.save_dir, file_name)
            print("writing to path", path)
            self.writer = imageio.get_writer(path, mode='I', fps=20)
            #self.writer = imageio.get_writer(path, fps=20)
            #imageio.mimsave(path, self.frames, fps=self.fps)

    def new_record(self, image):
        frame = image[0:3]
        frame = np.transpose(frame, (1, 2, 0))

        if self.enabled:
            self.writer.append_data(frame)

    def simple_record(self, image, flip = True):
        if self.enabled:
            if flip:
                frame = np.flipud(image)
            else:
                frame = image
            self.writer.append_data(frame)

    def clean_up(self):
        if self.enabled:
            print("I'm closing")
            self.writer.close()

