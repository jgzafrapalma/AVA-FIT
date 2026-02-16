import os
import cv2
import math
import datetime
import subprocess

class FrameExtractor():
    '''
    Class used for extracting frames from a video file.
    '''
    def __init__(self, video_path):
        self.video_path = video_path
        self.vid_cap    = cv2.VideoCapture(video_path)
        self.n_frames   = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps        = int(self.vid_cap.get(cv2.CAP_PROP_FPS))

    def get_video_duration(self):
        duration = self.n_frames/self.fps
        print(f'Duration: {datetime.timedelta(seconds=duration)}')

    def get_n_images(self, every_x_frame):
        n_images = math.floor(self.n_frames / every_x_frame) + 1
        print(f'Extracting every {every_x_frame} (nd/rd/th) frame would result in {n_images} images.')

    # def extract_frames(self, every_x_frame, img_name, dest_path=None, img_ext = '.png', start_frame=1000, end_frame=2000):
    #     if not self.vid_cap.isOpened():
    #         self.vid_cap = cv2.VideoCapture(self.video_path)

    #     if dest_path is None:
    #         dest_path = os.getcwd()
    #     else:
    #         if not os.path.isdir(dest_path):
    #             os.mkdir(dest_path); print(f'Created the following directory: {dest_path}')

    #     frame_cnt = 0; img_cnt = 0
    #     while self.vid_cap.isOpened():
    #         success,image = self.vid_cap.read()
    #         if not success: break
    #         if frame_cnt % every_x_frame == 0 and frame_cnt >= start_frame and (frame_cnt < end_frame or end_frame == -1):
    #             img_path = os.path.join(dest_path, ''.join([img_name,  '%06d' % img_cnt, img_ext]))
    #             cv2.imwrite(img_path, image)
    #             img_cnt += 1
    #         frame_cnt += 1
    #     self.vid_cap.release()
    #     cv2.destroyAllWindows()

    def extract_frames(self, every_x_frame, img_name, dest_path=None, img_ext='.png', start_frame=1000, end_frame=2000):
        if dest_path is None:
            dest_path = os.getcwd()
        else:
            if not os.path.isdir(dest_path):
                os.mkdir(dest_path)
        
        # Calcular fps de salida
        output_fps = self.fps / every_x_frame
        
        # Calcular tiempos de inicio y fin
        start_time = start_frame / self.fps
        
        if end_frame == -1:
            duration_cmd = []
        else:
            duration = (end_frame - start_frame) / self.fps
            duration_cmd = ['-t', str(duration)]
        
        # Comando FFmpeg
        output_pattern = os.path.join(dest_path, f'{img_name}%04d{img_ext}')
        
        cmd = [
            'ffmpeg',
            '-ss', str(start_time),  # Seek al inicio
            *duration_cmd,
            '-i', self.video_path,
            '-vf', f'fps={output_fps}',  # Filtro de fps
            '-start_number', '0',
            output_pattern,
            '-y'  # Sobrescribir
        ]
        
        subprocess.run(cmd, capture_output=True)