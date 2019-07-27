import sys
import threading
import cv2
import numpy as np
import matplotlib.pyplot as plt


class UnsupportedColorspaceException(Exception):
    pass


class Handler:
    def __init__(self, img_path, window_name):
        self.img_path = img_path
        self.window_name = window_name

        self.img = cv2.imread(img_path, 1)  # Read as BGR image
        self.masked_img = self.img.copy()
        self.colorspace = 'bgr'
        self.channel_0_min = 0
        self.channel_1_min = 0
        self.channel_2_min = 0
        self.channel_0_max = 255
        self.channel_1_max = 255
        self.channel_2_max = 255

        self.alive = True
        self.update = False
        self.thread = threading.Thread(target=self.gui_handler)
        self.thread.start()
        self.show_image()

    def show_image(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.tick_params(bottom=False, left=False, labelleft=False, labelbottom=False)
        ax2.tick_params(bottom=False, left=False, labelleft=False, labelbottom=False)
        while self.alive:
            if self.update:
                ax1.clear()
                ax1.imshow(self.img)
                ax1.set(title=self.colorspace.upper())
                ax2.clear()
                ax2.imshow(self.masked_img)
                ax2.set(title='Masked image')
                self.update = False
            plt.pause(0.00001)
        plt.close()

    def update_mask(self):
        lower = np.array([self.channel_0_min, self.channel_1_min, self.channel_2_min])
        upper = np.array([self.channel_0_max, self.channel_1_max, self.channel_2_max])
        mask = cv2.inRange(self.img, lower, upper)
        self.masked_img = cv2.bitwise_and(self.img, self.img, mask=mask)
        self.update = True
        cv2.imshow(self.window_name, mask)

    def change_colorspace(self, colorspace):
        lowercase = colorspace.lower()  # Make sure case doesn't matter
        if self.colorspace == lowercase:
            return
        if self.colorspace == 'bgr':
            if lowercase == 'rgb':
                conversion = cv2.COLOR_BGR2RGB
            elif lowercase == 'hsv':
                conversion = cv2.COLOR_BGR2HSV
            elif lowercase == 'hls':
                conversion = cv2.COLOR_BGR2HLS
            elif lowercase == 'lab':
                conversion = cv2.COLOR_BGR2LAB
            elif lowercase == 'ycrcb':
                conversion = cv2.COLOR_BGR2YCrCb
            elif lowercase == 'luv':
                conversion = cv2.COLOR_BGR2LUV
            elif lowercase == 'xyz':
                conversion = cv2.COLOR_BGR2XYZ
            elif lowercase == 'yuv':
                conversion = cv2.COLOR_BGR2YUV
            else:
                raise UnsupportedColorspaceException('Unsupported colorspace {}'.format(lowercase))
        elif self.colorspace == 'rgb':
            if lowercase == 'bgr':
                conversion = cv2.COLOR_RGB2BGR
            elif lowercase == 'hsv':
                conversion = cv2.COLOR_RGB2HSV
            elif lowercase == 'hls':
                conversion = cv2.COLOR_RGB2HLS
            elif lowercase == 'lab':
                conversion = cv2.COLOR_RGB2LAB
            elif lowercase == 'ycrcb':
                conversion = cv2.COLOR_RGB2YCrCb
            elif lowercase == 'luv':
                conversion = cv2.COLOR_RGB2LUV
            elif lowercase == 'xyz':
                conversion = cv2.COLOR_RGB2XYZ
            elif lowercase == 'yuv':
                conversion = cv2.COLOR_RGB2YUV
            else:
                raise UnsupportedColorspaceException('Unsupported colorspace {}'.format(lowercase))
        elif self.colorspace == 'hsv':
            if lowercase == 'bgr':
                conversion = cv2.COLOR_HSV2BGR
            elif lowercase == 'rgb':
                conversion = cv2.COLOR_HSV2RGB
            elif lowercase == 'hls':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_HSV2BGR)
                conversion = cv2.COLOR_BGR2HLS
            elif lowercase == 'lab':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_HSV2BGR)
                conversion = cv2.COLOR_BGR2LAB
            elif lowercase == 'ycrcb':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_HSV2BGR)
                conversion = cv2.COLOR_BGR2YCrCb
            elif lowercase == 'luv':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_HSV2BGR)
                conversion = cv2.COLOR_BGR2LUV
            elif lowercase == 'xyz':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_HSV2BGR)
                conversion = cv2.COLOR_BGR2XYZ
            elif lowercase == 'yuv':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_HSV2BGR)
                conversion = cv2.COLOR_BGR2YUV
            else:
                raise UnsupportedColorspaceException('Unsupported colorspace {}'.format(lowercase))
        elif self.colorspace == 'hls':
            if lowercase == 'bgr':
                conversion = cv2.COLOR_HLS2BGR
            elif lowercase == 'rgb':
                conversion = cv2.COLOR_HLS2RGB
            elif lowercase == 'hsv':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_HLS2BGR)
                conversion = cv2.COLOR_BGR2HSV
            elif lowercase == 'lab':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_HLS2BGR)
                conversion = cv2.COLOR_BGR2LAB
            elif lowercase == 'ycrcb':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_HLS2BGR)
                conversion = cv2.COLOR_BGR2YCrCb
            elif lowercase == 'luv':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_HLS2BGR)
                conversion = cv2.COLOR_BGR2LUV
            elif lowercase == 'xyz':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_HLS2BGR)
                conversion = cv2.COLOR_BGR2XYZ
            elif lowercase == 'yuv':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_HLS2BGR)
                conversion = cv2.COLOR_BGR2YUV
            else:
                raise UnsupportedColorspaceException('Unsupported colorspace {}'.format(lowercase))
        elif self.colorspace == 'lab':
            if lowercase == 'bgr':
                conversion = cv2.COLOR_LAB2BGR
            elif lowercase == 'rgb':
                conversion = cv2.COLOR_LAB2RGB
            elif lowercase == 'hls':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_LAB2BGR)
                conversion = cv2.COLOR_BGR2HLS
            elif lowercase == 'hsv':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_LAB2BGR)
                conversion = cv2.COLOR_BGR2HSV
            elif lowercase == 'ycrcb':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_LAB2BGR)
                conversion = cv2.COLOR_BGR2YCrCb
            elif lowercase == 'luv':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_LAB2BGR)
                conversion = cv2.COLOR_BGR2LUV
            elif lowercase == 'xyz':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_LAB2BGR)
                conversion = cv2.COLOR_BGR2XYZ
            elif lowercase == 'yuv':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_LAB2BGR)
                conversion = cv2.COLOR_BGR2YUV
            else:
                raise UnsupportedColorspaceException('Unsupported colorspace {}'.format(lowercase))
        elif self.colorspace == 'ycrcb':
            if lowercase == 'bgr':
                conversion = cv2.COLOR_YCrCb2BGR
            elif lowercase == 'rgb':
                conversion = cv2.COLOR_YCrCb2RGB
            elif lowercase == 'hls':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_YCrCb2BGR)
                conversion = cv2.COLOR_BGR2HLS
            elif lowercase == 'hsv':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_YCrCb2BGR)
                conversion = cv2.COLOR_BGR2HSV
            elif lowercase == 'lab':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_YCrCb2BGR)
                conversion = cv2.COLOR_BGR2LAB
            elif lowercase == 'luv':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_YCrCb2BGR)
                conversion = cv2.COLOR_BGR2LUV
            elif lowercase == 'xyz':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_YCrCb2BGR)
                conversion = cv2.COLOR_BGR2XYZ
            elif lowercase == 'yuv':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_YCrCb2BGR)
                conversion = cv2.COLOR_BGR2YUV
            else:
                raise UnsupportedColorspaceException('Unsupported colorspace {}'.format(lowercase))
        elif self.colorspace == 'luv':
            if lowercase == 'bgr':
                conversion = cv2.COLOR_LUV2BGR
            elif lowercase == 'rgb':
                conversion = cv2.COLOR_LUV2RGB
            elif lowercase == 'hls':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_LUV2BGR)
                conversion = cv2.COLOR_BGR2HLS
            elif lowercase == 'hsv':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_LUV2BGR)
                conversion = cv2.COLOR_BGR2HSV
            elif lowercase == 'lab':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_LUV2BGR)
                conversion = cv2.COLOR_BGR2LAB
            elif lowercase == 'ycrcb':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_LUV2BGR)
                conversion = cv2.COLOR_BGR2YCrCb
            elif lowercase == 'xyz':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_LUV2BGR)
                conversion = cv2.COLOR_BGR2XYZ
            elif lowercase == 'yuv':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_LUV2BGR)
                conversion = cv2.COLOR_BGR2YUV
            else:
                raise UnsupportedColorspaceException('Unsupported colorspace {}'.format(lowercase))
        elif self.colorspace == 'xyz':
            if lowercase == 'bgr':
                conversion = cv2.COLOR_XYZ2BGR
            elif lowercase == 'rgb':
                conversion = cv2.COLOR_XYZ2RGB
            elif lowercase == 'hls':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_XYZ2BGR)
                conversion = cv2.COLOR_BGR2HLS
            elif lowercase == 'hsv':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_XYZ2BGR)
                conversion = cv2.COLOR_BGR2HSV
            elif lowercase == 'lab':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_XYZ2BGR)
                conversion = cv2.COLOR_BGR2LAB
            elif lowercase == 'ycrcb':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_XYZ2BGR)
                conversion = cv2.COLOR_BGR2YCrCb
            elif lowercase == 'luv':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_XYZ2BGR)
                conversion = cv2.COLOR_BGR2LUV
            elif lowercase == 'yuv':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_XYZ2BGR)
                conversion = cv2.COLOR_BGR2YUV
            else:
                raise UnsupportedColorspaceException('Unsupported colorspace {}'.format(lowercase))
        elif self.colorspace == 'yuv':
            if lowercase == 'bgr':
                conversion = cv2.COLOR_YUV2BGR
            elif lowercase == 'rgb':
                conversion = cv2.COLOR_YUV2RGB
            elif lowercase == 'hls':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_YUV2BGR)
                conversion = cv2.COLOR_BGR2HLS
            elif lowercase == 'hsv':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_YUV2BGR)
                conversion = cv2.COLOR_BGR2HSV
            elif lowercase == 'lab':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_YUV2BGR)
                conversion = cv2.COLOR_BGR2LAB
            elif lowercase == 'ycrcb':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_YUV2BGR)
                conversion = cv2.COLOR_BGR2YCrCb
            elif lowercase == 'luv':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_YUV2BGR)
                conversion = cv2.COLOR_BGR2LUV
            elif lowercase == 'xyz':
                self.img = cv2.cvtColor(self.img, cv2.COLOR_YUV2BGR)
                conversion = cv2.COLOR_BGR2XYZ
            else:
                raise UnsupportedColorspaceException('Unsupported colorspace {}'.format(lowercase))
        else:  # Should never reach this statement
            raise UnsupportedColorspaceException('Unsupported colorspace {}'.format(self.colorspace))

        self.img = cv2.cvtColor(self.img, conversion)
        self.colorspace = lowercase
        self.update_mask()

    def on_trackbar_channel_0_min(self, val):
        self.channel_0_min = val
        self.update_mask()

    def on_trackbar_channel_1_min(self, val):
        self.channel_1_min = val
        self.update_mask()

    def on_trackbar_channel_2_min(self, val):
        self.channel_2_min = val
        self.update_mask()

    def on_trackbar_channel_0_max(self, val):
        self.channel_0_max = val
        self.update_mask()

    def on_trackbar_channel_1_max(self, val):
        self.channel_1_max = val
        self.update_mask()

    def on_trackbar_channel_2_max(self, val):
        self.channel_2_max = val
        self.update_mask()

    def initialize_trackbars(self):
        cv2.createTrackbar('0 min', self.window_name, 0, 255, self.on_trackbar_channel_0_min)
        cv2.createTrackbar('0 max', self.window_name, 255, 255, self.on_trackbar_channel_0_max)
        cv2.createTrackbar('1 min', self.window_name, 0, 255, self.on_trackbar_channel_1_min)
        cv2.createTrackbar('1 max', self.window_name, 255, 255, self.on_trackbar_channel_1_max)
        cv2.createTrackbar('2 min', self.window_name, 0, 255, self.on_trackbar_channel_2_min)
        cv2.createTrackbar('2 max', self.window_name, 255, 255, self.on_trackbar_channel_2_max)

    def gui_handler(self):
        cv2.namedWindow(self.window_name)
        self.initialize_trackbars()
        self.update_mask()
        while True:
            k = cv2.waitKey()
            if k == ord('1'):
                self.change_colorspace('bgr')
            elif k == ord('2'):
                self.change_colorspace('rgb')
            elif k == ord('3'):
                self.change_colorspace('hsv')
            elif k == ord('4'):
                self.change_colorspace('hls')
            elif k == ord('5'):
                self.change_colorspace('lab')
            elif k == ord('6'):
                self.change_colorspace('ycrcb')
            elif k == ord('7'):
                self.change_colorspace('luv')
            elif k == ord('8'):
                self.change_colorspace('xyz')
            elif k == ord('9'):
                self.change_colorspace('yuv')
            elif k == ord('q'):
                print('Final values')
                print('--------------------')
                print('Channel 0 min: {}'.format(self.channel_0_min))
                print('Channel 0 max: {}'.format(self.channel_0_max))
                print('Channel 1 min: {}'.format(self.channel_1_min))
                print('Channel 1 max: {}'.format(self.channel_1_max))
                print('Channel 2 min: {}'.format(self.channel_2_min))
                print('Channel 2 max: {}'.format(self.channel_2_max))
                print('--------------------')
                print('Easy copy for cv2.inRange')
                print('Lower')
                print('[{}, {}, {}]'.format(self.channel_0_min, self.channel_1_min, self.channel_2_min))
                print('Upper')
                print('[{}, {}, {}]'.format(self.channel_0_max, self.channel_1_max, self.channel_2_max))
                cv2.destroyAllWindows()
                self.alive = False
                break


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python visualization.py <img_path>')
        exit(1)
    img_path = sys.argv[1]

    handler = Handler(img_path, 'Editing mask')
