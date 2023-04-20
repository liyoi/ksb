import cv2

# 是否默认启动摄像头
start_video = False


# def recognize_a_frame():
#     capture = cv2.VideoCapture()
#     if capture.isOpened():
#         pass
#     else:
#         view_6 = cv2.imread("view.jpg")
#         view_6_tensor = torch.tensor(view_6)

# predict = model.predict(test_net, view_6_tensor, model.device)
# print(predict)
def pre_image(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img, gray_img


class Capture:
    def __init__(self):
        self.capture = None
        if start_video:
            self.capture = cv2.VideoCapture(0)
            # capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
            if not self.capture.isOpened():
                print("摄像机打开失败")
            print("摄像机打开成功")
            self.get_video_frame()

    def get_video_frame(self):
        if (self.capture == None):
            return
        while True:
            ret, frame = self.capture.read()  # 读取图像
            # cv2.namedWindow("frame")
            cv2.imshow('frame', frame)
            cv2.imwrite("../data/aa.jpg", frame)
            mykey = cv2.waitKey(0)
            if mykey == 'q':
                self.close()
                break
            print(self.capture.isOpened())

    def get_a_frame(self):
        if self.capture is None:
            self.open_capture()
        self.capture.read()  # 读取图像
        ret, frame = self.capture.read()
        # cv2.namedWindow("frame")
        # cv2.imshow('frame', frame)
        cv2.imwrite("../data/aa.jpg", frame)
        print(self.capture.isOpened())

    # 打开摄像头
    def open_capture(self):
        if self.capture is None:
            self.capture = self.capture = cv2.VideoCapture(0)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

    def close(self):
        if self.capture is None or (not self.capture.isOpened()):
            print("摄像机没有打开,不需要关闭")
            return

        self.capture.release()
