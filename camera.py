import cv2
#from imutils.video.pivideostream import PiVideoStream
#import imutils
import time
import numpy as np

classNames = {0: 'background',
                1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
                7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
                13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird',
                17: 'cat',
                18: 'cat',
                19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
                24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
                32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
                37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
                41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
                46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
                67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
                75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
                80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
                86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

class USBCamera(object):
    def __init__(self, flip=False):
        self.camera = cv2.VideoCapture(-1)
        self.flip = flip
        time.sleep(2.0)

    def __del__(self):
        self.camera.release()

    def flip_if_needed(self, frame):
        if self.flip:
            return np.flip(frame, 0)
        return frame

    def get_frame(self):
        ret, frame = self.camera.read()
        frame = self.flip_if_needed(frame)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def get_object_pt(self, predictor, class_names):

        found_objects = False
        ret, frame = self.camera.read()
        org_frame = self.flip_if_needed(frame)
        # ---- ???? check if color should be changed.
        frame = cv2.cvtColor(org_frame, cv2.COLOR_BGR2RGB)
        boxes, labels, probs = predictor.predict(frame, 10, 0.4)

        # Draw a rectangle around the objects
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            for j in range(4):
                if abs(box[j]) > 10000:
                    break
            else:
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
                # label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
                label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
                cv2.putText(frame, label,
                            (int(box[0]) + 20, int(box[1]) + 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,  # font scale
                            (255, 0, 255),
                            2)  # line type
                found_objects = True

        ret, jpeg = cv2.imencode('.jpg', frame)
        return (jpeg.tobytes(), found_objects)


    def get_object_cv(self, classifier):
        found_objects = False
        ret, frame = self.camera.read()
        frame = self.flip_if_needed(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        objects = classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(objects) > 0:
            found_objects = True

        # Draw a rectangle around the objects
        for (x, y, w, h) in objects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return (jpeg.tobytes(), found_objects)

    def get_object_cnn(self, classifier):
        ret, frame = self.camera.read()
        frame = self.flip_if_needed(frame)

        self.obj_of_interest = False
        image_height, image_width, _ = frame.shape

        classifier.setInput(cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True))
        objects = classifier.forward()

        # -- Draw a rectangle around the objects
        for detection in objects[0, 0, :, :]:
            confidence = detection[2]
            if confidence > .5:
                class_id = detection[1]
                class_name = classNames[class_id]
                if class_name == 'cat':
                    self.obj_of_interest = True
                box_x = detection[3] * image_width
                box_y = detection[4] * image_height
                box_width = detection[5] * image_width
                box_height = detection[6] * image_height
                cv2.rectangle(frame, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (23, 230, 210),
                              thickness=1)
                cv2.putText(frame, class_name, (int(box_x), int(box_y + .05 * image_height)), cv2.FONT_HERSHEY_SIMPLEX,
                            (.002 * image_width), (0, 0, 255))

        time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        cv2.putText(frame, time_stamp,
                    (image_height // 2, image_width // 10), cv2.FONT_HERSHEY_SIMPLEX,
                    (.001 * image_width), (0, 0, 255))

        return frame.tobytes(), self.obj_of_interest




class PiCamera(object):
    def __init__(self, flip = False):
        self.vs = PiVideoStream().start()
        self.flip = flip
        time.sleep(2.0)

    def __del__(self):
        self.vs.stop()

    def flip_if_needed(self, frame):
        if self.flip:
            return np.flip(frame, 0)
        return frame

    def get_frame(self):
        frame = self.flip_if_needed(self.vs.read())
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def get_object(self, classifier):
        found_objects = False
        frame = self.flip_if_needed(self.vs.read()).copy() 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        objects = classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(objects) > 0:
            found_objects = True

        # Draw a rectangle around the objects
        for (x, y, w, h) in objects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return (jpeg.tobytes(), found_objects)


