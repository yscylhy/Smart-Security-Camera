import cv2
import sys
from mail import sendEmail
from flask import Flask, render_template, Response
from camera import USBCamera
from flask_basicauth import BasicAuth
import time
import threading
import os
# --- https://github.com/yscylhy/pytorch-ssd.git
sys.path.append('../pytorch-ssd')
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor

email_update_interval = 10 # sends an email only once in this time interval
video_camera = USBCamera(flip=False) # creates a camera object, flip vertically
cur_path = os.path.dirname(os.path.realpath(__file__))

label_path = './models/voc-model-labels.txt'
model_path = './models/mb2-ssd-lite-mp-0_686.pth'
class_names = [name.strip() for name in open(label_path).readlines()]
net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
net.load(model_path)
object_classifier = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200, device='cuda')

# object_classifier = cv2.CascadeClassifier("models/facial_recognition_model.xml") # an opencv classifier



# --- change to your email --- #
with open('email_config.txt') as f:
    fromEmail, fromEmailPassword, toEmail, username, password = f.read().splitlines()


# App Globals (do not edit)
app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = username
app.config['BASIC_AUTH_PASSWORD'] = password
app.config['BASIC_AUTH_FORCE'] = True


basic_auth = BasicAuth(app)
last_epoch = 0


def check_for_objects():
    global last_epoch
    while True:
        try:
            frame, found_obj = video_camera.get_object_pt(object_classifier, class_names)
            # frame, found_obj = video_camera.get_object_cv(object_classifier)
            if found_obj and (time.time() - last_epoch) > email_update_interval:
                last_epoch = time.time()
                print("Sending email...")
                sendEmail(fromEmail, fromEmailPassword, toEmail, frame)
                print("done!")
        except:
            print("Error sending email: ", sys.exc_info()[0])


@app.route('/')
@basic_auth.required
def index():
    return render_template('index.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(video_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    t = threading.Thread(target=check_for_objects, args=())
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', debug=False)

