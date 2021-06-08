#Import necessary libraries
from flask import Flask, render_template, Response, request,jsonify
import cv2
import time
import os
import threading
import detect

#Initialize the Flask app
app = Flask(__name__)

#camera = cv2.VideoCapture(0)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# 화재 화면 보여주기
def gen_frames():
    fire_img_dir = "static/fire_img"
    detect_dir = "static/detect"

    while True:
        try:
            for i in range(0,3):
                f_name=f'fire{str(i)}.jpg'
                file_path = os.path.join(fire_img_dir,f_name)
                if not os.path.isfile(file_path)  :
                    continue
                # detect img
                re_alarm,re_labels =detect.detect(file_path)

                #print(f'detect_result> re_alarm: {re_alarm} , re_labels ={re_labels}')


                #화재 발생시 처리
                if re_alarm :
                    print('!!!!! 화재발생')


                # detection result img
                detect_path = os.path.join(detect_dir,f_name)
                img = cv2.imread(detect_path, cv2.IMREAD_COLOR)
                frame = cv2.imencode('.jpg', img)[1].tobytes()

                yield (b'--frame\r\n'
                    b'Content-Type: image/jpg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
                #time.sleep(0)

        except Exception as e:
            print(e)
            continue




if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)


