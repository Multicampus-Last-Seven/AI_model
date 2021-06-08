import time
import os
import threading
import cv2
import numpy
import paho.mqtt.client as mqtt


# on_connect는 subscriber가 브로커에 연결하면서 호출할 함수, rc가 0이면 정상 연결이 됐다는 의미
def on_connect(client, userdata, flags, rc):
    print("connect.." + str(rc))
    if rc == 0:  # 0이 정상 연결 -> 구독신청
        client.subscribe("mydata/fire/#")  # Topic명
    else:
        print("연결 실패")


# 메시지가 도착됐을 때 처리할 일들 - 여러가지 장비 제어하기, 값을 MongoDB에 저장 (ex) led on/off, 문 개폐
def on_message(client, userdata, msg):
    start_time = time.time()
    try:
        data = numpy.frombuffer(msg.payload, dtype=numpy.uint8)  # type - numpy.ndarray
        decimg = cv2.imdecode(data, 1)

        if msg.topic == "mydata/fire/ALR299PNY931":
            filePath = f'./static/fire_img/fire_ALR299PNY931.jpg'
        elif msg.topic == "mydata/fire/BTX586OWE168":
            filePath = f'./static/fire_img/fire_BTX586OWE168.jpg'

        # decimg를 파일로 저장
        cv2.imwrite(filePath, decimg)

        newpid = os.fork()
        if newpid == 0:
            fork_detect(filePath)
        else:
            print(f'parent: {os.getpid()}, child: {newpid}')
            info = os.waitpid(newpid, 0)  # parent process make a child process and wait until it will end
            print(info)
        # cv2.imshow('mqttsub', decimg)
        # cv2.waitKey(33)
    except Exception as e:
        print("에러발생: ", e)
    finally:
        end_time = time.time() - start_time
        print(f'callback function processing time: {end_time}')
    # cv2.destroyAllWindows()


def fork_detect(filePath):
    os.execl('/home/ubuntu/anaconda3/envs/last7/bin/python3', 'python3', 'detect.py', '--source', filePath)
    os._exit(0)
    # '/home/ubuntu/anaconda3/bin/python3'
    # '/bin/python3'
    # python3 detect.py --weights ./best.pt --img 416 --conf 0.5 --source ./test.jpg --save-txt --name result --project . --view-img
    #os.execl('/home/ubuntu/anaconda3/bin/python3', 'python3', 'detect.py', '--weights', './best.pt', '--img', '416', '--conf', '0.5',
    #         '--source', filename, '--save-txt', '--name', 'result', '--project', '.', '--view-img')


# callback함수 : 이벤트가 발생했을 때 실행할 메소드
if __name__ == '__main__':
    mqttClient = mqtt.Client()
    mqttClient.on_connect = on_connect  # 브로커에 연결이 되면 on_connect라는 함수가 실행되도록 등록 (핸들러함수(콜백) 등록)
    mqttClient.on_message = on_message  # 브로커에서 메시지가 전달되면 내가 등록해놓은 on_message함수가 실행
    mqttClient.connect("15.165.185.201", 1883, 60);  # 브로커에 연결하기 # 연결 # keepalive : 브로커와 통신할 최대 시간
    mqttClient.loop_forever()  # 토픽이 전달될 때 까지 수신 대기

