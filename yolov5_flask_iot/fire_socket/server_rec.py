import socket
import cv2
import pickle
import struct
import time
import os
import threading

ip = 'localhost' # ip 주소
port = 5020 # port 번호

s = socket.socket(socket.AF_INET,socket.SOCK_STREAM) # 소켓 객체를 생성
s.bind((ip, port)) # 바인드(bind) : 소켓에 주소, 프로토콜, 포트를 할당
s.listen(10) # 연결 수신 대기 상태(리스닝 수(동시 접속) 설정)

#----------------------
# image detction
#----------------------



print('S>클라이언트 연결 대기')

# 연결 수락(클라이언트 소켓 주소를 반환)
conn, addr = s.accept()
print(addr) # 클라이언트 주소 출력




def webcam():

    data = b"" # 수신한 데이터를 넣을 변수
    payload_size = struct.calcsize(">L")
    # 파일 저장 위치
    f_dir = "../fire_webserver/static/fire_img"
    f_name =""

    while True:
        for i in range(0,3):
            f_name=f'fire{str(i)}.jpg'
            file_path = os.path.join(f_dir,f_name)

            # 프레임 수신
            while len(data) < payload_size:
                data += conn.recv(4096)
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]
            while len(data) < msg_size:
                data += conn.recv(4096)
            frame_data = data[:msg_size]
            data = data[msg_size:]
            print("S>Frame Size : {}".format(msg_size)) # 프레임 크기 출력

            # 역직렬화(de-serialization) : 직렬화된 파일이나 바이트를 원래의 객체로 복원하는 것
            frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes") # 직렬화되어 있는 binary file로 부터 객체로 역직렬화
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR) # 프레임 디코딩

            #파일저장
            cv2.imwrite(file_path, frame)

            # 영상 출력
            # cv2.imshow('S>TCP_Frame_Socket',frame)

            #time.sleep(0.1)
            # 1초 마다 키 입력 상태를 받음
            if cv2.waitKey(1) == ord('q') : # q를 입력하면 종료
                break

t = threading.Thread(target=webcam)
t.start()
