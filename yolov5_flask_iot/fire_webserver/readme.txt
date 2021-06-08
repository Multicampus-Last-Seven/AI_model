1.네트워크 소켓 통신 

 1)소켓통신 
  - 클라이언트에서 webcam을 자동실행해서 webcam 이미지를  서버로 소켓전송
  - 서버를 클라이언트에서 받은 이미지를  fire_detection/static/fire_img/ 에 저장 
  2)파일 
    fire_socket > server_rec.py 
    fire_socket > client_snd.py 
  3) 실행하기  
    (1) ip config 변경   
     * (windows) >ipconfig  명령어로 ip 주소 확인하여  변경 
    -----------------------------
      #ip = 'localhost' # ip 주소 
      ip = '2033.14.1.2' # ip 주소 
    -----------------------------
    (2) server side 실행  >python server_rec.py  
    (3) clinet side 실행  >python client_snd.py  
   * 웹서버  실행하여  웹페이지에서 ai모델 결과 확인 


2.웹서버 
   1) 웹서버 실행 
     fire_webserver/
     > python app.py
   2) http://127.0.0.1:5000/ 에서 확인  
   3) ai 모델 파일 위치 


3.ai 모델 detection  모듈  
#-----------------------------------------------------------------
# 1) 파일 실행 : detect.py --source  static/fire_img/fire0.jpg
# 2) 웹캠 실행 : detect.py --source 0
# 3) Webserver 호출 :   detect.detect()
#-----------------------------------------------------------------
 이미지 처리 결과 저장 위치  : fire_webserver/static/detet/


 
