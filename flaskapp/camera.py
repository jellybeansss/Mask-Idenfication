import cv2
import boto3
import datetime
import requests
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor=0.6

count=0

class VideoCamera(object):    
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        #count=0
        global count
        success, image = self.video.read()
        is_success, im_buf_arr = cv2.imencode(".jpg", image)
        image1 = im_buf_arr.tobytes()
        client=boto3.client('rekognition',
                        aws_access_key_id="ASIAUJQNPAQSL6XCU2GU",
                        aws_secret_access_key="pgWMJ5VTz/q9gHMdxHYCWNf51M6CchPwGYqV+Hjs",
                        aws_session_token="FwoGZXIvYXdzEJT//////////wEaDA+o7OSr7dtzFVBUkyLEAbDmx1hkCERvIncvoswOzZ3luhRMxdYPEUNivlVKc+3FbTKn2IaxUzfSV5gyQBHafMpPwCg/aOF+TeZF911Lz4jtGc6ZZVNvks66WIIJMZOUCV6P6hkzHRn63+tYFoWJZS83ErYLdrnEc1NUFcExpQ2RJZ4Cjy3zi5I+PiukWSZ/NqS8lhGr6MQEkcvXadGG4jBVRQ1bcaie8wgkLdCAlQ+4tIP1RpjIXChqEb50SIKcX4QlpAR1spRRbql0Mp/9GpSDcU8oucXy+gUyLTANvXTzXxvtuUE9mdAF3YjuUiLn8QDQw1A0wqp4VI0jEB0E89UDllJcy2sUdA==",
                        region_name='us-east-1')
        response = client.detect_custom_labels(
        ProjectVersionArn='arn:aws:rekognition:us-east-1:295307248676:project/SafetyRating1/version/SafetyRating1.2020-09-11T18.10.14/1599828015297')
        print(response['CustomLabels'])
        
        if not len(response['CustomLabels']):
            count=count+1
            date = str(datetime.datetime.now()).split(" ")[0]
            #print(date)
            url = "https://92njv4n4ci.execute-api.us-east-1.amazonaws.com/Main123/?date="+date+"&count="+str(count)
            resp = requests.get(url)
            f = open("countfile.txt", "w")
            f.write(str(count))
            f.close()
            #print(count)

        image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in face_rects:
        	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        	break
        ret, jpeg = cv2.imencode('.jpg', image)
        #cv2.putText(image, text = str(count), org=(10,40), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(1,0,0))
        cv2.imshow('image',image)
        return jpeg.tobytes()
