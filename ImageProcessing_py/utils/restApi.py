import os
import requests
import cv2
from datetime import datetime

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

def sendEvent(id, objects, thumbFrame, frameList):
    url = "http://127.0.0.1:8000/createEvent"
    date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    project_path = os.getcwd()
    createFolder(project_path + "/media/" + date)
    print(object)
    cv2.imwrite("media/" + date + "/thumb.jpg", thumbFrame)
    video = cv2.VideoWriter("media/" + date + "/event.avi", cv2.VideoWriter_fourcc(*'DIVX'), 10,
                            (thumbFrame.shape[1], thumbFrame.shape[0]))
    for f in frameList:
        video.write(f)
    video.release()
    print(objects)
    # payload = {'object': object,
    #            'dateTime': date,
    #            'id': id}
    # project_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
    # files = [
    #     ('video',
    #      ('panoramaTest1.mp4', open(project_path + '/res/0511sam/4.mp4', 'rb'),
    #       'application/octet-stream')),
    #     ('thumbNail', ('20210409(600).png', open(project_path + '/res/panorama/20210409(600).png', 'rb'), 'image/png'))
    # ]
    # response = requests.request("POST", url, data=payload, files=files)
    # print(response)


if __name__ == '__main__':
    sendEvent(1, "테스트오브젝트", None, None)
