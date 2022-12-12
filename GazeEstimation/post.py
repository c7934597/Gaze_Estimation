import cv2
import requests

def main():
    image = cv2.imread('gaze.jpg')
    is_success, image_buffer_array = cv2.imencode(".jpg", image)
    byte_image = image_buffer_array.tobytes()

    # 資料
    my_file = {"image": byte_image}
    my_data = {"meetingId": "1", "name": "2", "timestamp": "3"}

    # 將資料加入 POST 請求中
    result = requests.post('http://localhost:8000/rgb', files = my_file, data = my_data)
    print(result.status_code)
    print(result.text)


    # my_header = {'Content-Type': 'application/json', 'X-AI-Token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzZXJ2aWNlIjoidmlkZW8tYWkifQ.9usTiECVh5htqL5x3luux3DpFCKpPXkhNLu-3xBPZEQ'}
    # my_json = {
    #         "meetingId" : "abc123",
    #         "name" : "Ming",
    #         "timestamp" : "2022-03-22T08:29:04.410Z",
    #         "result" :
    #             {
    #                 "estimation" : [10.0, 20.0, 10.0, 20.0],
    #                 "confidence" : 1
    #             }
    #         }

    # # 將資料加入 POST 請求中
    # result = requests.post('https://api-dev.sap.gozeppelin.com/reports/v1/aiReport', headers = my_header, json = my_json)
    # print(result.status_code)
    # print(result.text)


if __name__ == "__main__":
    main()