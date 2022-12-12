import cv2
import logging
import sqlite3
import requests
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from common import Face, Item, DataVerify
from omegaconf import DictConfig
from gaze_estimator import GazeEstimator

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)


class GazeInference:
    def __init__(self, args, config: DictConfig):
        self.args = args
        self.config = config
        self.gaze_estimator = GazeEstimator(config)

    def run(self, MASTER_DOMAIN_PREFIX, image, meetingId, name, timestamp, data) -> None:
        return self._process_image(MASTER_DOMAIN_PREFIX, image, meetingId, name, timestamp, data)

    def run_time_interval(self, MASTER_DOMAIN_PREFIX,  item: Item = None) -> None:
        return self._process_time_interval_image(MASTER_DOMAIN_PREFIX, item)

    def _process_image(self, MASTER_DOMAIN_PREFIX, image, meetingId, name, timestamp, data) -> None:
        undistorted = cv2.undistort(
            image, self.gaze_estimator.camera.camera_matrix,
            self.gaze_estimator.camera.dist_coefficients)

        _head_pitch, _head_yaw = 0, 0
        _gaze_pitch, _gaze_yaw = 0, 0
        _confidence = 0
        each_confidence = 0
        faces = self.gaze_estimator.detect_faces(undistorted)
        for face in faces:
            self.gaze_estimator.estimate_gaze(undistorted, face)
            head_pitch, head_yaw, head_roll, distance = self._head_pose(face)
            gaze_pitch, gaze_yaw = self._gaze_vector(face)
            
            if (head_pitch >= -30 and head_pitch <= 30) and (head_yaw >= -30 and head_yaw <= 30):
                if (gaze_pitch >= -30 and gaze_pitch <= 30) and (gaze_yaw >= -30 and gaze_yaw <= 30):
                    each_confidence = 1

            # 判斷最高確信度的值
            if each_confidence > _confidence:
                _confidence = each_confidence

            _head_pitch = head_pitch
            _head_yaw = head_yaw
            _gaze_pitch = gaze_pitch
            _gaze_yaw = gaze_yaw

        # 將資料寫入資料庫
        if meetingId != "test":
            if self.args.debug:
                sqlite_result = self.write_sqlite_database(meetingId, name, data, timestamp, _head_pitch, _head_yaw, _gaze_pitch, _gaze_yaw, _confidence)
                if sqlite_result == 200:
                    return {"results": {"meetingId" : meetingId, "name" : name, "data" : data, "timestamp" : timestamp,"estimation" : [_head_pitch, _head_yaw, _gaze_pitch, _gaze_yaw], "confidence" : _confidence}}
                else:
                    return {"code": sqlite_result, "message": "debug mode, write sqlite database fail", "result": {"meetingId" : meetingId, "name" : name, "data" : data, "timestamp" : timestamp,"estimation" : [_head_pitch, _head_yaw, _gaze_pitch, _gaze_yaw], "confidence" : _confidence}}
            else:
                result = self.write_database(MASTER_DOMAIN_PREFIX, meetingId, name, timestamp, data, _head_pitch, _head_yaw, _gaze_pitch, _gaze_yaw, _confidence)
                if result.status_code == 200:
                    sqlite_result = self.write_sqlite_database(meetingId, name, data, timestamp, _head_pitch, _head_yaw, _gaze_pitch, _gaze_yaw, _confidence)
                    if sqlite_result == 200:
                        return {"code": str(result.status_code), "message": "write database success", "results": {"meetingId" : meetingId, "name" : name, "timestamp" : timestamp, "data" : data,"estimation" : [_head_pitch, _head_yaw, _gaze_pitch, _gaze_yaw], "confidence" : _confidence}}
                    else:
                        return {"code": sqlite_result, "message": "write database success, but write sqlite database fail", "results": {"meetingId" : meetingId, "name" : name, "timestamp" : timestamp, "data" : data,"estimation" : [_head_pitch, _head_yaw, _gaze_pitch, _gaze_yaw], "confidence" : _confidence}}
                else:
                    return {"code": str(result.status_code), "message": "write database fail"}
    
    def _process_time_interval_image(self, MASTER_DOMAIN_PREFIX, item: Item = None):
        # 依照資料生成趨勢圖
        x = np.linspace(0, 15, 15)

        if item == None:
            y_head_pitch = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            y_head_yaw = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        else:
            y_head_pitch = [item.values[0][0], item.values[1][0], item.values[2][0], item.values[3][0], item.values[4][0], item.values[5][0], item.values[6][0], item.values[7][0], item.values[8][0], item.values[9][0], item.values[10][0], item.values[11][0], item.values[12][0], item.values[13][0], item.values[14][0]]
            y_head_yaw = [item.values[0][1], item.values[1][1], item.values[2][1], item.values[3][1], item.values[4][1], item.values[5][1], item.values[6][1], item.values[7][1], item.values[8][1], item.values[9][1], item.values[10][1], item.values[11][1], item.values[12][1], item.values[13][1], item.values[14][1]]
        plt.xlim([0,15])
        plt.ylim([-90,90])
        plt.gcf().set_size_inches(3, 3)
        plt.plot(x, y_head_pitch)
        plt.grid(True)
        plt.savefig("head_pitch.jpg", dpi=96)
        plt.close()
        image_head_pitch = cv2.imread("head_pitch.jpg")
        image_head_pitch = cv2.cvtColor(image_head_pitch, cv2.COLOR_BGR2RGB)
        percentage_head_pitch = self.gaze_estimator._run_gaze_time_interval_model(image_head_pitch)

        plt.xlim([0,15])
        plt.ylim([-90,90])
        plt.gcf().set_size_inches(3, 3)
        plt.plot(x, y_head_yaw)
        plt.grid(True)
        plt.savefig("head_yaw.jpg", dpi=96)
        plt.close()
        image_head_yaw = cv2.imread("head_yaw.jpg")
        image_head_yaw = cv2.cvtColor(image_head_yaw, cv2.COLOR_BGR2RGB)
        percentage_head_yaw = self.gaze_estimator._run_gaze_time_interval_model(image_head_yaw)

        # 返回每個預測值的百分數
        estimation = {}

        if DataVerify.check_zero_data(y_head_pitch, y_head_yaw):
            estimation["inattentive"] = "100"
            estimation["focus"] = "0"
        else:
            estimation_all = []

            estimation_head_pitch = []
            count_head_pitch = 0
            result_head_pitch = ""
            for i in percentage_head_pitch:
                if count_head_pitch == 0:
                    estimation_all.append(round(i, 2))
                    estimation_head_pitch.append(round(i, 2))
                    if round(i, 2) > 50:
                        result_head_pitch = "inattentive"
                else:
                    estimation_all.append(round(i, 2))
                    estimation_head_pitch.append(round(i, 2))
                    if round(i, 2) > 50:
                        result_head_pitch = "focus"
                count_head_pitch += 1

            estimation_head_yaw = []
            count_head_yaw = 0
            result_head_yaw = ""
            for i in percentage_head_yaw:
                if count_head_yaw == 0:
                    estimation_all.append(round(i, 2))
                    estimation_head_yaw.append(round(i, 2))
                    if round(i, 2) > 50:
                        result_head_yaw = "inattentive"
                else:
                    estimation_all.append(round(i, 2))
                    estimation_head_yaw.append(round(i, 2))
                    if round(i, 2) > 50:
                        result_head_yaw = "focus"
                count_head_yaw += 1

            if result_head_pitch != result_head_yaw:
                if result_head_pitch == "inattentive":
                    estimation["inattentive"] = str(max(estimation_head_pitch))
                    estimation["focus"] = str(round(100 - max(estimation_head_pitch), 2))
                else:
                    estimation["inattentive"] = str(max(estimation_head_yaw))
                    estimation["focus"] = str(round(100 - max(estimation_head_yaw), 2))
            else:
                if result_head_pitch == "inattentive":
                    estimation["inattentive"] = str(max(estimation_all))
                    estimation["focus"] = str(round(100 - max(estimation_all), 2))
                else:
                    estimation["focus"] = str(max(estimation_all))
                    estimation["inattentive"] = str(round(100 - max(estimation_all), 2))

        if item != None:
            if self.args.debug:
                return {"results": {"meetingId" : item.meetingId, "name" : item.name, "data" : item.data, "startTimestamp" : item.startTimestamp, "endTimestamp" : item.endTimestamp, "result" : {"focus": estimation["focus"], "inattentive": estimation["inattentive"]}}}
            else:
                result = self.write_interval_database(MASTER_DOMAIN_PREFIX, item.meetingId, item.name, item.data, item.startTimestamp, item.endTimestamp, estimation)
                if result.status_code == 200:
                    return {"code": str(result.status_code), "message": "write interval database success", "results": {"meetingId" : item.meetingId, "name" : item.name, "data" : item.data, "startTimestamp" : item.startTimestamp, "endTimestamp" : item.endTimestamp, "result" : {"focus": estimation["focus"], "inattentive": estimation["inattentive"]}}}
                else:
                    return {"code": str(result.status_code), "message": "write interval database fail"}

    def _head_pose(self, face: Face) -> None:
        euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
        pitch, yaw, roll = face.change_coordinate_system(euler_angles)
        # logger.info(f'[head] pitch: {pitch:.2f}, yaw: {yaw:.2f}, '
        #             f'roll: {roll:.2f}, distance: {face.distance:.2f}')
        return pitch, yaw, roll, face.distance

    def _gaze_vector(self, face: Face) -> None:
        pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
        # logger.info(f'[gaze] pitch: {pitch:.2f}, yaw: {yaw:.2f}')
        return pitch, yaw

    def write_sqlite_database(self, meetingId, name, data, timestamp, head_pitch, head_yaw, gaze_pitch, gaze_yaw, confidence) -> None:
        try:
            db = sqlite3.connect('sqlite.db')
            cursor = db.cursor()
            cursor.execute("""INSERT INTO gaze_estimation (meetingId, name, data, timestamp, headPitch, headYaw, gazePitch, gazeYaw, confidence) VALUES (?,?,?,?,?,?,?,?,?);""", (meetingId, name, data, timestamp, head_pitch, head_yaw, gaze_pitch, gaze_yaw, confidence))
            db.commit()
            return 200
        except Exception as e:
            logger.error(f'[write sqlite database fail] {e}')
            return str(e)
        finally:
            db.close()

    def write_database(self, MASTER_DOMAIN_PREFIX, meetingId, name, timestamp, data, head_pitch, head_yaw, gaze_pitch, gaze_yaw, confidence) -> None:
        my_header = {'Content-Type': 'application/json', 'X-AI-Token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzZXJ2aWNlIjoidmlkZW8tYWkifQ.9usTiECVh5htqL5x3luux3DpFCKpPXkhNLu-3xBPZEQ'}
        my_json = {
            "meetingId" : meetingId,
            "name" : name,
            "timestamp" : timestamp,
            "data" : data,
            "result" :
                {
                    "estimation" : [head_pitch, head_yaw, gaze_pitch, gaze_yaw],
                    "confidence" : confidence
                }
        }

        # 將資料加入 POST 請求中
        return requests.post('https://' + MASTER_DOMAIN_PREFIX + '.sap.gozeppelin.com/reports/v1/aiReport', headers = my_header, json = my_json)

    def write_interval_database(self, MASTER_DOMAIN_PREFIX, meetingId, name, data, startTimestamp, endTimestamp, estimation) -> None:
        my_header = {'Content-Type': 'application/json', 'X-AI-Token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzZXJ2aWNlIjoidmlkZW8tYWkifQ.9usTiECVh5htqL5x3luux3DpFCKpPXkhNLu-3xBPZEQ'}
        my_params = {'t': 15}
        my_json = {
            "meetingId" : meetingId,
            "name" : name,
            "data" : data,
            "startTimestamp": startTimestamp,
            "endTimestamp": endTimestamp,
            "result" : {"focus": estimation["focus"], "inattentive": estimation["inattentive"]}
        }

        # 將資料加入 POST 請求中
        return requests.post('https://' + MASTER_DOMAIN_PREFIX + '.sap.gozeppelin.com/reports/v1/aiReport', params = my_params, headers = my_header, json = my_json)