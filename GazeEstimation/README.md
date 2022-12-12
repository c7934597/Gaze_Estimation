# Gaze_Estimation
Gaze_Estimation

# Environment Version
Ubuntu : 18.04
Python : 3.8
CUDA : 11.3
cuDNN : 8.2.1
Pytorch : 1.11

# Deployment
## Install Pytorch
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

## Install requirements.txt
pip3 install -r requirements.txt

## If Python3.6, Upgrade Colorama
sudo pip3 install colorama --upgrade

# Run
cd video-ai/GazeEstimation

sudo python3 api.py --device cuda

or

sudo python3 api.py --device cuda --face-detector face_alignment_sfd

# Test
## Test API
cd video-ai/GazeEstimation

sudo python3 api.py --device cuda --face-detector face_alignment_sfd --debug

## Test Single Image
cd video-ai/GazeEstimation

sudo python3 main.py --device cuda --face-detector face_alignment_sfd --image gaze.jpg --no-screen

# Return Format
## Gaze Estimation
{
  "meetingId" : "abc123",
  "name" : "Ming",
  "timestamp" : "2022-03-22T08:29:04.410Z",
  "data" : "data",
  "result" :
   {
      "estimation" : [10.0, 20.0, 10.0, 20.0],
      "confidence" : 10
   }
}

### estimation : [0, 1, 2, 3]
estimation[0] : head pitch,
estimation[1] : head yaw,
estimation[2] : gaze pitch,
estimation[3] : gaze yaw

## Gaze Estimation Time Interval
{
  "meeting_id" : "meeting_id",
  "name" : "name",
  "data" : "data",
  "start_timestamp": 1653440711000,
  "end_timestamp": 1653440711000,
  "values": [[5,6,7,8],[8,6,7,8],[9,6,7,8],[3,6,7,8],[5,6,7,8],[1,6,7,8],[4,6,7,8],[7,6,7,8],[2,6,7,8],[3,6,7,8],[5,6,7,8],[5,6,7,8],[3,6,7,8],[4,6,7,8],[5,6,7,8]]
}

# Generate Dataset

1. Raw Data是15組(頭部垂直角度、頭部水平角度、眼睛垂直角度、眼睛水平角度)，每組角度值為[-90, 90]之間的數值。

2. Raw Data會生成4張趨勢圖，分別代表15秒區間(頭部垂直角度、頭部水平角度、眼睛垂直角度、眼睛水平角度)。

3. 使用generate_dataset.ipynb檔案並依照Raw Data生成趨勢圖。

4. 依照檔案啟用自動標註功能或是只輸出趨勢圖並自行標註。

5. 標註規則:
   5.1 看鏡頭才專心的標註標準:
      每個點-30~30之間的數值為專心，因此一張趨勢圖，依照閾值12，即15秒至少有12秒都為專心，則判定該張圖為專心，否則低於12秒為專心，即判定該張圖為不專心。
   5.2 不看鏡頭，但是長時間維持單方向的標註標準:
      不管趨勢圖起始點為何，依照趨勢圖平滑程度來標註專心或不專心，標註標準以人為找尋趨勢圖的起始點為準。
