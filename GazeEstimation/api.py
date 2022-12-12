import os
import cv2
import torch
import uvicorn
import logging
import pathlib
import argparse
import numpy as np

from PIL import Image
from io import BytesIO
from common import Item, DataVerify
from gaze_inference import GazeInference
from omegaconf import DictConfig, OmegaConf
from utils import (check_path_all, expanduser_all, expanduser_env_all)

from typing import Optional
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request, File, Form
from fastapi.exceptions import RequestValidationError

app = FastAPI()

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        help='Config file. When using a config file, all the other '
        'commandline arguments are ignored. '
    )
    parser.add_argument(
        '--face-detector',
        type=str,
        default='mediapipe',
        choices=[
            'face_alignment_sfd', 'mediapipe'
        ],
        help='The method used to detect faces and find face landmarks '
        '(default: \'mediapipe\')')
    parser.add_argument('--device',
                        type=str,
                        choices=['cpu', 'cuda'],
                        help='Device used for model inference.')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def load_mode_config(args: argparse.Namespace) -> DictConfig:
    package_root = pathlib.Path(__file__).parent.resolve()
    path = package_root / 'data/configs/gaze.yaml'
    config = OmegaConf.load(path)
    config.PACKAGE_ROOT = package_root.as_posix()

    if args.face_detector:
        config.face_detector.mode = args.face_detector
    if args.device:
        config.device = args.device
    if config.device == 'cuda' and not torch.cuda.is_available():
        config.device = 'cpu'
        logger.warning('CUDA is not available, using CPU instead.')

    return config


def load_env_config() -> DictConfig:
    package_root = pathlib.Path(__file__).parent.resolve()
    path = package_root / 'env.yaml'
    env_config = OmegaConf.load(path)
    return env_config


args = parse_args()
if args.debug:
    logging.getLogger('Debug Mode').setLevel(logging.DEBUG)

MASTER_DOMAIN_PREFIX = ""

try:
    logger.info('Using config start')
    if args.config:
        config = OmegaConf.load(args.config)
    else:
        config = load_mode_config(args)
    expanduser_all(config)
    OmegaConf.set_readonly(config, True)
    logger.info(OmegaConf.to_yaml(config))
    check_path_all(config)
    logger.info('Using config end')

    if "MASTER_DOMAIN_PREFIX" in os.environ:
        if os.environ["MASTER_DOMAIN_PREFIX"] == "undefined" or os.environ["MASTER_DOMAIN_PREFIX"] == "":
            logger.error("Environment Variable - MASTER_DOMAIN_PREFIX value is error")
            os._exit(0)
        else:
            MASTER_DOMAIN_PREFIX = os.environ["MASTER_DOMAIN_PREFIX"]

    if MASTER_DOMAIN_PREFIX == "":
        logger.info('Using env_config start')
        env_config = load_env_config()
        expanduser_env_all(env_config)
        OmegaConf.set_readonly(env_config, True)
        logger.info(OmegaConf.to_yaml(env_config))
        MASTER_DOMAIN_PREFIX = env_config.MASTER_DOMAIN_PREFIX
        logger.info('Using env_config end')

    logger.info('Load model start')
    gaze_inference = GazeInference(args, config)
    logger.info('Load model end')
except Exception as e:
    logger.error(e)
    os._exit(0)


@app.exception_handler(RequestValidationError)
def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"引數不對{request.method} {request.url}")
    return JSONResponse({"code": "400", "message": exc.errors()})


@app.post("/cmyk")
def main(image: bytes = File(...), meetingId: str = Form(...), name: str = Form(...), timestamp: str = Form(...), data: str = Form(...)):
    try:
        image = np.array(Image.open(BytesIO(image)).convert('RGB')) # 將圖片轉成numpy array
        image = image[:, :, ::-1] # 將圖片轉成BGR
        result = gaze_inference.run(MASTER_DOMAIN_PREFIX, image, meetingId, name, timestamp, data) # 執行API
        logger.info(result)
        return result
    except Exception as e:
        result = {"code": "400", "message": str(e)}
        logger.error(result)
        return result


@app.post("/rgb")
def main(image: bytes = File(...), meetingId: str = Form(...), name: str = Form(...), timestamp: str = Form(...), data: str = Form(...)):
    try:
        image = np.array(Image.open(BytesIO(image))) # 將圖片轉成numpy array
        image = image[:, :, ::-1] # 將圖片轉成BGR
        result = gaze_inference.run(MASTER_DOMAIN_PREFIX, image, meetingId, name, timestamp, data) # 執行API
        logger.info(result)
        return result
    except Exception as e:
        result = {"code": "400", "message": str(e)}
        logger.error(result)
        return result


@app.post("/time_interval")
def main(item: Item):
    try:
        if DataVerify.pre_check_data(item):
            result = gaze_inference.run_time_interval(MASTER_DOMAIN_PREFIX, item)
        else:
            result = "資料格式不對"
        logger.info(result)
        return result
    except Exception as e:
        result = {"code": "400", "message": str(e)}
        logger.error(result)
        return result


logging_config = {
    "version": 1, 
    "formatters": {
        "simple": {
            "format": '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO", 
            "formatter": "simple", 
            "stream": "ext://sys.stdout"
        }
    },
    "loggers": {
        "simple_example": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": "no"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"]
    }
}


if __name__ == "__main__":
    logger.info('API test start')
    gaze_inference.run(MASTER_DOMAIN_PREFIX, cv2.imread("gaze.jpg"), "test", "test", "test", "test")
    gaze_inference.run_time_interval(MASTER_DOMAIN_PREFIX, None)
    logger.info('API test end')

    logger.info('API server start')
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=args.debug, debug=args.debug, log_level="info", log_config=logging_config)
    logger.info('API server end')