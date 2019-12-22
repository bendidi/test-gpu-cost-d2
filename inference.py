
import time
import os
import torch
import cv2

from detectron2.utils.logger import setup_logger
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils import comm
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "test.jpeg")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pth")


def main():
    cfg = get_cfg()
    cfg.merge_from_file(CONFIG_PATH)
    cfg.MODEL.WEIGHTS = MODEL_PATH
    cfg.DATASETS.TEST = ("dummy")
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"

    rank = comm.get_rank()
    logger = setup_logger(distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    predictor = DefaultPredictor(cfg)
    image = cv2.imread(IMAGE_PATH)

    logger.info("Inference started in a infinite loop for profiling gpu cost, use ctrl+c to stop.")

    while 1:
        start = time.time()
        results = predictor(image)
        logger.info("Num_instances={} => Time: {} seconds.".format(len(results["instances"]), time.time() - start))


if __name__ == "__main__":
    main()