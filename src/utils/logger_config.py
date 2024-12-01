import logging

logging.basicConfig(filename='/root/DOMR_torch/experiment/logs/training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
