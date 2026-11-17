import os
import zipfile
import gdown
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        """
        Fetch data from the URL if not already present.
        """
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = Path(self.config.local_data_file)
            os.makedirs(zip_download_dir.parent, exist_ok=True)

            if not zip_download_dir.exists() or zip_download_dir.stat().st_size == 0:
                logger.info(f"Downloading dataset from {dataset_url} into file {zip_download_dir}")
                file_id = dataset_url.split("/")[-2]
                prefix = 'https://drive.google.com/uc?/export=download&id='
                gdown.download(prefix + file_id, str(zip_download_dir), quiet=False)
                logger.info(f"Downloaded data into {zip_download_dir}")
            else:
                logger.info(f"Dataset zip file already exists of size: {get_size(zip_download_dir)}")

            return str(zip_download_dir)

        except Exception as e:
            logger.exception(f"Failed to download data: {e}")
            raise e

    def extract_zip_file(self):
        """
        Extracts the zip file into the data directory.
        """
        unzip_path = Path(self.config.unzip_dir)
        os.makedirs(unzip_path, exist_ok=True)
        logger.info(f"Extracting zip file {self.config.local_data_file} into {unzip_path}...")
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"Extraction completed successfully at {unzip_path}")
