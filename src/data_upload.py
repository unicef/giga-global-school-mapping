import os
from utils import data_store

CONNECTION_STRING = os.environ.get("ADLS_CONNECTION_STRING")
CONTAINER_NAME = "giga"

giga_store = data_store.ADLSDataStore(
    container=CONTAINER_NAME, connection_string=CONNECTION_STRING
)

dir_path = "output/TJK/results"
blob_dir_path = f"project/Dell_HPC/{dir_path}"
giga_store.upload_directory(dir_path, blob_dir_path)
# giga_store.upload_file(dir_path, blob_dir_path)

# blob_dir_path = "project/Dell_HPC/output/RWA/results/ARCHIVE"
# giga_store.rmdir(blob_dir_path)
