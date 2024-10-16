from azure.storage.blob import BlobServiceClient
import io
import contextlib
import logging
import os

# We use this logger to disable logs from the blob storage module
logger = logging.getLogger("adls_custom_logger")
logger.disabled = True


class ADLSDataStore:
    """
    An implementation of DataStore for Azure Data Lake Storage.
    """

    def __init__(self, container, connection_string):
        """
        Create a new instance of ADLSDataStore
        :param container: The name of the container in ADLS to interact with.
        """
        self.blob_service_client = BlobServiceClient.from_connection_string(
            connection_string, logger=logger
        )
        self.container_client = self.blob_service_client.get_container_client(
            container=container
        )
        self.container = container

    def read_file(self, path):
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container, blob=path, snapshot=None
        )
        return blob_client.download_blob(encoding="UTF-8").readall()

    def read_nonutf_file(self, path: str):
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container, blob=path, snapshot=None
        )
        return blob_client.download_blob().readall()

    def write_file(self, path: str, data) -> None:
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container, blob=path, snapshot=None
        )

        if type(data) is str:
            binary_data = data.encode()
        elif type(data) is bytes:
            binary_data = data
        else:
            print('Unsupported data type. Only "bytes" or "string" accepted')
            return

        blob_client.upload_blob(binary_data, overwrite=True)

    def upload_file(self, file_path, blob_path):
        """Uploads a single file to Azure Blob Storage."""
        try:
            blob_client = self.container_client.get_blob_client(blob_path)
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            print(f"Uploaded {file_path} to {blob_path}")
        except Exception as e:
            print(f"Failed to upload {file_path}: {e}")

    def upload_directory(self, dir_path, blob_dir_path):
        """Uploads all files from a directory to Azure Blob Storage."""
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, dir_path)
                blob_file_path = os.path.join(blob_dir_path, relative_path).replace(
                    "\\", "/"
                )

                self.upload_file(local_file_path, blob_file_path)

    def file_exists(self, path: str) -> bool:
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container, blob=path, snapshot=None
        )
        return blob_client.exists()

    def file_size(self, path: str) -> float:
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container, blob=path, snapshot=None
        )
        properties = blob_client.get_blob_properties()

        # The size is in bytes, convert it to kilobytes
        size_in_bytes = properties.size
        size_in_kb = size_in_bytes / 1024.0
        return size_in_kb

    def list_files(self, path: str):
        blob_items = self.container_client.list_blobs(name_starts_with=path)
        return [item["name"] for item in blob_items]

    def walk(self, top: str):
        top = top.rstrip("/") + "/"
        blob_items = self.container_client.list_blobs(name_starts_with=top)
        blobs = [item["name"] for item in blob_items]
        for blob in blobs:
            dirpath, filename = os.path.split(blob)
            yield (dirpath, [], [filename])

    @contextlib.contextmanager
    def open(self, path: str, mode: str = "r"):
        # read or write depending on operation
        if mode == "w":
            # create file object that will be written to
            file = io.StringIO()
            yield file

            # save the data from the file object to blob storage
            data = file.getvalue()
            self.write_file(path, data)

        elif mode == "wb":
            # create file object that will be written to
            file = io.BytesIO()
            yield file

            # save the data from the file object to blob storage
            data = file.getvalue()
            self.write_file(path, data)

        elif mode == "r":
            # download the data from blob storage
            data = self.read_file(path)

            # add data to the file object so that it can be read
            file = io.StringIO(data)
            yield file

        elif mode == "rb":
            # download the data from blob storage
            data = self.read_nonutf_file(path)

            # add data to the file object so that it can be read
            file = io.BytesIO(data)
            yield file

    def is_file(self, path: str) -> bool:
        return self.file_exists(path)

    def is_dir(self, path: str) -> bool:
        blobs = self.list_files(path=path)
        for blob in blobs:
            if blob != path:
                return True
        return False

    def rmdir(self, dir: str) -> None:
        blobs = self.list_files(dir)
        self.container_client.delete_blobs(*blobs)

    def remove(self, path: str) -> None:
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container, blob=path, snapshot=None
        )
        if blob_client.exists():
            blob_client.delete_blob()
