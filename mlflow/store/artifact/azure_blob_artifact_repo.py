import os
import posixpath
import re
import urllib.parse

from azure.storage.blob import BlobServiceClient
from mlflow.entities import FileInfo
from mlflow.environment_variables import MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.tracking._tracking_service.utils import _get_default_host_creds

class AzureBlobArtifactRepository(ArtifactRepository):
    """
    Stores artifacts on Azure Blob Storage & Azurite.

    This repository is used with URIs of the form
    ``wasbs://<container-name>@<storage-account-name>.blob.core.windows.net/<path>``,
    following the same URI scheme as Hadoop on Azure blob storage. It requires either that:
    - Azure storage connection string is in the env var ``AZURE_STORAGE_CONNECTION_STRING``
    - Azure storage access key is in the env var ``AZURE_STORAGE_ACCESS_KEY``
    - DefaultAzureCredential is configured

    Can also be used with an Azurite local storage emulator
    following the URI scheme explained here:
    https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azurite
    ``<scheme>://<local-machine-address>:<port>/<account-name>/<container-name>``
    - A connection string is set in the env var ``AZURITE_STORAGE_CONNECTION_STRING``
    - DefaultAzureCredential is configured
    """

    def __init__(self, artifact_uri, client=None):
        super().__init__(artifact_uri)

        _DEFAULT_TIMEOUT = 600  # 10 minutes
        self.write_timeout = MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT.get() or _DEFAULT_TIMEOUT

        # Allow override for testing
        if client:
            self.client = client
            return
        
        self.is_azurite = self.is_azurite_uri(artifact_uri) or "AZURITE_STORAGE_CONNECTION_STRING" in os.environ

        self._create_client_azurite(artifact_uri) if self.is_azurite else self._create_client_blob_storage(artifact_uri)

    @staticmethod
    def is_azurite_uri(uri):
        """Determine if URI is pointed to a local Azurite emulator"""
        parsed = urllib.parse.urlparse(uri)

        # Currently the DefaultAzureCredential TokenCredential authentication flow does not support HTTP
        # If not using a connection string with SAS token or other credentialing
        # An HTTPS URI must be used with DefaultAzureCredential. Added HTTP for posterity 
        # More info: https://blog.jongallant.com/2020/02/azurite-https-defaultazurecredential/
        return True if (parsed.scheme == "http" or parsed.scheme == "https") and ":" in parsed.netloc else False    

    @staticmethod
    def parse_azurite_uri(uri):
        """Parse a localhost Azurite URI"""
        parsed = urllib.parse.urlparse(uri)
        (scheme, netloc, path, _, _, _) = parsed

        container = path.split('/')[-1]
        if path.startswith("/"):
            path = path[1:]

        return container, scheme, path, netloc

    @staticmethod
    def parse_wasbs_uri(uri):
        """Parse a wasbs:// URI, returning (container, storage_account, path, api_uri_suffix)."""
        parsed = urllib.parse.urlparse(uri)
        (scheme, netloc, path, _, _, _) = parsed

        if scheme != "wasbs":
            raise Exception(f"Not a WASBS URI: {uri}")

        pattern = r"([^@]+)@(([^.]+)\.(blob\.core\.(windows\.net|chinacloudapi\.cn|usgovcloudapi\.net)))"

        match = re.match(
            pattern, netloc
        )

        if match is None:
            raise Exception(
                "WASBS URI must be of the form "
                "<container>@<account>.blob.core.windows.net"
                " or <container>@<account>.blob.core.chinacloudapi.cn"
                " or <container>@<account>.blob.core.usgovcloudapi.net"
            )
        
        container = match.group(1)
        storage_account = match.group(2)
        api_uri_suffix = match.group(3)

        if path.startswith("/"):
            path = path[1:]

        return container, storage_account, path, api_uri_suffix

    def log_artifact(self, local_file, artifact_path=None):
        (container, _, dest_path, _) = self.parse_azurite_uri(self.artifact_uri) if self.is_azurite else self.parse_wasbs_uri(self.artifact_uri)
        container_client = self.client.get_container_client(container)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        dest_path = posixpath.join(dest_path, os.path.basename(local_file))
        with open(local_file, "rb") as file:
            container_client.upload_blob(
                dest_path, file, overwrite=True, timeout=self.write_timeout
            )

    def log_artifacts(self, local_dir, artifact_path=None):
        (container, _, dest_path, _) = self.parse_azurite_uri(self.artifact_uri) if self.is_azurite else self.parse_wasbs_uri(self.artifact_uri)
        container_client = self.client.get_container_client(container)
        if artifact_path:
            dest_path = posixpath.join(dest_path, artifact_path)
        local_dir = os.path.abspath(local_dir)
        for root, _, filenames in os.walk(local_dir):
            upload_path = dest_path
            if root != local_dir:
                rel_path = os.path.relpath(root, local_dir)
                upload_path = posixpath.join(dest_path, rel_path)
            for f in filenames:
                remote_file_path = posixpath.join(upload_path, f)
                local_file_path = os.path.join(root, f)
                with open(local_file_path, "rb") as file:
                    container_client.upload_blob(
                        remote_file_path, file, overwrite=True, timeout=self.write_timeout
                    )

    def list_artifacts(self, path=None):
        # Newer versions of `azure-storage-blob` (>= 12.4.0) provide a public
        # `azure.storage.blob.BlobPrefix` object to signify that a blob is a directory,
        # while older versions only expose this API internally as
        # `azure.storage.blob._models.BlobPrefix`
        try:
            from azure.storage.blob import BlobPrefix
        except ImportError:
            from azure.storage.blob._models import BlobPrefix

        def is_dir(result):
            return isinstance(result, BlobPrefix)
        (container, _, artifact_path, _) = self.parse_azurite_uri(self.artifact_uri) if self.is_azurite else self.parse_wasbs_uri(self.artifact_uri)
        container_client = self.client.get_container_client(container)
        dest_path = artifact_path
        if path:
            dest_path = posixpath.join(dest_path, path)
        infos = []
        prefix = dest_path if dest_path.endswith("/") else dest_path + "/"
        results = container_client.walk_blobs(name_starts_with=prefix)

        for result in results:
            if (
                dest_path == result.name
            ):  # result isn't actually a child of the path we're interested in, so skip it
                continue

            if not result.name.startswith(artifact_path):
                raise MlflowException(
                    "The name of the listed Azure blob does not begin with the specified"
                    f" artifact path. Artifact path: {artifact_path}. Blob name: {result.name}"
                )

            if is_dir(result):
                subdir = posixpath.relpath(path=result.name, start=artifact_path)
                if subdir.endswith("/"):
                    subdir = subdir[:-1]
                infos.append(FileInfo(subdir, is_dir=True, file_size=None))
            else:  # Just a plain old blob
                file_name = posixpath.relpath(path=result.name, start=artifact_path)
                infos.append(FileInfo(file_name, is_dir=False, file_size=result.size))

        # The list_artifacts API expects us to return an empty list if the
        # the path references a single file.
        rel_path = dest_path[len(artifact_path) + 1 :]
        if (len(infos) == 1) and not infos[0].is_dir and (infos[0].path == rel_path):
            return []
        return sorted(infos, key=lambda f: f.path)

    def delete_artifacts(self, artifact_path=None):
        raise MlflowException("Not implemented yet")

    def _download_file(self, file_path, local_path):
        (container, _, root_path, _) = self.parse_azurite_uri(self.artifact_uri) if self.is_azurite else self.parse_wasbs_uri(self.artifact_uri)
        container_client = self.client.get_container_client(container)
        full_path = posixpath.join(root_path, file_path)
        with open(local_path, "wb") as file:
            container_client.download_blob(full_path).readinto(file)


    def _create_client_azurite(self, artifact_uri):
        if "AZURITE_STORAGE_CONNECTION_STRING" in os.environ:
            self.client = BlobServiceClient.from_connection_string(
                conn_str=os.environ.get("AZURITE_STORAGE_CONNECTION_STRING"),
                connection_verify=_get_default_host_creds(artifact_uri).verify,
            )
        else:
            self.client = BlobServiceClient(
                account_url=artifact_uri,
                credential=self._get_default_azure_credential(),
                connection_verify=_get_default_host_creds(artifact_uri).verify,
            )  

    def _create_client_blob_storage(self, artifact_uri):
        (_, account, _, api_uri_suffix) = self.parse_wasbs_uri(artifact_uri)

        if "AZURE_STORAGE_CONNECTION_STRING" in os.environ:
            self.client = BlobServiceClient.from_connection_string(
                conn_str=os.environ.get("AZURE_STORAGE_CONNECTION_STRING"),
                connection_verify=_get_default_host_creds(artifact_uri).verify,
            )
        elif "AZURE_STORAGE_ACCESS_KEY" in os.environ:
            self.client = BlobServiceClient(
                account_url=f"https://{account}.{api_uri_suffix}",
                credential=os.environ.get("AZURE_STORAGE_ACCESS_KEY"),
                connection_verify=_get_default_host_creds(artifact_uri).verify,
            )
        else:   
            self.client = BlobServiceClient(
                account_url=f"https://{account}.{api_uri_suffix}",
                credential=self._get_default_azure_credential(),
                connection_verify=_get_default_host_creds(artifact_uri).verify,
            )  
    
    def _get_default_azure_credential(self):
        try:
            from azure.identity import DefaultAzureCredential
            return DefaultAzureCredential()
        except ImportError as exc:
            raise ImportError(
                "Using DefaultAzureCredential requires the azure-identity package. "
                "Please install it via: pip install azure-identity"
            ) from exc
