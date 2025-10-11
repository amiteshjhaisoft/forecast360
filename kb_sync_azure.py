# Author: Amitesh Jha | iSOFT

from __future__ import annotations
import os, mimetypes, pathlib, hashlib
from typing import Optional, Iterable

# Azure SDK
from azure.storage.blob import BlobServiceClient, BlobClient, ContentSettings, ContainerClient
from azure.identity import DefaultAzureCredential


def _iter_local_files(root: str | os.PathLike) -> Iterable[pathlib.Path]:
    root = pathlib.Path(root)
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def _blob_key(local_root: pathlib.Path, file_path: pathlib.Path, prefix: str = "") -> str:
    rel = file_path.relative_to(local_root).as_posix()
    key = f"{prefix.strip('/')}/{rel}" if prefix else rel
    return key


def _content_settings_for(path: pathlib.Path) -> ContentSettings:
    ctype, _ = mimetypes.guess_type(str(path))
    return ContentSettings(content_type=ctype or "application/octet-stream")


def _get_blob_service(
    account_url: Optional[str] = None,
    connection_string: Optional[str] = None,
    container_sas_url: Optional[str] = None,
) -> tuple[BlobServiceClient | None, ContainerClient | None]:
    """
    Build clients from (in priority):
    - container_sas_url (returns None for service, but direct ContainerClient)
    - connection_string
    - account_url + DefaultAzureCredential (Managed Identity, Azure CLI login, etc.)
    """
    if container_sas_url:
        cc = ContainerClient.from_container_url(container_sas_url)
        return None, cc

    if connection_string:
        svc = BlobServiceClient.from_connection_string(connection_string)
        return svc, None

    if account_url:
        cred = DefaultAzureCredential(exclude_interactive_browser_credential=True)
        svc = BlobServiceClient(account_url=account_url, credential=cred)
        return svc, None

    raise ValueError("No Azure auth settings supplied.")


def ensure_container(
    container: str,
    account_url: Optional[str] = None,
    connection_string: Optional[str] = None,
    container_sas_url: Optional[str] = None,
) -> ContainerClient:
    svc, cc = _get_blob_service(account_url, connection_string, container_sas_url)
    if cc:
        # SAS already points to a specific container
        return cc
    assert svc is not None
    cc = svc.get_container_client(container)
    try:
        cc.create_container()   # idempotent; will 409 if exists
    except Exception:
        pass
    return cc


def _should_upload(cc: ContainerClient, blob_name: string, local_path: pathlib.Path) -> bool:
    # Heuristic sync: compare sizes; upload if missing or sizes differ
    try:
        props = cc.get_blob_client(blob_name).get_blob_properties()
        return (props.size or -1) != local_path.stat().st_size
    except Exception:
        return True  # not found or error -> upload


def sync_folder_to_blob(
    local_folder: str | os.PathLike,
    container: str,
    *,
    prefix: str = "",
    account_url: Optional[str] = None,
    connection_string: Optional[str] = None,
    container_sas_url: Optional[str] = None,
    delete_extraneous: bool = False,
    verbose: bool = True,
) -> None:
    """
    Mirror a local folder to an Azure Blob container (flat namespace with optional prefix).
    - If delete_extraneous is True, blobs under prefix that are not present locally are deleted.
    """
    local_root = pathlib.Path(local_folder).resolve()
    if not local_root.exists():
        raise FileNotFoundError(f"Folder not found: {local_root}")

    cc = ensure_container(
        container=container,
        account_url=account_url,
        connection_string=connection_string,
        container_sas_url=container_sas_url,
    )

    # Optionally compute remote listing (for deletion later)
    remote_blob_names = set()
    if delete_extraneous:
        remote_blob_names = {
            b.name for b in cc.list_blobs(name_starts_with=(prefix.strip('/') + '/') if prefix else None)
        }

    # Upload/update files
    uploaded = 0
    for f in _iter_local_files(local_root):
        name = _blob_key(local_root, f, prefix=prefix)
        if _should_upload(cc, name, f):
            if verbose: print(f"↑ {name}")
            with f.open("rb") as fh:
                cc.upload_blob(
                    name=name,
                    data=fh,
                    overwrite=True,
                    content_settings=_content_settings_for(f),
                )
            uploaded += 1
        else:
            if verbose: print(f"= {name} (unchanged)")
        if delete_extraneous and name in remote_blob_names:
            remote_blob_names.remove(name)

    # Delete orphans
    deleted = 0
    if delete_extraneous:
        for name in sorted(remote_blob_names):
            if verbose: print(f"× {name}")
            try:
                cc.delete_blob(name)
                deleted += 1
            except Exception:
                pass

    if verbose:
        print(f"Sync complete: uploaded {uploaded}, deleted {deleted} (prefix='{prefix}')")
