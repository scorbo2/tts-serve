"""Stub of ``indextts.utils.model_download`` for test machines without the real package.

The server imports ``snapshot_download`` at module level for its first-start
download path; a raising stand-in keeps the import surface identical without
pulling in huggingface_hub.
"""


def snapshot_download(repo_id, local_dir, revision=None, force_download=False, **kwargs):
    raise NotImplementedError(
        "indextts stub: snapshot_download() is not available in tests"
    )
