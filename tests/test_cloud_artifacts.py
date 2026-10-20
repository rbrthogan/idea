import hashlib

import pytest
from google.api_core.exceptions import NotFound, PreconditionFailed

from idea.artifact_store import ArtifactIntegrityError, ArtifactNotFoundError, ArtifactSizeLimitError
from idea.cloud_artifacts import CloudArtifactStore


class Blob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name
        self.size = self.generation = None

    def reload(self, **kwargs):
        if self.name not in self.bucket.objects:
            raise NotFound('missing')
        contents, generation = self.bucket.objects[self.name]
        self.size, self.generation = len(contents), generation

    def upload_from_file(self, stream, **kwargs):
        assert kwargs['if_generation_match'] == 0
        if self.name in self.bucket.objects:
            raise PreconditionFailed('exists')
        self.bucket.objects[self.name] = (stream.read(), 1)

    def download_as_bytes(self, **kwargs):
        contents, generation = self.bucket.objects[self.name]
        assert kwargs['if_generation_match'] == generation
        assert kwargs['raw_download'] is True
        return contents[kwargs['start']:kwargs['end'] + 1]

    def delete(self, **kwargs):
        assert kwargs['if_generation_match'] == self.bucket.objects[self.name][1]
        del self.bucket.objects[self.name]


class Bucket:
    def __init__(self):
        self.objects = {}

    def blob(self, name):
        return Blob(self, name)

    def list_blobs(self, prefix, **kwargs):
        for name in self.objects:
            if name.startswith(prefix):
                blob = self.blob(name)
                blob.reload()
                yield blob


def store(root, bucket, owner='a', **limits):
    return CloudArtifactStore(root, 'test-artifacts', owner * 32 + '/' + 'b' * 32, bucket=bucket, **limits)


def test_worker_upload_can_be_read_by_fresh_web_instance(tmp_path):
    bucket = Bucket()
    output = tmp_path / 'output'
    output.mkdir()
    (output / 'idea.txt').write_text('A held-out result')
    writer = store(tmp_path / 'worker', bucket)
    ref = writer.import_output(output, 'idea.txt', 'text/plain')
    reader = store(tmp_path / 'web', bucket)
    with reader.open(ref['storage_key'], 'r') as stream:
        assert stream.read() == 'A held-out result'
    assert reader.resolve(ref['storage_key']).read_text() == 'A held-out result'
    assert reader.delete(ref['storage_key']) is True
    assert reader.delete(ref['storage_key']) is False


def test_cloud_ownership_prefix_and_digest_are_enforced(tmp_path):
    bucket = Bucket()
    output = tmp_path / 'output'
    output.mkdir()
    (output / 'idea.txt').write_text('original')
    writer = store(tmp_path / 'worker', bucket)
    ref = writer.import_output(output, 'idea.txt')
    with pytest.raises(ArtifactNotFoundError):
        store(tmp_path / 'other-user', bucket, owner='c').open(ref['storage_key'])
    name = writer.prefix + ref['sha256']
    bucket.objects[name] = (b'changed!', 2)
    with pytest.raises(ArtifactIntegrityError, match='digest'):
        writer.open(ref['storage_key'])


def test_existing_cloud_quota_applies_after_restart(tmp_path):
    bucket = Bucket()
    output = tmp_path / 'output'
    output.mkdir()
    (output / 'idea.txt').write_text('123456')
    first = store(tmp_path / 'first', bucket, max_file_bytes=12, max_total_bytes=12)
    first.import_output(output, 'idea.txt')
    (output / 'idea.txt').write_text('abcdefg')
    fresh = store(tmp_path / 'fresh', bucket, max_file_bytes=12, max_total_bytes=12)
    with pytest.raises(ArtifactSizeLimitError, match='quota'):
        fresh.import_output(output, 'idea.txt')


def test_cloud_download_size_is_checked_before_fetch(tmp_path):
    bucket = Bucket()
    reader = store(tmp_path / 'web', bucket, max_file_bytes=8, max_total_bytes=8)
    contents = b'too large'
    digest = hashlib.sha256(contents).hexdigest()
    bucket.objects[reader.prefix + digest] = (contents, 1)
    with pytest.raises(ArtifactSizeLimitError):
        reader.open('art-sha256-' + digest)
