from phytoanalytic.images import is_allowed_url

HOSTS = ("firebasestorage.googleapis.com",)


def test_allows_https_firebase_storage():
    assert is_allowed_url("https://firebasestorage.googleapis.com/v0/b/app/o/leaf.jpg", HOSTS)


def test_rejects_plain_http_and_other_hosts():
    assert not is_allowed_url("http://firebasestorage.googleapis.com/x.jpg", HOSTS)
    assert not is_allowed_url("https://169.254.169.254/latest/meta-data", HOSTS)
    assert not is_allowed_url("file:///etc/passwd", HOSTS)
