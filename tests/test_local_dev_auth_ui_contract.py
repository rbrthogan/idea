from pathlib import Path


HTML_FILES = [
    Path("idea/static/html/viewer.html"),
    Path("idea/static/html/rater.html"),
    Path("idea/static/html/admin.html"),
]


def test_pages_define_dev_auth_config_before_auth_logic():
    for path in HTML_FILES:
        html = path.read_text()
        config_index = html.index("window.IDEA_DEV_AUTH")
        auth_logic_index = html.index("/static/js/auth_logic.js")

        assert config_index < auth_logic_index, path


def test_auth_logic_can_activate_local_dev_auth():
    script = Path("idea/static/js/auth_logic.js").read_text()

    assert "window.IDEA_DEV_AUTH" in script
    assert "activateDevAuth" in script
    assert "getIdToken" in script
    assert "local-dev-auth" in script
