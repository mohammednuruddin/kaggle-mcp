import pytest

from kaggle_mcp.kaggle_api import (
    find_server_url,
    resolve_accelerator,
    server_url_candidates,
    url_shape,
)
from kaggle_mcp.refs import (
    UnsupportedNotebookUrl,
    UnsupportedServerUrl,
    parse_notebook_ref,
    parse_server_url,
    redact_url,
)


@pytest.mark.parametrize(
    "link",
    [
        "owner/my-notebook",
        "https://www.kaggle.com/code/owner/my-notebook",
        "https://www.kaggle.com/code/owner/my-notebook/edit",
        "www.kaggle.com/code/owner/my-notebook/edit/run/123456",
        "https://www.kaggle.com/code/owner/my-notebook?scriptVersionId=99#cell",
        "https://kaggle.com/kernels/owner/my-notebook",
        "https://www.kaggle.com/owner/my-notebook",
    ],
)
def test_notebook_refs(link):
    ref = parse_notebook_ref(link)
    assert ref.ref == "owner/my-notebook"
    assert ref.edit_url == "https://www.kaggle.com/code/owner/my-notebook/edit"


@pytest.mark.parametrize(
    "link",
    [
        "",
        "https://www.kaggle.com/competitions/titanic",
        "https://www.kaggle.com/datasets/owner/data",
        "https://colab.research.google.com/drive/abc",
        "kaggle.com/code/owner",
    ],
)
def test_bad_notebook_refs(link):
    with pytest.raises(UnsupportedNotebookUrl):
        parse_notebook_ref(link)


def test_server_url_splits_token_and_ui_path():
    ep = parse_server_url(
        "https://host.example/k/123/proxy/lab/tree/x.ipynb?token=abc&foo=1"
    )
    assert ep.base_url == "https://host.example/k/123/proxy?foo=1"
    assert ep.token == "abc"
    assert parse_server_url(ep.url) == ep


def test_server_url_with_path_secret():
    ep = parse_server_url(
        "https://kkb.example/k/1/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9/proxy/"
    )
    assert ep.token is None
    assert ep.base_url.endswith("/proxy")
    assert "eyJhbGci" not in ep.redacted


@pytest.mark.parametrize("bad", ["", "ftp://x", "not a url", "owner/slug"])
def test_bad_server_urls(bad):
    with pytest.raises(UnsupportedServerUrl):
        parse_server_url(bad)


def test_redact_hides_query_token():
    assert "abc" not in redact_url("https://h/x?token=abc")


def test_find_server_url_prefers_jupyter_links():
    payload = {
        "response": {
            "kernelUrl": "https://www.kaggle.com/code/owner/nb",
            "session": {
                "jupyterServerUrl": "https://kkb-production.jupyter-proxy.kaggle.net/k/1/tok/proxy"
            },
        }
    }
    assert "jupyter-proxy" in find_server_url(payload)
    assert (
        find_server_url({"response": {"url": "https://www.kaggle.com/code/a/b"}})
        is None
    )


def test_accelerators():
    assert resolve_accelerator("T4") == "NvidiaTeslaT4"
    assert resolve_accelerator("tpu") == "TpuV5E8"
    assert resolve_accelerator("cpu") is None
    assert resolve_accelerator("NvidiaL4") == "NvidiaL4"


def test_candidates_from_create_session_response():
    # Shape seen from Kaggle in Oct 2026: a bare proxy host carries the token.
    payload = {
        "response": {
            "jupyterUrl": "https://kkb-production.jupyter-proxy.kaggle.net/k/123/abcdefghijklmnopqrstuvwxyz/proxy",
            "tokenizedJupyterUrl": "https://kkb-production.jupyter-proxy.kaggle.net?token=SECRET",
        }
    }
    urls = server_url_candidates(payload)
    assert urls[0].endswith("/proxy?token=SECRET")
    assert urls[1].endswith("/proxy")
    assert urls[2] == "https://kkb-production.jupyter-proxy.kaggle.net?token=SECRET"
    assert "SECRET" not in url_shape(urls[0])
    assert "abcdefghijklmnop" not in url_shape(urls[0])


def test_redact_hides_kaggle_jwe_path_secret():
    # Real Kaggle session URLs carry a dotted JWE in the path.
    secret = "eyJhbGciOiJkaXIifQ..AAAAAAAAAAAAAAAAAAAAAA.BBBBBBBBBBBBBBBBBBBBBB.CCCCCCCCCCCCCCCCCCCCCC"
    url = f"https://kkb-production.jupyter-proxy.kaggle.net/k/123456789/{secret}/proxy"
    ep = parse_server_url(url)
    assert ep.base_url == url and ep.token is None
    assert secret not in ep.redacted and "eyJ" not in ep.redacted
    assert ep.redacted.endswith("/k/123456789/***/proxy")
