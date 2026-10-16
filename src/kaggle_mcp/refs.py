"""Turn the links people paste into Kaggle notebook refs and Jupyter endpoints.

Two kinds of link reach the agent:

* a notebook link (``kaggle.com/code/<owner>/<slug>``, often with ``/edit`` or a
  version query) or a bare ``owner/slug`` ref, which names the notebook to pull
  and save back;
* the Jupyter server URL from the notebook editor's
  "Run > Kaggle Jupyter Server" panel, which is where code actually runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

KAGGLE = "https://www.kaggle.com"
KAGGLE_HOSTS = {"kaggle.com", "www.kaggle.com"}

_SLUG = r"[A-Za-z0-9][A-Za-z0-9_.-]*"
_REF = re.compile(rf"^(?P<owner>{_SLUG})/(?P<slug>{_SLUG})$")
# /code/<owner>/<slug>, /kernels/<owner>/<slug>, or the legacy /<owner>/<slug>
_CODE_PATH = re.compile(
    rf"^/(?:(?:code|kernels)/)?(?P<owner>{_SLUG})/(?P<slug>{_SLUG})(?:/.*)?$"
)
# First path segments that are Kaggle sections, not usernames.
_RESERVED = {
    "code",
    "kernels",
    "competitions",
    "datasets",
    "models",
    "discussions",
    "docs",
    "settings",
    "learn",
    "benchmarks",
    "organizations",
    "c",
    "d",
    "m",
}
_TOKEN_KEYS = {"token", "access_token"}


class UnsupportedNotebookUrl(ValueError):
    """The link does not name a Kaggle notebook."""


class UnsupportedServerUrl(ValueError):
    """The link is not a Jupyter server URL."""


@dataclass(frozen=True)
class NotebookRef:
    owner: str
    slug: str

    @property
    def ref(self) -> str:
        return f"{self.owner}/{self.slug}"

    @property
    def url(self) -> str:
        return f"{KAGGLE}/code/{self.owner}/{self.slug}"

    @property
    def edit_url(self) -> str:
        return f"{self.url}/edit"


def parse_notebook_ref(link: str) -> NotebookRef:
    """Return the notebook a link points at, or raise UnsupportedNotebookUrl.

    Accepts ``owner/slug``, and Kaggle URLs under ``/code/``, ``/kernels/`` or
    the legacy ``/<owner>/<slug>`` form, with or without ``/edit``, version
    paths, queries and fragments.
    """
    link = link.strip()
    if not link:
        raise UnsupportedNotebookUrl("notebook_url is empty")
    match = _REF.match(link)
    if match and "." not in match["owner"]:
        return NotebookRef(match["owner"], match["slug"])

    parts = urlsplit(link if "://" in link else f"https://{link}")
    if parts.netloc.lower() not in KAGGLE_HOSTS:
        raise UnsupportedNotebookUrl(
            f"'{link}' is not a Kaggle notebook. Use a kaggle.com/code/<owner>/<slug> link or owner/slug."
        )
    match = _CODE_PATH.match(parts.path)
    if not match or match["owner"] in _RESERVED:
        raise UnsupportedNotebookUrl(
            f"'{link}' does not point at a notebook (expected kaggle.com/code/<owner>/<slug>)"
        )
    return NotebookRef(match["owner"], match["slug"])


@dataclass(frozen=True)
class ServerEndpoint:
    """A Jupyter server: the base URL plus the token, if it came as ``?token=``.

    Kaggle's proxy URLs may carry the credential in the path instead; then
    ``token`` is None and the base URL itself is the secret.
    """

    base_url: str
    token: str | None = None

    @property
    def url(self) -> str:
        """The full URL, round-trippable through parse_server_url."""
        if not self.token:
            return self.base_url
        sep = "&" if "?" in self.base_url else "?"
        return f"{self.base_url}{sep}{urlencode({'token': self.token})}"

    @property
    def redacted(self) -> str:
        return redact_url(self.url)


_UI_SUFFIX = re.compile(r"/(?:lab|tree|notebooks|api)(?:/.*)?$")


def parse_server_url(url: str) -> ServerEndpoint:
    """Split a Jupyter server URL into base URL and token.

    Drops the fragment and any UI path (``/lab``, ``/tree``, ``/api/...``) so
    REST and websocket paths can be appended. Other query parameters are kept
    on the base URL, since a proxy may need them.
    """
    url = url.strip()
    if not url:
        raise UnsupportedServerUrl("server_url is empty")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise UnsupportedServerUrl(
            "server_url must be an http(s) Jupyter server URL, e.g. the 'VS Code compatible URL' "
            "from Run > Kaggle Jupyter Server in the Kaggle notebook editor"
        )
    query = parse_qsl(parts.query, keep_blank_values=True)
    token = next((v for k, v in query if k in _TOKEN_KEYS and v), None)
    rest = [(k, v) for k, v in query if k not in _TOKEN_KEYS]
    path = _UI_SUFFIX.sub("", parts.path).rstrip("/")
    base = urlunsplit((parts.scheme, parts.netloc, path, urlencode(rest), ""))
    return ServerEndpoint(base_url=base, token=token)


_LONG_SEGMENT = re.compile(r"(?<=/)[A-Za-z0-9._~-]{24,}(?=/|$|\?)")


def redact_url(url: str | None) -> str | None:
    """Hide tokens, including long opaque path segments, for logs and status."""
    if not url:
        return url
    parts = urlsplit(url)
    query = [
        (k, "***" if k in _TOKEN_KEYS else v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
    ]
    path = _LONG_SEGMENT.sub("***", parts.path)
    return urlunsplit(
        (parts.scheme, parts.netloc, path, urlencode(query, safe="*"), "")
    )
