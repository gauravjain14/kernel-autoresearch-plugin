"""Documentation for agents: the harness fetches public GPU documentation from an allowlist, caches it as sectioned
text, and searches it. It runs in the harness process, so an agent whose sandbox has no network can still read the
PTX ISA or the tuning guide for its GPU through `docs`.

    docs list                                   the known documents
    docs find "TERM" [--in NAME,NAME]           sections that mention TERM, best first, with a snippet each
    docs get NAME|URL [--section "TERM"]        a document's sections (and links), or one section in full

Any page on an allowed domain can be fetched by URL, which is how an agent follows a link from an index page.
Nothing is shipped with the harness: pages are downloaded on first use and cached under ~/.cache/kar/docs.
"""
import hashlib
import html.parser
import json
import os
import re
import time
import urllib.parse
import urllib.request

CACHE = os.path.expanduser(os.environ.get("KAR_DOCS_CACHE", "~/.cache/kar/docs"))
MAX_BYTES = 60_000_000
MAX_AGE_S = 30 * 24 * 3600
ALLOW = ("docs.nvidia.com", "nvidia.github.io", "developer.nvidia.com", "triton-lang.org", "pytorch.org")
UA = "kar-docs/0.3 (documentation reader for a GPU-kernel optimization harness)"

# name -> (url, what it is for). Architecture tuning guides are listed so `docs list` can point at the right one.
SOURCES = {
    "ptx": ("https://docs.nvidia.com/cuda/parallel-thread-execution/index.html",
            "PTX ISA: every instruction, memory consistency model, async copies, mbarriers, wgmma/tcgen05, clusters"),
    "cuda-guide": ("https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html",
                   "CUDA C++ programming guide: execution model, memory hierarchy, streams, graphs, cooperative groups"),
    "best-practices": ("https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html",
                       "CUDA best practices: coalescing, occupancy, shared memory, instruction throughput"),
    "hopper": ("https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html", "Hopper (sm_90) tuning guide"),
    "blackwell": ("https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html", "Blackwell (sm_100/sm_120) tuning guide"),
    "ada": ("https://docs.nvidia.com/cuda/ada-tuning-guide/index.html", "Ada (sm_89) tuning guide"),
    "ampere": ("https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html", "Ampere (sm_80/sm_86) tuning guide"),
    "inline-ptx": ("https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html", "inline PTX in CUDA C++ (asm())"),
    "driver-api": ("https://docs.nvidia.com/cuda/cuda-driver-api/index.html",
                   "CUDA driver API: contexts, modules, cuLaunchKernel, launch attributes, memory"),
    "nvrtc": ("https://docs.nvidia.com/cuda/nvrtc/index.html", "NVRTC: runtime compilation, options, lowered names"),
    "math-api": ("https://docs.nvidia.com/cuda/cuda-math-api/index.html", "CUDA math API: intrinsics, bf16/fp16/fp8"),
    "binary-utils": ("https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html",
                     "cuobjdump and nvdisasm: reading SASS, the instruction set per architecture"),
    "ncu-guide": ("https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html",
                  "Nsight Compute profiling guide: what each metric and section means, roofline, stall reasons"),
    "ncu-cli": ("https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html", "Nsight Compute command line"),
    "nsys-guide": ("https://docs.nvidia.com/nsight-systems/UserGuide/index.html", "Nsight Systems user guide"),
    "cuda-python": ("https://nvidia.github.io/cuda-python/cuda-bindings/latest/", "cuda-python (cuda.bindings) docs"),
    "triton": ("https://triton-lang.org/main/index.html", "Triton language and compiler docs"),
    "pytorch-cuda": ("https://pytorch.org/docs/stable/notes/cuda.html",
                     "PyTorch CUDA semantics: streams, the caching allocator, CUDA graphs"),
    "cutlass": ("https://docs.nvidia.com/cutlass/index.html", "CUTLASS and CuTe docs"),
}
ARCH_GUIDE = {"sm_80": "ampere", "sm_86": "ampere", "sm_87": "ampere", "sm_89": "ada", "sm_90": "hopper",
              "sm_90a": "hopper", "sm_100": "blackwell", "sm_100a": "blackwell", "sm_120": "blackwell"}


class DocsError(Exception):
    pass


def _allow():
    from kar import config
    return tuple(ALLOW) + tuple(config.load().get("docs_allow", ()))


def _allowed(url):
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return urllib.parse.urlparse(url).scheme == "https" and any(host == d or host.endswith("." + d) for d in _allow())


class _Sections(html.parser.HTMLParser):
    """HTML -> [{title, level, anchor, text}] split at headings, plus the page's links."""

    SKIP = {"script", "style", "nav", "header", "footer", "noscript", "svg"}
    HEAD = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5}

    def __init__(self, base):
        super().__init__(convert_charrefs=True)
        self.base = base
        self.sections = [{"title": "(top)", "level": 0, "anchor": "", "text": []}]
        self.links = []
        self.skip = 0
        self.head = None
        self.pre = 0
        self.last_id = ""
        self.href = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if a.get("id"):
            self.last_id = a["id"]
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.HEAD and not self.skip:
            self.head = {"title": "", "level": self.HEAD[tag], "anchor": a.get("id") or self.last_id, "text": []}
        elif tag == "pre":
            self.pre += 1
            self._emit("\n")
        elif tag in ("p", "div", "li", "tr", "br", "dt", "dd", "table"):
            self._emit("\n")
        elif tag in ("td", "th"):
            self._emit(" | ")
        elif tag == "a" and a.get("href"):
            self.href = urllib.parse.urljoin(self.base, a["href"])

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.HEAD and self.head is not None:
            self.head["title"] = re.sub(r"\s+", " ", self.head["title"]).strip().rstrip("").strip()
            self.sections.append(self.head)
            self.head = None
        elif tag == "pre":
            self.pre = max(0, self.pre - 1)
            self._emit("\n")
        elif tag == "a":
            self.href = None

    def handle_data(self, data):
        if self.skip:
            return
        if self.head is not None:
            self.head["title"] += data
            return
        if self.href and data.strip() and _allowed(self.href):
            self.links.append((data.strip()[:80], self.href.split("#")[0]))
        self._emit(data if self.pre else re.sub(r"\s+", " ", data))

    def _emit(self, s):
        self.sections[-1]["text"].append(s)

    def result(self):
        out = []
        for s in self.sections:
            text = re.sub(r"[ \t]+\n", "\n", "".join(s["text"]))
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            if s["title"] or text:
                out.append({"title": s["title"], "level": s["level"], "anchor": s["anchor"], "text": text})
        seen, links = set(), []
        for label, href in self.links:
            if href not in seen and href != self.base.split("#")[0]:
                seen.add(href)
                links.append({"label": label, "url": href})
        return out, links


def _cache_path(url):
    return os.path.join(CACHE, hashlib.sha256(url.encode()).hexdigest()[:24] + ".json")


def page(url, refresh=False):
    """{url, title, sections, links} for an allowed URL, from the cache when fresh."""
    if not _allowed(url):
        raise DocsError(f"{url}: not an allowed documentation site ({', '.join(ALLOW)}, https only)")
    p = _cache_path(url)
    if not refresh and os.path.exists(p) and time.time() - os.path.getmtime(p) < MAX_AGE_S:
        with open(p) as f:
            return json.load(f)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            final = r.geturl()
            if not _allowed(final):
                raise DocsError(f"{url} redirected to {final}, which is not an allowed site")
            raw = r.read(MAX_BYTES + 1)
            charset = r.headers.get_content_charset() or "utf-8"
    except OSError as e:
        raise DocsError(f"could not fetch {url}: {e}") from None
    if len(raw) > MAX_BYTES:
        raise DocsError(f"{url} is larger than {MAX_BYTES // 1_000_000} MB")
    sp = _Sections(final)
    sp.feed(raw.decode(charset, errors="replace"))
    sections, links = sp.result()
    title = next((s["title"] for s in sections if s["level"] == 1), sections[0]["title"] if sections else url)
    doc = {"url": final, "title": title, "sections": sections, "links": links[:400], "fetched": time.time()}
    os.makedirs(CACHE, exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f)
    os.replace(tmp, p)
    return doc


def _resolve(name_or_url):
    if name_or_url in SOURCES:
        return SOURCES[name_or_url][0]
    if name_or_url.startswith("https://"):
        return name_or_url
    raise DocsError(f"unknown document {name_or_url!r}; `docs list` shows the names, or give an https URL")


def list_text(arch=None):
    mine = ARCH_GUIDE.get(arch or "")
    lines = ["Documents (`docs get NAME`, `docs find \"TERM\" --in NAME`):", ""]
    for name, (url, about) in SOURCES.items():
        star = "  <- this GPU's architecture" if name == mine else ""
        lines.append(f"  {name:15s} {about}{star}")
    lines += ["", f"Any https page on {', '.join(ALLOW)} can also be fetched by URL (follow links from an index page).",
              "Pages are downloaded on first use and cached for 30 days."]
    return "\n".join(lines)


def _anchor_url(doc, s):
    return doc["url"].split("#")[0] + (f"#{s['anchor']}" if s.get("anchor") else "")


def get_text(name_or_url, section=None, limit=24_000):
    doc = page(_resolve(name_or_url))
    secs = doc["sections"]
    if section:
        term = section.lower()
        scored = []
        for i, s in enumerate(secs):
            t = s["title"].lower()
            score = (3 if t == term else 2 if term in t else 0) + min(2, s["text"].lower().count(term) / 20)
            if score > 0:
                scored.append((score, i))
        if not scored:
            return f"no section of {doc['title']} matches {section!r}; `docs find \"{section}\" --in {name_or_url}`"
        _, i = max(scored)
        s = secs[i]
        body = s["text"]
        # a heading's own text can be short: include its sub-sections up to the next heading at its level
        for t in secs[i + 1:]:
            if t["level"] and t["level"] <= s["level"]:
                break
            body += f"\n\n{'#' * t['level']} {t['title']}\n\n{t['text']}"
            if len(body) > limit:
                break
        more = "\n\n(cut at the size limit; ask for a sub-section by name)" if len(body) > limit else ""
        return f"{'#' * max(1, s['level'])} {s['title']}\n{_anchor_url(doc, s)}\n\n{body[:limit]}{more}"
    toc = [f"{doc['title']}\n{doc['url']}\n", f"{len(secs)} sections ({sum(len(s['text']) for s in secs) // 1000} k "
           f"characters). `docs get {name_or_url} --section \"TITLE\"` prints one.", ""]
    toc += [f"{'  ' * max(0, s['level'] - 1)}- {s['title']}" for s in secs if s["title"] and s["level"]][:600]
    if doc["links"] and len(secs) < 40:            # an index page: show where to go next
        toc += ["", "Links on this page (`docs get URL`):"]
        toc += [f"  {lk['label']}: {lk['url']}" for lk in doc["links"][:120]]
    return "\n".join(toc)


def find_text(term, names=None, top=8, arch=None):
    if not term.strip():
        raise DocsError("give a term to search for")
    if names:
        names = [n.strip() for n in names.split(",") if n.strip()]
    else:                                   # by default the core set, plus this GPU's tuning guide
        names = ["ptx", "cuda-guide", "best-practices", "ncu-guide", "driver-api"]
        if ARCH_GUIDE.get(arch or ""):
            names.insert(1, ARCH_GUIDE[arch])
    words = [w for w in re.findall(r"[\w.:%]+", term.lower()) if w]
    hits, errors = [], []
    for n in names:
        try:
            doc = page(_resolve(n))
        except DocsError as e:
            errors.append(str(e))
            continue
        for s in doc["sections"]:
            t, x = s["title"].lower(), s["text"].lower()
            if not all(w in t or w in x for w in words):
                continue
            score = sum(3 * (w in t) + min(10, x.count(w)) for w in words) + (8 if term.lower() in x else 0)
            pos = x.find(words[0])
            snip = s["text"][max(0, pos - 160): pos + 360].replace("\n", " ")
            hits.append((score, n, doc, s, snip))
    hits.sort(key=lambda h: -h[0])
    if not hits:
        return f"nothing in {', '.join(names)} mentions all of: {', '.join(words)}" + \
            ("\n" + "\n".join(errors) if errors else "")
    out = [f"{len(hits)} sections mention {term!r} (best {min(top, len(hits))} shown; `docs get NAME --section "
           f"\"TITLE\"` for the full text):", ""]
    for score, n, doc, s, snip in hits[:top]:
        out.append(f"- [{n}] {s['title']}  ({_anchor_url(doc, s)})\n    ...{snip.strip()}...")
    if errors:
        out += ["", *errors]
    return "\n".join(out)


def main(argv, arch=None):
    """CLI and lane command: docs list | find TERM [--in A,B] | get NAME|URL [--section TERM]."""
    try:
        if not argv or argv[0] in ("list", "help", "-h", "--help"):
            return list_text(arch), 0
        cmd, rest = argv[0], argv[1:]
        if cmd == "find":
            names = None
            if "--in" in rest:
                i = rest.index("--in")
                names = rest[i + 1] if i + 1 < len(rest) else None
                rest = rest[:i] + rest[i + 2:]
            return find_text(" ".join(rest), names, arch=arch), 0
        if cmd == "get":
            if not rest:
                raise DocsError("docs get NAME|URL [--section \"TERM\"]")
            sec = None
            if "--section" in rest:
                i = rest.index("--section")
                sec = " ".join(rest[i + 1:])
                rest = rest[:i]
            return get_text(rest[0], sec), 0
        raise DocsError(f"unknown docs command {cmd!r}: list, find, get")
    except DocsError as e:
        return str(e), 1
