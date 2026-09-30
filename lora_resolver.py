"""
lora_resolver.py - identity, local relocation and Civitai download for the
LoRA Wrangler stack nodes.

A workflow row can carry an identity stamp in its comment
(`# sha: 0123456789ab; civitai: 123456@654321; size: 144703488`). With
that, a lora referenced by a name the host doesn't have can still be
found (same file under another name/folder) or fetched from Civitai into
`models/loras/<download folder>/` before the run would fail.

Everything here is stdlib + folder_paths. Never imports torch.

The API key and the auto-download mode (off / manifest / any) live in
civitai_key.json next to this file, never in a workflow and never in
ComfyUI's settings store, which anyone reaching the port can write. The
endpoint that changes them only accepts requests from the server machine.
Env overrides: LORA_WRANGLER_AUTO_DOWNLOAD, CIVITAI_API_KEY (also
CIVITAI_TOKEN / CIVITAI_API_TOKEN). Download folder and folder matching
are plain ComfyUI settings (LoraWrangler.DownloadFolder / MatchFolders);
downloads stay inside the loras root whatever they say.

Multi-user note: execution has no request context, so run-time downloads
use this one server-wide key and mode.
"""

import os
import re
import json
import time
import shutil
import hashlib
import logging
import threading
import urllib.error
import urllib.parse
import urllib.request

log = logging.getLogger("LoraWrangler")

_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
FETCH_CACHE_FILE = os.path.join(_NODE_DIR, "civitai_trigger_cache.json")
KEY_FILE = os.path.join(_NODE_DIR, "civitai_key.json")
USER_AGENT = "comfyui-lora-wrangler/1.0.0"
CIVITAI_API = "https://civitai.com/api/v1"
NEG_TTL = 7 * 24 * 3600          # re-ask civitai about unknown hashes weekly
FAIL_MEMO_TTL = 600              # after a network failure, back off 10 min
DEFAULT_DOWNLOAD_FOLDER = "auto_download"
ALLOWED_URL_HOSTS = ("civitai.com", "huggingface.co", "hf.co")
STAMP_SHA_LEN = 12               # matches Civitai's SHA256_12

AUTO_MODES = ("off", "manifest", "any")
LORA_TYPES = ("LORA", "LoCon", "DoRA")    # civitai model types we download
SETTING_FOLDER = "LoraWrangler.DownloadFolder"
SETTING_MATCH = "LoraWrangler.MatchFolders"


class DownloadError(Exception):
    """User-facing reason a download could not happen."""


# ---------------------------------------------------------------- settings / key

def _user_settings():
    try:
        import folder_paths
        p = os.path.join(folder_paths.get_user_directory(), "default",
                         "comfy.settings.json")
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _truthy(v):
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _read_server_settings():
    """civitai_key.json: the API key and the auto-download mode. Kept out of
    ComfyUI's settings store, which anyone reaching the port can write."""
    try:
        with open(KEY_FILE, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_server_settings(d):
    d = {k: v for k, v in d.items() if v not in (None, "")}
    if not d:
        try:
            os.remove(KEY_FILE)
        except FileNotFoundError:
            pass
        return
    tmp = KEY_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f)
    os.replace(tmp, KEY_FILE)
    try:
        os.chmod(KEY_FILE, 0o600)
    except OSError:
        pass


def _parse_mode(v):
    v = str(v).strip().lower()
    if v in ("manifest", "manifests"):
        return "manifest"
    if v in ("1", "true", "yes", "on", "any"):
        return "any"
    return "off"


def auto_download_mode():
    """(mode, source): "off" | "manifest" | "any", from the
    LORA_WRANGLER_AUTO_DOWNLOAD env var or the server-side settings file."""
    env = os.environ.get("LORA_WRANGLER_AUTO_DOWNLOAD")
    if env is not None and env.strip() != "":
        return _parse_mode(env), "env"
    return _parse_mode(_read_server_settings().get("auto_download", "off")), "file"


def set_auto_download(mode):
    if mode not in AUTO_MODES:
        return {"ok": False, "error": f"mode must be one of {', '.join(AUTO_MODES)}"}
    if auto_download_mode()[1] == "env":
        return {"ok": False, "error": "auto-download is set by "
                "LORA_WRANGLER_AUTO_DOWNLOAD on this server"}
    d = _read_server_settings()
    d["auto_download"] = mode
    _write_server_settings(d)
    log.info(f"auto-download mode: {mode}")
    return {"ok": True, "auto_download_mode": mode}


def download_folder_name():
    v = (os.environ.get("LORA_WRANGLER_DOWNLOAD_FOLDER")
         or _user_settings().get(SETTING_FOLDER)
         or DEFAULT_DOWNLOAD_FOLDER)
    v = str(v).strip().replace("\\", "/").strip("/")
    parts = [p for p in v.split("/") if p and p not in (".", "..")]
    return "/".join(parts) if parts else DEFAULT_DOWNLOAD_FOLDER


def match_folders_enabled():
    """Put a download into the row's own subfolder when that folder already
    exists locally (same organisation on both machines). Default on."""
    env = os.environ.get("LORA_WRANGLER_MATCH_FOLDERS")
    if env is not None and env.strip() != "":
        return _truthy(env)
    v = _user_settings().get(SETTING_MATCH)
    return True if v is None else bool(v)


def manifest_push_allowed():
    """POST /lora_wrangler/manifest is off unless the server's environment
    opts in - a manifest is content anyone reaching the port could plant,
    so it can't be enabled from the browser."""
    return _truthy(os.environ.get("LORA_WRANGLER_ALLOW_MANIFEST_PUSH", ""))


def get_api_key():
    """(key, source) with source in {"env", "file"}, or (None, None)."""
    for name in ("CIVITAI_API_KEY", "CIVITAI_TOKEN", "CIVITAI_API_TOKEN"):
        v = os.environ.get(name)
        if v and v.strip():
            return v.strip(), "env"
    k = str(_read_server_settings().get("key") or "").strip()
    return (k, "file") if k else (None, None)


def set_api_key(key):
    """Store (non-empty) or clear (empty/None) the key. Never logs it."""
    key = (key or "").strip()
    d = _read_server_settings()
    d["key"] = key
    _write_server_settings(d)
    if not key:
        log.info("civitai api key cleared")
        return {"ok": True, "set": False}
    log.info("civitai api key saved (...%s)", key[-4:])
    return {"ok": True, "set": True, "hint": key[-4:]}


def key_status():
    k, src = get_api_key()
    mode, mode_src = auto_download_mode()
    return {"set": bool(k), "source": src, "hint": (k[-4:] if k else None),
            "auto_download": mode != "off",
            "auto_download_mode": mode, "auto_download_source": mode_src,
            "folder": download_folder_name(),
            "match_folders": match_folders_enabled()}


# ---------------------------------------------------------------- hash cache

_cache = None
_cache_lock = threading.RLock()


def get_cache():
    """{"files": {abs_path: {mtime, size, sha256}},
        "hashes": {sha256: {words, fetched, version_id, ...}}}"""
    global _cache
    with _cache_lock:
        if _cache is None:
            try:
                with open(FETCH_CACHE_FILE, "r", encoding="utf-8") as f:
                    _cache = json.load(f)
            except Exception:
                _cache = {}
            if not isinstance(_cache, dict):
                _cache = {}
            _cache.setdefault("files", {})
            _cache.setdefault("hashes", {})
        return _cache


_last_save = 0.0
_dirty = False
SAVE_INTERVAL = 2.0     # bulk hashing must not rewrite the file per lora


def save_cache(force=False):
    """Write the cache; bursts are coalesced to one write per SAVE_INTERVAL
    (flush_cache() or force=True writes pending changes immediately)."""
    global _last_save, _dirty
    with _cache_lock:
        cache = get_cache()
        now = time.time()
        if not force and now - _last_save < SAVE_INTERVAL:
            _dirty = True
            return
        try:
            tmp = FETCH_CACHE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cache, f)
            os.replace(tmp, FETCH_CACHE_FILE)
            _last_save = now
            _dirty = False
        except Exception as e:
            log.warning(f"could not save trigger cache: {e}")


def flush_cache():
    """Write out anything a debounced save left pending."""
    if _dirty:
        save_cache(force=True)


import atexit
atexit.register(flush_cache)


def _norm_path(p):
    return os.path.normcase(os.path.normpath(p))


def sha256_of(path):
    """Full lowercase SHA256, cached by path+mtime+size."""
    st = os.stat(path)
    with _cache_lock:
        cache = get_cache()
        ent = cache["files"].get(path)
        if (ent and ent.get("mtime") == st.st_mtime
                and ent.get("size") == st.st_size and ent.get("sha256")):
            return ent["sha256"].lower()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 23), b""):
            h.update(chunk)
    sha = h.hexdigest()
    with _cache_lock:
        get_cache()["files"][path] = {"mtime": st.st_mtime,
                                      "size": st.st_size, "sha256": sha}
        save_cache()
    return sha


def cached_sha256(path):
    """The cached hash if still valid, else None. Never hashes."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    with _cache_lock:
        ent = get_cache()["files"].get(path)
    if (ent and ent.get("mtime") == st.st_mtime
            and ent.get("size") == st.st_size and ent.get("sha256")):
        return ent["sha256"].lower()
    return None


# ---------------------------------------------------------------- civitai api

_recent_failures = {}


def _failed_recently(key):
    return time.time() - _recent_failures.get(key, 0) < FAIL_MEMO_TTL


def _get_json(url, timeout=15):
    """(data, None) on 200; (None, 404) on 404; raises otherwise."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace")), None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None, 404
        raise


def split_trained_words(items):
    out = []
    for t in items or []:
        out += [x.strip() for x in str(t).split(",")]
    return [w for w in out if w][:8]


def pick_file(files, sha=None):
    """The file matching a sha (prefix ok), else primary, else first Model."""
    sha = (sha or "").lower()
    if sha:
        for f in files:
            h = ((f.get("hashes") or {}).get("SHA256") or "").lower()
            if h and h.startswith(sha):
                return f
    prim = [f for f in files if f.get("primary")]
    models = [f for f in files if f.get("type") == "Model"]
    return (prim or models or list(files) or [None])[0]


def entry_from_version(data, sha=None):
    """Compact cache entry from a model-version API response."""
    f = pick_file(data.get("files") or [], sha) or {}
    full_sha = ((f.get("hashes") or {}).get("SHA256") or sha or "").lower()
    size_kb = f.get("sizeKB")
    model = data.get("model") or {}
    return {
        "words": split_trained_words(data.get("trainedWords")),
        "fetched": time.time(),
        "version_id": data.get("id"),
        "model_id": data.get("modelId"),
        "model_name": model.get("name"),
        "version_name": data.get("name"),
        "base_model": data.get("baseModel"),
        "air": data.get("air"),
        "file_name": f.get("name"),
        "size": int(round(float(size_kb) * 1024)) if size_kb else None,
        "sha256": full_sha or None,
        "nsfw": model.get("nsfw"),
    }


def lookup_by_hash(sha, allow_network=True, need_ids=False):
    """Cache-aware model-version lookup by SHA256 (full or prefix).

    Returns the cache entry (see entry_from_version) or None. A negative
    result is an entry with "missing": True. Entries written by older
    versions lack version_id; with need_ids they are refreshed once.
    Never raises.
    """
    sha = (sha or "").lower()
    if not sha:
        return None
    now = time.time()
    with _cache_lock:
        cache = get_cache()
        ent = cache["hashes"].get(sha)
        if ent is None and len(sha) < 64:
            for k, v in cache["hashes"].items():
                if k.startswith(sha):
                    sha, ent = k, v
                    break
    if ent is not None:
        if ent.get("missing"):
            if now - ent.get("fetched", 0) < NEG_TTL:
                return ent
        elif "version_id" in ent:
            return ent
        else:
            # legacy entry: words only. Good enough unless ids are wanted.
            if not need_ids:
                if ent.get("words") or now - ent.get("fetched", 0) < NEG_TTL:
                    return ent
    if not allow_network or _failed_recently(sha):
        return ent
    try:
        data, status = _get_json(f"{CIVITAI_API}/model-versions/by-hash/{sha}")
    except Exception as e:
        _recent_failures[sha] = now
        log.warning(f"civitai lookup failed for {sha[:12]}: {e}")
        return ent
    if data is None:
        new = {"words": [], "fetched": now, "missing": True}
    else:
        new = entry_from_version(data, sha)
        if new.get("sha256") and len(sha) < 64:
            sha = new["sha256"]
    with _cache_lock:
        get_cache()["hashes"][sha] = new
        save_cache()
    return new


def fetch_version(version_id):
    """Full model-version response, or None if it is gone. Never raises."""
    key = f"v{version_id}"
    if _failed_recently(key):
        return None
    try:
        data, status = _get_json(f"{CIVITAI_API}/model-versions/{int(version_id)}")
        return data
    except Exception as e:
        _recent_failures[key] = time.time()
        log.warning(f"civitai version fetch failed for {version_id}: {e}")
        return None


# ---------------------------------------------------------------- local identity

def sidecar_identity(lora_path):
    """Identity declared by sidecars other tools write (.civitai.info from
    Civitai Helper, generic .json). {"sha256", "version_id", "model_id"},
    any of which may be absent. Never raises."""
    out = {}
    stem = os.path.splitext(lora_path)[0]
    for cand in (stem + ".civitai.info", stem + ".json", lora_path + ".json"):
        try:
            if not os.path.isfile(cand):
                continue
            with open(cand, "r", encoding="utf-8", errors="replace") as f:
                d = json.load(f)
            if not isinstance(d, dict):
                continue
            for fl in d.get("files") or []:
                h = ((fl or {}).get("hashes") or {}).get("SHA256")
                if h:
                    out.setdefault("sha256", str(h).lower())
                    break
            for k in ("sha256", "SHA256", "hash"):
                v = d.get(k)
                if isinstance(v, str) and re.fullmatch(r"[0-9a-fA-F]{64}", v):
                    out.setdefault("sha256", v.lower())
            if d.get("modelId") is not None:
                try:
                    out.setdefault("model_id", int(d["modelId"]))
                except (TypeError, ValueError):
                    pass
            vid = None
            if d.get("modelVersionId") is not None:
                vid = d["modelVersionId"]
            elif "modelId" in d and "files" in d and d.get("id") is not None:
                vid = d["id"]           # civitai.info = a model-version dump
            if vid is not None:
                try:
                    out.setdefault("version_id", int(vid))
                except (TypeError, ValueError):
                    pass
        except Exception:
            continue
    return out


def identity_for_path(path, allow_network=False):
    """{"sha", "size", "version_id", "model_id"} for a local file. Hashes
    (cached) and consults the cache / sidecars / optionally Civitai."""
    st = os.stat(path)
    sha = sha256_of(path)
    ident = {"sha": sha, "size": st.st_size,
             "version_id": None, "model_id": None}
    ent = lookup_by_hash(sha, allow_network=allow_network, need_ids=True)
    if ent and ent.get("version_id"):
        ident["version_id"] = ent.get("version_id")
        ident["model_id"] = ent.get("model_id")
    else:
        sc = sidecar_identity(path)
        if sc.get("version_id"):
            ident["version_id"] = sc["version_id"]
            ident["model_id"] = sc.get("model_id")
    return ident


def _name_paths(available):
    """{lora_name: abs_path} for everything folder_paths lists."""
    import folder_paths
    roots = folder_paths.get_folder_paths("loras")
    out = {}
    for n in available:
        for r in roots:
            p = os.path.join(r, n)
            if os.path.isfile(p):
                out[n] = p
                break
    return out


def find_local_by_stamp(stamp, available):
    """Locate a lora on this machine by identity rather than name.

    stamp: {"sha": prefix, "size": int, "version_id": int} (any subset).
    Returns (lora_name, how) with how in {"hash-cache", "sidecar", "hash",
    "version"}, or (None, None). Cheap paths first; hashing only happens
    for files whose byte size matches the stamp.
    """
    sha = (stamp.get("sha") or "").lower()
    size = stamp.get("size")
    vid = stamp.get("version_id")
    if not (sha or vid):
        return None, None
    paths = _name_paths(available)
    by_path = {_norm_path(p): n for n, p in paths.items()}
    with _cache_lock:
        cache_files = dict(get_cache()["files"])
        cache_hashes = dict(get_cache()["hashes"])

    def _valid(p, ent):
        try:
            st = os.stat(p)
        except OSError:
            return False
        return st.st_mtime == ent.get("mtime") and st.st_size == ent.get("size")

    if sha:
        for p, ent in cache_files.items():
            n = by_path.get(_norm_path(p))
            if n and (ent.get("sha256") or "").lower().startswith(sha) \
                    and _valid(p, ent):
                return n, "hash-cache"

    cands = []
    if size:
        for n, p in paths.items():
            try:
                if os.path.getsize(p) == int(size):
                    cands.append((n, p))
            except OSError:
                pass
    for n, p in cands:
        sc = sidecar_identity(p)
        if sha and (sc.get("sha256") or "").startswith(sha):
            return n, "sidecar"
        if vid and not sha and sc.get("version_id") == vid:
            return n, "sidecar"
    for n, p in cands:
        if not sha:
            break
        try:
            if sha256_of(p).startswith(sha):
                return n, "hash"
        except OSError:
            continue

    if vid:
        for p, ent in cache_files.items():
            n = by_path.get(_norm_path(p))
            if not n or not _valid(p, ent):
                continue
            h = cache_hashes.get((ent.get("sha256") or "").lower())
            if h and h.get("version_id") == vid:
                return n, "version"
        if not size:
            for n, p in paths.items():
                if sidecar_identity(p).get("version_id") == vid:
                    return n, "version"
    return None, None


# ---------------------------------------------------------------- download

_dl_locks = {}
_dl_locks_guard = threading.Lock()


def _lock_for(key):
    with _dl_locks_guard:
        lk = _dl_locks.get(key)
        if lk is None:
            lk = _dl_locks[key] = threading.Lock()
        return lk


def stamp_downloadable(stamp):
    """Does the stamp carry enough to attempt a download?"""
    if not stamp:
        return False
    return bool(stamp.get("version_id") or stamp.get("sha") or stamp.get("url"))


def download_root():
    """First writable loras root."""
    import folder_paths
    roots = folder_paths.get_folder_paths("loras")
    for r in roots:
        if os.path.isdir(r) and os.access(r, os.W_OK):
            return r
    if roots:
        os.makedirs(roots[0], exist_ok=True)
        return roots[0]
    raise DownloadError("no loras folder is configured")


def _existing_subdir(root, rel_dir):
    """Walk rel_dir under root, matching each segment case-insensitively
    (so a workflow's 'nsfw/x' finds a local 'NSFW'). Real path or None."""
    cur = root
    segs = [s for s in rel_dir.replace("\\", "/").split("/")
            if s and s not in (".", "..")]
    for seg in segs:
        try:
            names = os.listdir(cur)
        except OSError:
            return None
        # exact name first, then case-insensitive; always return the name
        # as it exists on disk so the resulting lora name matches the scan
        hit = seg if seg in names else next(
            (n for n in names if n.lower() == seg.lower()), None)
        if hit is None or not os.path.isdir(os.path.join(cur, hit)):
            return None
        cur = os.path.join(cur, hit)
    return cur


def choose_dest_dir(target_name):
    """Where a download for this row lands: (root, dest_dir, matched).

    With match-folders on and the row's own subfolder (e.g. 'NSFW/' in
    'NSFW/foo') already present under a writable loras root, that folder
    is used - the two machines evidently share the same organisation.
    Otherwise <first writable root>/<download folder>."""
    import folder_paths
    rel_dir = os.path.dirname((target_name or "").replace("\\", "/")).strip("/")
    if rel_dir and match_folders_enabled():
        for r in folder_paths.get_folder_paths("loras"):
            if not os.path.isdir(r):
                continue
            d = _existing_subdir(r, rel_dir)
            if d and os.access(d, os.W_OK):
                return r, d, True
    root = download_root()
    dest = os.path.join(root, *download_folder_name().split("/"))
    if not _inside(root, dest):
        # e.g. a folder setting of "C:/x" joins to a path on another drive
        dest = os.path.join(root, DEFAULT_DOWNLOAD_FOLDER)
    return root, dest, False


def _inside(root, path):
    """Lexical containment, so symlinked LoRA subfolders keep working."""
    root, path = os.path.abspath(root), os.path.abspath(path)
    try:
        return os.path.commonpath([root, path]) == root
    except ValueError:      # different drives
        return False


def sanitize_filename(name):
    base = os.path.basename((name or "").replace("\\", "/"))
    base = re.sub(r'[<>:"|?*\x00-\x1f]', "_", base).strip().strip(".")
    root, ext = os.path.splitext(base)
    if ext.lower() not in (".safetensors", ".sft", ".st"):
        base = (root or base) + ".safetensors"
    return base or "lora.safetensors"


def _host_allowed(url):
    try:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
    except Exception:
        return False
    return any(host == h or host.endswith("." + h) for h in ALLOWED_URL_HOSTS)


def _http_reason(e, key, data):
    code = getattr(e, "code", None)
    if code == 401:
        return ("Civitai requires an API key for this download"
                + (" (the configured key was rejected)" if key
                   else " - add one in Settings > LoRA Wrangler"))
    if code == 403:
        why = "the creator restricted downloads"
        if data and (data.get("earlyAccessEndsAt")
                     or data.get("availability") == "EarlyAccess"):
            why = "it is in early access (unlock it on Civitai first)"
        elif not key:
            why = ("it needs a logged-in account - add an API key in "
                   "Settings > LoRA Wrangler")
        return f"Civitai refused the download: {why}"
    if code == 404:
        return ("the file is no longer on Civitai (removed by the creator "
                "or by Civitai)")
    return f"HTTP {code} from the download server"


def resolve_download(stamp):
    """What a download would fetch: (version_data, file_dict, url).
    Raises DownloadError with a user-facing reason."""
    url = stamp.get("url")
    if url:
        if not _host_allowed(url):
            raise DownloadError(
                "auto-download only fetches from civitai.com and "
                "huggingface.co; download this one by hand")
        return None, None, url
    vid = stamp.get("version_id")
    sha = stamp.get("sha")
    data = fetch_version(vid) if vid else None
    if data is None and sha:
        ent = lookup_by_hash(sha, allow_network=True, need_ids=True)
        if ent and ent.get("version_id"):
            data = fetch_version(ent["version_id"])
    if data is None:
        raise DownloadError(
            "not on Civitai (removed, private, or never uploaded) - "
            "place the file in models/loras by hand")
    mtype = (data.get("model") or {}).get("type")
    if mtype and mtype not in LORA_TYPES:
        raise DownloadError(f"Civitai lists this as a {mtype}, not a LoRA")
    f = pick_file(data.get("files") or [], sha)
    if not f:
        raise DownloadError("the Civitai entry has no downloadable file")
    fmt = ((f.get("metadata") or {}).get("format") or "").lower()
    fname = (f.get("name") or "").lower()
    if (fmt and fmt != "safetensor") or not fname.endswith(".safetensors"):
        raise DownloadError(
            f"only safetensors files are auto-downloaded ({f.get('name')})")
    url = f.get("downloadUrl") or data.get("downloadUrl")
    if not url:
        raise DownloadError("Civitai returned no download url")
    return data, f, url


def _rel_name(final, root):
    return os.path.relpath(final, root)


def download_lora(stamp, target_name=None, progress=None, interrupt=None):
    """Fetch the lora a stamp identifies into <loras root>/<download folder>.

    target_name: preferred filename (the workflow's row name; only the
    basename is used). progress(done, total) and interrupt() are optional
    callbacks (interrupt raises to abort). Returns the lora name as
    folder_paths lists it (e.g. 'auto_download\\foo.safetensors').
    Raises DownloadError with a reason meant for the report.
    """
    key, _src = get_api_key()
    data, f, url = resolve_download(stamp)
    f = f or {}
    size_kb = f.get("sizeKB")
    size = (int(round(float(size_kb) * 1024)) if size_kb
            else int(stamp.get("size") or 0))
    expected_sha = ((f.get("hashes") or {}).get("SHA256") or "").lower()
    stamp_sha = (stamp.get("sha") or "").lower()
    if stamp_sha and expected_sha and not expected_sha.startswith(stamp_sha):
        raise DownloadError(
            "Civitai's file hash does not match the workflow's stamp")
    lock_key = expected_sha or stamp_sha or url
    with _lock_for(lock_key):
        root, dest_dir, matched = choose_dest_dir(target_name)
        os.makedirs(dest_dir, exist_ok=True)
        if matched:
            log.info(f"download target: the row's own folder exists here, "
                     f"using {dest_dir}")
        fname = sanitize_filename(
            target_name or f.get("name")
            or os.path.basename(urllib.parse.urlparse(url).path))
        final = os.path.join(dest_dir, fname)
        want = expected_sha or stamp_sha
        if os.path.exists(final):
            try:
                have = sha256_of(final)
                if want and have.startswith(want):
                    return _rel_name(final, root)   # a parallel run got it
            except OSError:
                pass
            stem, ext = os.path.splitext(fname)
            fname = f"{stem}_{(want or 'dl')[:STAMP_SHA_LEN]}{ext}"
            final = os.path.join(dest_dir, fname)
            if os.path.exists(final):
                try:
                    if want and sha256_of(final).startswith(want):
                        return _rel_name(final, root)
                except OSError:
                    pass
                raise DownloadError(
                    f"{fname} already exists and is a different file")
        try:
            free = shutil.disk_usage(dest_dir).free
            if size and free < size + (64 << 20):
                raise DownloadError(
                    f"not enough disk space ({free >> 20} MB free, "
                    f"{size >> 20} MB needed)")
        except OSError:
            pass

        dl_url = url
        host = urllib.parse.urlparse(url).hostname or ""
        if key and host.endswith("civitai.com"):
            dl_url += (("&" if "?" in url else "?")
                       + "token=" + urllib.parse.quote(key))
        req = urllib.request.Request(dl_url, headers={"User-Agent": USER_AGENT})
        try:
            resp = urllib.request.urlopen(req, timeout=60)
        except urllib.error.HTTPError as e:
            raise DownloadError(_http_reason(e, key, data))
        except Exception as e:
            raise DownloadError(f"network error: {e}")

        part = final + ".part"
        h = hashlib.sha256()
        done = 0
        with resp:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "text/html" in ctype or "application/json" in ctype:
                raise DownloadError(
                    "the server answered with a web page instead of the file"
                    + (" - the API key is probably missing or invalid"
                       if not key else
                       " - the key may be invalid, or the model needs "
                       "purchase / early access"))
            total = int(resp.headers.get("Content-Length") or size or 0)
            try:
                free = shutil.disk_usage(dest_dir).free
            except OSError:
                free = None
            if free is not None and total and free < total + (64 << 20):
                raise DownloadError(
                    f"not enough disk space ({free >> 20} MB free, "
                    f"{total >> 20} MB needed)")
            log.info(f"downloading {fname} ({total >> 20} MB) -> {dest_dir}")
            if progress:
                progress(0, total)
            try:
                with open(part, "wb") as out:
                    while True:
                        if interrupt:
                            interrupt()
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                        h.update(chunk)
                        done += len(chunk)
                        if progress:
                            progress(done, total)
            except BaseException:
                try:
                    os.remove(part)
                except OSError:
                    pass
                raise
        got = h.hexdigest()
        bad = ((expected_sha and got != expected_sha)
               or (stamp_sha and not got.startswith(stamp_sha))
               or (size and done != size))
        if bad:
            try:
                os.remove(part)
            except OSError:
                pass
            raise DownloadError(
                f"downloaded file failed verification "
                f"(got sha {got[:STAMP_SHA_LEN]}, {done} bytes)")
        os.replace(part, final)
        log.info(f"downloaded {fname}: sha {got[:STAMP_SHA_LEN]} ok")

        # remember everything we learned so no run ever re-asks
        try:
            st = os.stat(final)
            with _cache_lock:
                cache = get_cache()
                cache["files"][final] = {"mtime": st.st_mtime,
                                         "size": st.st_size, "sha256": got}
                if data is not None:
                    cache["hashes"][got] = entry_from_version(data, got)
                save_cache(force=True)
            if data is not None:
                info = os.path.splitext(final)[0] + ".civitai.info"
                if not os.path.exists(info):
                    with open(info, "w", encoding="utf-8") as fh:
                        json.dump(data, fh)
        except Exception as e:
            log.warning(f"post-download bookkeeping failed: {e}")
        return _rel_name(final, root)


# ---------------------------------------------------------------- manifests
#
# A manifest lists loras that a workflow may reference even though this
# server doesn't have them: the row UI shows them in the picker and a row
# named after one resolves (and downloads) through the usual ladder.
#
# Files: manifests/*.json next to this module, plus any files/dirs named
# in LORA_WRANGLER_MANIFESTS (';'-separated). Later files win on collisions.
# Accepted shapes: {"loras": [ {...}, ... ]}, [ {...}, ... ], or
# {"<name>": {...}}. Each record: name/path (relative to the loras root,
# the folder doubles as the download target when it exists locally), and
# any of sha/sha256, civitai ("modelId@versionId", a version id, or
# {"modelId", "versionId"}), size (bytes) / sizeKB, url, triggers,
# range [lo, hi], note. GET /lora_wrangler/manifest/export builds one from
# a server that has the files.

MANIFEST_DIR = os.path.join(_NODE_DIR, "manifests")
_MANIFEST_EXTS = (".safetensors", ".sft", ".st", ".pt", ".pth", ".ckpt")
_manifest_cache = {"stamp": None, "entries": {}}


def _norm_name(n):
    n = (n or "").strip().replace("\\", "/").strip("/").lower()
    root, ext = os.path.splitext(n)
    return root if ext in _MANIFEST_EXTS else n


def _civitai_str(v):
    """'modelId@versionId' / 'versionId' from the shapes people write."""
    if isinstance(v, dict):
        ver = (v.get("versionId") or v.get("version_id")
               or v.get("modelVersionId") or v.get("id"))
        mod = v.get("modelId") or v.get("model_id")
        if ver and mod:
            return f"{mod}@{ver}"
        return str(ver) if ver else ""
    return str(v or "").strip()


def _manifest_entry(name, raw):
    """Normalise one manifest record; None if it has no usable name."""
    if not isinstance(raw, dict):
        raw = {}
    name = str(name or raw.get("name") or raw.get("path") or "").strip()
    name = name.replace("\\", "/").strip("/")
    if not name:
        return None
    e = {"name": name}
    sha = str(raw.get("sha") or raw.get("sha256") or raw.get("hash") or "")
    sha = sha.strip().lower()
    if re.fullmatch(r"[0-9a-f]{10,64}", sha):
        e["sha"] = sha
    civ = _civitai_str(raw.get("civitai"))
    if not civ:
        ver = (raw.get("modelVersionId") or raw.get("version_id")
               or raw.get("versionId"))
        mod = raw.get("modelId") or raw.get("model_id")
        civ = f"{mod}@{ver}" if (ver and mod) else (str(ver) if ver else "")
    if re.match(r"^(?:urn:air:[^:]*:[^:]*:civitai:)?(?:\d+@)?\d+$", civ, re.I):
        e["civitai"] = civ
    size = raw.get("size") if raw.get("size") is not None else raw.get("bytes")
    if size is None and raw.get("sizeKB"):
        size = round(float(raw["sizeKB"]) * 1024)
    try:
        size = int(size) if size is not None else None
    except (TypeError, ValueError):
        size = None
    if size:
        e["size"] = size
    url = str(raw.get("url") or raw.get("downloadUrl") or "").strip()
    if url.startswith(("http://", "https://")):
        e["url"] = url
    tw = (raw.get("triggers") or raw.get("trainedWords")
          or raw.get("trigger_words"))
    if isinstance(tw, str):
        tw = [tw]
    if isinstance(tw, list):
        words = split_trained_words(tw)
        if words:
            e["triggers"] = words
    rng = raw.get("range")
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        try:
            lo, hi = float(rng[0]), float(rng[1])
            e["range"] = [min(lo, hi), max(lo, hi)]
        except (TypeError, ValueError):
            pass
    for k in ("note", "base_model", "model_name"):
        if raw.get(k):
            e[k] = str(raw[k])
    return e


def manifest_sources():
    """Manifest files in load order (later overrides earlier)."""
    dirs, files = [MANIFEST_DIR], []
    for extra in (os.environ.get("LORA_WRANGLER_MANIFESTS") or "").split(";"):
        extra = extra.strip()
        if not extra:
            continue
        (dirs if os.path.isdir(extra) else files).append(extra)
    out = []
    for d in dirs:
        try:
            out += sorted(os.path.join(d, f) for f in os.listdir(d)
                          if f.lower().endswith(".json"))
        except OSError:
            pass
    return out + files


def load_manifests():
    """{normalised name: entry} over every manifest file; re-read only when
    a file changes. Never raises."""
    paths = manifest_sources()
    stamp = []
    for p in paths:
        try:
            stamp.append((p, os.path.getmtime(p)))
        except OSError:
            pass
    stamp = tuple(stamp)
    if _manifest_cache["stamp"] == stamp:
        return _manifest_cache["entries"]
    entries = {}
    for p, _m in stamp:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log.warning(f"manifest {os.path.basename(p)} skipped: {e}")
            continue
        items = data.get("loras", data) if isinstance(data, dict) else data
        if isinstance(items, dict):
            pairs = list(items.items())
        elif isinstance(items, list):
            pairs = [(None, it) for it in items]
        else:
            continue
        n = 0
        for name, raw in pairs:
            e = _manifest_entry(name, raw)
            if e:
                entries[_norm_name(e["name"])] = e
                n += 1
        log.info(f"manifest {os.path.basename(p)}: {n} loras")
    _manifest_cache["stamp"] = stamp
    _manifest_cache["entries"] = entries
    return entries


def manifest_lookup(name):
    """Entry for a name (full relative path, or a basename that is unique
    in the manifest; case- and extension-insensitive), or None."""
    entries = load_manifests()
    if not entries:
        return None
    n = _norm_name(name)
    if n in entries:
        return entries[n]
    base = n.rsplit("/", 1)[-1]
    hits = [e for k, e in entries.items() if k.rsplit("/", 1)[-1] == base]
    return hits[0] if len(hits) == 1 else None


def manifest_by_sha(sha):
    sha = (sha or "").lower()
    if not sha:
        return None
    for e in load_manifests().values():
        s = e.get("sha") or ""
        if s and (s.startswith(sha) or sha.startswith(s)):
            return e
    return None


def manifest_stamp(entry):
    """The stamp dict (as the node's parse_stamp returns) for an entry."""
    out = {}
    if entry.get("sha"):
        out["sha"] = entry["sha"]
    m = re.match(r"^(?:urn:air:[^:]*:[^:]*:civitai:)?(?:(\d+)@)?(\d+)$",
                 entry.get("civitai") or "", re.I)
    if m:
        out["version_id"] = int(m.group(2))
        out["model_id"] = int(m.group(1)) if m.group(1) else None
    if entry.get("size"):
        out["size"] = int(entry["size"])
    if entry.get("url"):
        out["url"] = entry["url"]
    return out


def manifest_entry_for_stamp(stamp):
    """The manifest entry a stamp refers to (same hash, Civitai version or
    url), or None."""
    sha = (stamp.get("sha") or "").lower()
    vid = stamp.get("version_id")
    url = stamp.get("url")
    for e in load_manifests().values():
        es = e.get("sha") or ""
        if sha and es and (es.startswith(sha) or sha.startswith(es)):
            return e
        if vid and manifest_stamp(e).get("version_id") == vid:
            return e
        if url and e.get("url") == url:
            return e
    return None


def download_policy(stamp):
    """(stamp to fetch, None) or (None, reason) under the current mode. In
    manifest mode the manifest entry's own identity is what gets fetched,
    so a workflow can only pick from what the server's owner listed."""
    mode = auto_download_mode()[0]
    if mode == "off":
        return None, "auto-download is off on this server"
    if mode == "manifest":
        ent = manifest_entry_for_stamp(stamp)
        if ent is None:
            return None, ("auto-download on this server only fetches LoRAs "
                          "listed in its manifests")
        return manifest_stamp(ent), None
    return stamp, None


def manifest_fields(entry):
    """Comment fields for a row added from the manifest."""
    f = {}
    if entry.get("sha"):
        f["sha"] = entry["sha"][:STAMP_SHA_LEN]
    if entry.get("civitai"):
        f["civitai"] = entry["civitai"]
    if entry.get("size"):
        f["size"] = str(int(entry["size"]))
    if entry.get("url"):
        f["url"] = entry["url"]
    return f


def save_manifest(name, data):
    """Store a manifest pushed over the API as manifests/<slug>.json.
    Returns (path, entry count)."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "default"))
    slug = slug.strip("._") or "default"
    items = data.get("loras", data) if isinstance(data, dict) else data
    if isinstance(items, dict):
        count = sum(1 for k, v in items.items() if _manifest_entry(k, v))
    elif isinstance(items, list):
        count = sum(1 for it in items if _manifest_entry(None, it))
    else:
        raise ValueError("manifest must be a list of records, "
                         "{name: record}, or {\"loras\": [...]}")
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    path = os.path.join(MANIFEST_DIR, slug + ".json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)
    _manifest_cache["stamp"] = None
    return path, count
