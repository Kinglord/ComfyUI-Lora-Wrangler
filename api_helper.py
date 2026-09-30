"""
Client-side helpers for driving LoRA Wrangler stack nodes over the ComfyUI API.

Not imported by ComfyUI (no comfy dependencies) — copy it into your own
codebase or import it from this folder. Works on an API-format workflow
dict, i.e. the JSON you POST to /prompt:

    {"prompt": {"12": {"class_type": "SliderLoraStack",
                       "inputs": {"model": [...], "loras": "...", ...}}}}

Stack text format (what the row UI serializes to):

    detail_slider : 0.40
    !contrast_slider : -0.25    # leading '!' = toggled off

All edits preserve line order and untouched lines. Example:

    import json, urllib.request
    from api_helper import find_stack_node, set_strength, set_many, zero_all

    wf = json.load(open("workflow_api.json"))
    nid = find_stack_node(wf)

    zero_all(wf, nid)
    set_many(wf, nid, {"detail_slider": 0.4, "contrast_slider": -0.25})
    set_enabled(wf, nid, "grain_slider", False)

    req = urllib.request.Request(
        "http://127.0.0.1:8188/prompt",
        data=json.dumps({"prompt": wf}).encode(),
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req)

Identity stamps (v1.15+): a row comment can carry
`# sha: 0123456789ab; civitai: 123456@654321; size: 144703488`, which
lets ANY server locate the same file under another name or download it
from Civitai at run time. Stamp a workflow once on the machine that has
the files (stamp_stack), then send the identical JSON to every server —
nothing per-server has to change in the request. check_stack /
download_missing are optional read-only / warm-up calls against a
server; they never modify the workflow. get_stack returns comments and
render keeps them, so read-modify-render round-trips keep stamps.
"""

import json
import os
import re

_LORA_TAG_RE = re.compile(r"^<\s*lora\s*:\s*(?P<inner>.+?)\s*>$", re.IGNORECASE)
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_LINE_RE = re.compile(
    rf"^(?P<name>.+?)\s*(?P<sep>[:=@])\s*(?P<strength>{_NUM})"
    rf"(?:\s*[:=@]\s*(?P<clip>{_NUM}))?\s*$"
)


def _norm(name):
    n = name.strip().lower().replace("\\", "/")
    root, ext = os.path.splitext(n)
    if ext in (".safetensors", ".sft", ".st", ".pt", ".pth", ".ckpt"):
        n = root
    return n


def _names_match(a, b):
    na, nb = _norm(a), _norm(b)
    if na == nb:
        return True
    # allow matching by basename so "detail_slider" hits "krea/detail_slider"
    return os.path.basename(na) == os.path.basename(nb)


def _split_line(raw):
    """Return (code, comment) where comment includes the leading '#'."""
    if "#" in raw:
        i = raw.index("#")
        return raw[:i], raw[i:]
    return raw, ""


def _parse_code(code):
    """Parse the non-comment part of a line.

    Returns (name, strength, clip_strength, sep, enabled) or None.
    A single value means clip_strength == strength.
    """
    line = code.strip()
    if not line or line.startswith("//"):
        return None
    enabled = True
    if line.startswith("!"):
        enabled = False
        line = line[1:].strip()
        if not line:
            return None
    tag = _LORA_TAG_RE.match(line)
    if tag:
        line = tag.group("inner")
        m = _LINE_RE.match(line)
        if not m:
            return None
        sm = float(m.group("strength"))
        sc = float(m.group("clip")) if m.group("clip") is not None else sm
        return m.group("name").strip(), sm, sc, ":", enabled
    m = _LINE_RE.match(line)
    if not m:
        return None
    sm = float(m.group("strength"))
    sc = float(m.group("clip")) if m.group("clip") is not None else sm
    return m.group("name").strip(), sm, sc, m.group("sep"), enabled


def _emit(name, strength, clip_strength, sep, enabled, comment=""):
    bang = "" if enabled else "!"
    if f"{clip_strength:g}" != f"{strength:g}":
        line = f"{bang}{name} {sep} {strength:g} {sep} {clip_strength:g}"
    else:
        line = f"{bang}{name} {sep} {strength:g}"
    return line + (f"  {comment}" if comment else "")


STACK_CLASSES = ("SliderLoraStack", "CallsignLoraStack")


def find_stack_node(workflow, title=None, class_types=STACK_CLASSES):
    """Return the node id of the first stack node (either variant).
    If title is given, match _meta.title instead (lets you keep several)."""
    for nid, node in workflow.items():
        if node.get("class_type") not in class_types:
            continue
        if title is None:
            return nid
        if node.get("_meta", {}).get("title") == title:
            return nid
    raise KeyError("No LoRA Wrangler stack node"
                   + (f" titled {title!r}" if title else "") + " in workflow")


def get_stack(workflow, node_id):
    """Return {name: {"strength", "clip_strength", "on", "comment", "stamp"}}
    in file order. clip_strength equals strength for single-valued lines;
    comment is the row's trailing '# ...' text (or ""); stamp is its
    identity fields ({"sha", "civitai", "size", "url"} subset, or {}).
    Feed the dict back to render() and comments — stamps included —
    survive the round-trip."""
    text = workflow[node_id]["inputs"]["loras"]
    out = {}
    for raw in text.splitlines():
        code, comment = _split_line(raw)
        parsed = _parse_code(code)
        if parsed:
            name, sm, sc, _sep, enabled = parsed
            fields, _ = parse_comment_fields(comment)
            out[name] = {"strength": sm, "clip_strength": sc, "on": enabled,
                         "comment": comment,
                         "stamp": {k: fields[k] for k in STAMP_KEYS
                                   if fields.get(k)}}
    return out


def set_strength(workflow, node_id, name, strength, clip_strength=None,
                 enable=None, append_missing=True):
    """Set one entry's strength in place.

    clip_strength: None = set both model and clip to `strength` (the
    simple/default mental model); a number = set them separately.
    enable: None = leave the on/off toggle as-is; True/False = force it.
    Returns True if an existing line was edited, False if appended
    (or missing with append_missing=False).
    """
    sc = strength if clip_strength is None else clip_strength
    inputs = workflow[node_id]["inputs"]
    lines = inputs["loras"].splitlines()
    for i, raw in enumerate(lines):
        code, comment = _split_line(raw)
        parsed = _parse_code(code)
        if parsed and _names_match(parsed[0], name):
            pname, _om, _oc, sep, enabled = parsed
            if enable is not None:
                enabled = enable
            lines[i] = _emit(pname, strength, sc, sep, enabled, comment)
            inputs["loras"] = "\n".join(lines)
            return True
    if append_missing:
        lines.append(_emit(name, strength, sc, ":",
                           True if enable is None else enable))
        inputs["loras"] = "\n".join(lines)
    return False


def set_many(workflow, node_id, updates, append_missing=True):
    """updates: {name: strength}. One pass per entry; fine at this scale."""
    for name, strength in updates.items():
        set_strength(workflow, node_id, name, strength,
                     append_missing=append_missing)


def set_enabled(workflow, node_id, name, enabled):
    """Flip the on/off toggle without touching strengths. Returns True if
    the entry existed."""
    inputs = workflow[node_id]["inputs"]
    lines = inputs["loras"].splitlines()
    for i, raw in enumerate(lines):
        code, comment = _split_line(raw)
        parsed = _parse_code(code)
        if parsed and _names_match(parsed[0], name):
            pname, sm, sc, sep, _old = parsed
            lines[i] = _emit(pname, sm, sc, sep, enabled, comment)
            inputs["loras"] = "\n".join(lines)
            return True
    return False


def zero_all(workflow, node_id):
    """Set every listed entry to 0 (keeps roster and on/off flags)."""
    inputs = workflow[node_id]["inputs"]
    lines = inputs["loras"].splitlines()
    for i, raw in enumerate(lines):
        code, comment = _split_line(raw)
        parsed = _parse_code(code)
        if parsed:
            name, _sm, _sc, sep, enabled = parsed
            lines[i] = _emit(name, 0, 0, sep, enabled, comment)
    inputs["loras"] = "\n".join(lines)


def set_solo(workflow, node_id, name, strength=None, clip_strength=None):
    """Exclusive-bank helper: turn `name` on and everything else off.
    Optionally set its strength(s) at the same time (clip follows model
    unless given). Returns True if the entry existed."""
    inputs = workflow[node_id]["inputs"]
    lines = inputs["loras"].splitlines()
    found = False
    for i, raw in enumerate(lines):
        code, comment = _split_line(raw)
        parsed = _parse_code(code)
        if not parsed:
            continue
        pname, sm, sc, sep, _en = parsed
        if _names_match(pname, name):
            found = True
            if strength is not None:
                sm = strength
                sc = strength if clip_strength is None else clip_strength
            elif clip_strength is not None:
                sc = clip_strength
            lines[i] = _emit(pname, sm, sc, sep, True, comment)
        else:
            lines[i] = _emit(pname, sm, sc, sep, False, comment)
    if found:
        inputs["loras"] = "\n".join(lines)
    return found


def list_loras(base_url="http://127.0.0.1:8188", folder=None, search=None,
               triggers=False, manifest=False, timeout=10):
    """Query the server for available loras with optional filtering.

    list_loras(folder="characters")            -> ["Characters/a.safetensors", ...]
    list_loras(search="detail slider")          -> names matching both terms
    list_loras(folder="Body", triggers=True)   -> [{"name", "triggers", "source"}, ...]
    manifest=True also lists manifest loras the server doesn't have yet
    (detailed entries carry "virtual": True).
    """
    import urllib.parse
    import urllib.request
    q = {}
    if folder:
        q["folder"] = folder
    if search:
        q["search"] = search
    if triggers:
        q["triggers"] = "1"
    if manifest:
        q["manifest"] = "1"
    url = base_url.rstrip("/") + "/lora_wrangler/loras"
    if q:
        url += "?" + urllib.parse.urlencode(q)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))["loras"]


def get_range(base_url="http://127.0.0.1:8188", name="", fetch=False,
              timeout=15):
    """A lora's recommended strength range.

    Returns the endpoint payload: {"range": [lo, hi]|None,
    "source": "user"|"civitai"|None, "candidates": [[lo, hi], ...]|None}.
    fetch=True searches the Civitai description when nothing is saved
    (hashes the file on first contact — allow a few seconds).
    """
    import urllib.parse
    import urllib.request
    q = urllib.parse.urlencode({"name": name, "fetch": "1" if fetch else "0"})
    url = base_url.rstrip("/") + "/lora_wrangler/range?" + q
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def set_range(base_url="http://127.0.0.1:8188", name="", range=None,
              timeout=10):
    """Set (e.g. [-2, 3]) or clear (None) a lora's range. Saved as a
    user range: outranks and is never overwritten by fetched ones."""
    import urllib.request
    req = urllib.request.Request(
        base_url.rstrip("/") + "/lora_wrangler/range",
        data=json.dumps({"name": name, "range": range}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# ---------------------------------------------------------------- comment fields / stamps
#
# Row comments are ';'-separated "key: value" fields plus free text (same
# grammar as the node and the row UI):
#   # triggers: a, b; sha: 0123456789ab; civitai: 123456@654321; size: 144703488
# 'triggers' is the per-workflow trigger override; sha / civitai / size /
# url are the identity stamp. Unknown keys and free text are preserved.

_FIELD_RE = re.compile(r"^\s*([A-Za-z_][\w-]*)\s*:\s*(.*?)\s*$")
_FIELD_ORDER = ("triggers", "sha", "civitai", "size", "url")
_TRIG_KEYS = ("triggers", "trigger", "trig")
STAMP_KEYS = ("sha", "civitai", "size", "url")


def parse_comment_fields(comment):
    """'# triggers: a, b; sha: x; note' ->
    ({"triggers": "a, b", "sha": "x"}, ["note"])"""
    text = (comment or "").lstrip()
    if text.startswith("#"):
        text = text[1:]
    fields, free = {}, []
    for seg in text.split(";"):
        if not seg.strip():
            continue
        m = _FIELD_RE.match(seg)
        if m and not m.group(2).startswith("//"):   # 'https://x' is prose
            k = m.group(1).lower()
            if k in _TRIG_KEYS:
                k = "triggers"
            fields[k] = m.group(2)
        else:
            free.append(seg.strip())
    return fields, free


def render_comment_fields(fields, free=()):
    """Inverse of parse_comment_fields; '' when there is nothing to say."""
    parts = []
    for k in _FIELD_ORDER:
        v = fields.get(k)
        if v is None:
            continue
        v = str(v).strip()
        if k == "triggers" or v:
            parts.append(f"{k}: {v}".rstrip())
    for k, v in fields.items():
        if k not in _FIELD_ORDER and v is not None and str(v).strip():
            parts.append(f"{k}: {str(v).strip()}")
    parts += [s for s in free if s]
    return "# " + "; ".join(parts) if parts else ""


def get_stamp(workflow, node_id, name):
    """The row's identity fields ({"sha", "civitai", "size", "url"} subset),
    or {} when the row is unstamped or absent."""
    for raw in workflow[node_id]["inputs"]["loras"].splitlines():
        code, comment = _split_line(raw)
        parsed = _parse_code(code)
        if parsed and _names_match(parsed[0], name):
            fields, _ = parse_comment_fields(comment)
            return {k: fields[k] for k in STAMP_KEYS if fields.get(k)}
    return {}


def set_stamp(workflow, node_id, name, fields, replace=False):
    """Write identity fields into the row's comment (fills gaps; replace=True
    overwrites). Other fields and free text survive. Returns True if the
    row exists."""
    inputs = workflow[node_id]["inputs"]
    lines = inputs["loras"].splitlines()
    for i, raw in enumerate(lines):
        code, comment = _split_line(raw)
        parsed = _parse_code(code)
        if not parsed or not _names_match(parsed[0], name):
            continue
        cur, free = parse_comment_fields(comment)
        for k in STAMP_KEYS:
            if fields.get(k) and (replace or not cur.get(k)):
                cur[k] = str(fields[k])
        pname, sm, sc, sep, enabled = parsed
        lines[i] = _emit(pname, sm, sc, sep, enabled,
                         render_comment_fields(cur, free))
        inputs["loras"] = "\n".join(lines)
        return True
    return False


# ---------------------------------------------------------------- server calls
#
# None of these change the workflow except stamp_stack, which only ADDS
# identity fields. Use it once, on the machine that has the files; every
# other server resolves the stamps on its own at run time.

def _post_json(base_url, path, payload, timeout):
    import urllib.request
    req = urllib.request.Request(
        base_url.rstrip("/") + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def stamp_stack(workflow, node_id, base_url="http://127.0.0.1:8188",
                only_missing=True, network=True, timeout=300):
    """Stamp rows by asking the server that HAS the files for each lora's
    hash / Civitai id / size, and write them into the row comments. Hashes
    on first contact (a second or so per file, cached afterwards);
    network=True also fetches the Civitai id, once per file. Returns
    {name: fields} for stamped rows, {name: {"error": ...}} for the rest."""
    stack = get_stack(workflow, node_id)
    names = [n for n, v in stack.items()
             if not (only_missing
                     and (v["stamp"].get("sha") or v["stamp"].get("url")))]
    if not names:
        return {}
    d = _post_json(base_url, "/lora_wrangler/identity",
                   {"names": names, "network": network}, timeout)
    out = {}
    for n in names:
        res = (d.get("results") or {}).get(n) or {"error": "no result"}
        if res.get("fields"):
            set_stamp(workflow, node_id, n, res["fields"])
            out[n] = res["fields"]
        else:
            out[n] = {"error": res.get("error", "unknown")}
    return out


def _rows_for_check(workflow, node_id, active_only):
    rows = []
    for raw in workflow[node_id]["inputs"]["loras"].splitlines():
        code, comment = _split_line(raw)
        parsed = _parse_code(code)
        if not parsed:
            continue
        name, sm, sc, _sep, enabled = parsed
        if active_only and (not enabled or (sm == 0 and sc == 0)):
            continue
        rows.append({"name": name, "comment": comment})
    return rows


def check_stack(workflow, node_id, base_url="http://127.0.0.1:8188",
                active_only=True, timeout=60):
    """Ask a server what it would do with each row, changing nothing:
    {"rows": [{"name", "status": "found"|"relocated"|"downloadable"|
    "missing", ...}], "settings": {"set": key present, "auto_download",
    "folder"}}. Local-only on the server (no hashing, no network), so it
    is cheap enough to run before every job."""
    return _post_json(base_url, "/lora_wrangler/check",
                      {"rows": _rows_for_check(workflow, node_id, active_only)},
                      timeout)


def download_missing(workflow, node_id, base_url="http://127.0.0.1:8188",
                     active_only=True, timeout=3600):
    """Warm a server up: download every row it lacks and can fetch, now,
    so the run itself never blocks on a transfer. The workflow is not
    modified. Blocks until the transfers finish. Returns
    {name: {"ok": bool, "resolved" | "error": ...}} for rows that needed
    a download; {} when the server already has everything."""
    d = check_stack(workflow, node_id, base_url, active_only, timeout=60)
    comments = {r["name"]: r["comment"]
                for r in _rows_for_check(workflow, node_id, active_only)}
    out = {}
    for r in d.get("rows") or []:
        if r.get("status") != "downloadable":
            continue
        name = r["name"]
        out[name] = _post_json(base_url, "/lora_wrangler/download",
                               {"name": name, "comment": comments.get(name, "")},
                               timeout)
    return out


def civitai_status(base_url="http://127.0.0.1:8188", timeout=10):
    """{"set": key present, "source": "env"|"file"|None, "hint": last 4
    chars, "auto_download": bool, "auto_download_mode": "off"|"manifest"|
    "any", "folder": str, ...}. Never returns the key."""
    import urllib.request
    url = base_url.rstrip("/") + "/lora_wrangler/civitai_key"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def set_civitai_key(base_url="http://127.0.0.1:8188", key="", timeout=10):
    """Store (or clear, with "") a server's Civitai API key. Only accepted
    when this script runs on the server machine itself; anywhere else the
    reply is {"ok": False, "local_only": True}. Remote servers use the
    CIVITAI_API_KEY env var."""
    return _post_json(base_url, "/lora_wrangler/civitai_key", {"key": key},
                      timeout)


def get_manifest(base_url="http://127.0.0.1:8188", timeout=30):
    """The server's merged manifest: [{"name", "sha", "civitai", "size",
    "triggers", "range", "local": bool, "resolved"}]."""
    import urllib.request
    url = base_url.rstrip("/") + "/lora_wrangler/manifest"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8")).get("entries") or []


def push_manifest(base_url="http://127.0.0.1:8188", manifest=None,
                  name="default", timeout=60):
    """Install a manifest on a server as manifests/<name>.json. manifest is
    a dict / list, or a path to a .json file. Rows named after its entries
    then resolve and download there, and the picker shows them.
    Disabled by default: the target server must run with
    LORA_WRANGLER_ALLOW_MANIFEST_PUSH=1, otherwise the reply is
    {"ok": False, "disabled": True}. Copying the file by hand is the
    zero-exposure alternative."""
    if isinstance(manifest, str) and os.path.isfile(manifest):
        with open(manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    return _post_json(base_url, "/lora_wrangler/manifest",
                      {"name": name, "manifest": manifest}, timeout)


def export_manifest(base_url="http://127.0.0.1:8188", folder=None,
                    search=None, hash=True, network=True, only_info=False,
                    only_known=False, export_all=False, timeout=3600):
    """Build a manifest from a server that HAS the files: every lora (or
    those under `folder` / matching `search`) with hash, Civitai id, size,
    effective triggers and saved range. hash=True hashes files not yet
    cached (slow the first time); network=True fetches Civitai ids once per
    file. only_info=True keeps just loras with a .civitai.info sidecar (the
    ones that have been used), only_known=True just those already hashed —
    both skip the slow part for everything else. Returns the manifest dict:
    json.dump it where you ship it, or push_manifest it to another server."""
    import urllib.parse
    import urllib.request
    q = {"hash": "1" if hash else "0", "network": "1" if network else "0",
         "info": "1" if only_info else "0", "known": "1" if only_known else "0",
         "all": "1" if export_all else "0"}
    if folder:
        q["folder"] = folder
    if search:
        q["search"] = search
    url = (base_url.rstrip("/") + "/lora_wrangler/manifest/export?"
           + urllib.parse.urlencode(q))
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def render(stack):
    """Rebuild widget text wholesale from {name: strength} or
    {name: {"strength": s, "clip_strength": c, "on": bool, "comment": str}}.
    A "comment" (as get_stack returns it) is kept, so stamps and trigger
    annotations survive a get_stack -> edit -> render round-trip.
    Standalone comment lines / section headers are not part of the dict
    and are dropped — use set_strength & co. to keep those."""
    lines = []
    for name, v in stack.items():
        if isinstance(v, dict):
            sm = v["strength"]
            sc = v.get("clip_strength", sm)
            lines.append(_emit(name, sm, sc, ":", v.get("on", True),
                               v.get("comment", "") or ""))
        else:
            lines.append(_emit(name, v, v, ":", True))
    return "\n".join(lines)
