"""
LoRA Wrangler — multi-LoRA stack nodes for ComfyUI.

Two nodes sharing one engine, mirroring core's LoraLoaderModelOnly vs
LoraLoader split:

  * SliderLoraStack (Slider Stack)  — MODEL only, CLIP never touched. For slider LoRAs.
  * CallsignLoraStack (LoRA Stack) — MODEL + CLIP. By default one strength drives both;
    flip `separate_clip_strength` (advanced section) and the row UI shows
    independent model and clip fields. Also has `exclusive` (advanced):
    only one entry may be active at once — the row UI radio-toggles and
    the backend enforces it, so API edits can't silently violate it.

Frontend: a Power-Lora-Loader-style row UI (toggle / name / strength /
add, filterable chooser) implemented as a DOM widget in
web/lora_wrangler.js. DOM widgets are HTML, not litegraph canvas
drawing, which is why this works under Nodes 2.0. If the UI layer ever
fails, the node falls back to the text widget and keeps working.

Source of truth is always the "loras" STRING widget, one entry per line:

    detail_slider : 0.40
    body_b : 0.8 : 0.5              # model strength : clip strength
    !contrast_slider : -0.25        # leading '!' = toggled off
    kreasliders/zoom_slider = 0     # ':' '=' '@' all work
    <lora:grain_slider:0.15>        # a1111-style tags accepted
    # comments and blank lines ignored

One value means it applies to both model and clip (on the model-only
node, clip is always untouched regardless). External code (API) edits
exactly one string: prompt[node_id]["inputs"]["loras"]. See api_helper.py.

Entries that are toggled off or with all strengths 0 are skipped
entirely: no disk read, no model patch.
"""

import os
import re
import json
import time
import random
import struct
import html as _html
import logging
import ipaddress
import urllib.error
import urllib.request

import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management

from . import lora_resolver as _lr
from .lora_resolver import (DownloadError, lookup_by_hash, identity_for_path,
                            find_local_by_stamp, stamp_downloadable,
                            download_lora, download_policy, key_status,
                            set_api_key, set_auto_download)

log = logging.getLogger("LoraWrangler")

__version__ = "1.0.0"
log.info(f"LoRA Wrangler v{__version__} loaded")

_LORA_TAG_RE = re.compile(r"^<\s*lora\s*:\s*(?P<inner>.+?)\s*>$", re.IGNORECASE)
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_LINE_RE = re.compile(
    rf"^(?P<name>.+?)\s*[:=@]\s*(?P<sm>{_NUM})"
    rf"(?:\s*[:=@]\s*(?P<sc>{_NUM}))?\s*$"
)

_EXTS = (".safetensors", ".sft", ".st", ".pt", ".pth", ".ckpt")

# Row comments are ';'-separated "key: value" fields plus free text:
#   # triggers: a, b; sha: 0123456789ab; civitai: 123456@654321; size: 144703488
# 'triggers' is the per-workflow trigger override. sha / civitai / size /
# url form the identity stamp that lets another machine locate the same
# file under a different name or fetch it from Civitai. Unknown keys and
# free text are preserved verbatim by every edit path (backend and UI).
_FIELD_RE = re.compile(r"^\s*([A-Za-z_][\w-]*)\s*:\s*(.*?)\s*$")
_FIELD_ORDER = ("triggers", "sha", "civitai", "size", "url")
_TRIG_KEYS = ("triggers", "trigger", "trig")
_CIVITAI_ID_RE = re.compile(
    r"^(?:urn:air:[^:]*:[^:]*:civitai:)?(?:(?P<model>\d+)@)?(?P<version>\d+)$",
    re.IGNORECASE)


def parse_comment_fields(comment):
    """'# triggers: a, b; sha: x; note' -> ({"triggers": "a, b", "sha": "x"},
    ["note"]). Keys are lowercased, trigger aliases collapse to 'triggers',
    segments that aren't 'key: value' stay as free text."""
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
    """Inverse of parse_comment_fields; '' when there is nothing to say.
    An empty 'triggers' value is kept (it means 'explicitly none')."""
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


def parse_stamp(fields):
    """Identity stamp from comment fields, or None when there is none:
    {"sha": hex prefix (10..64), "size": int, "version_id": int,
     "model_id": int|None, "url": str} (each optional)."""
    out = {}
    sha = (fields.get("sha") or fields.get("sha256") or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{10,64}", sha):
        out["sha"] = sha
    m = _CIVITAI_ID_RE.match((fields.get("civitai") or "").strip())
    if m:
        out["version_id"] = int(m.group("version"))
        out["model_id"] = int(m.group("model")) if m.group("model") else None
    size = (fields.get("size") or "").strip()
    if size.isdigit():
        out["size"] = int(size)
    url = (fields.get("url") or "").strip()
    if url.startswith(("http://", "https://")):
        out["url"] = url
    return out or None


def stamp_fields(ident):
    """Comment fields for an identity dict from lora_resolver."""
    f = {}
    if ident.get("sha"):
        f["sha"] = ident["sha"][:_lr.STAMP_SHA_LEN]
    if ident.get("version_id"):
        f["civitai"] = (f"{ident['model_id']}@{ident['version_id']}"
                        if ident.get("model_id") else str(ident["version_id"]))
    if ident.get("size"):
        f["size"] = str(int(ident["size"]))
    return f


def parse_stack_text(text):
    """Parse the loras widget text.

    Returns (entries, errors); entries is a list of
    (name, strength_model, strength_clip, enabled, trigger_annotation,
    stamp) in file order. A single value means clip strength = model
    strength. trigger_annotation is a list of phrases from a
    '# triggers: a, b' comment field, or None. stamp is the row's identity
    (see parse_stamp) or None.
    """
    entries = []
    errors = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        code, _, comment = raw.partition("#")
        line = code.strip()
        fields, _free = parse_comment_fields(comment)
        ann = None
        if "triggers" in fields:
            # empty value ("# triggers:") = explicitly none
            ann = [t.strip() for t in fields["triggers"].split(",")
                   if t.strip()]
        stamp = parse_stamp(fields)
        if not line or line.startswith("//"):
            continue

        enabled = True
        if line.startswith("!"):
            enabled = False
            line = line[1:].strip()
            if not line:
                continue

        tag = _LORA_TAG_RE.match(line)
        if tag:
            line = tag.group("inner")

        m = _LINE_RE.match(line)
        if not m:
            errors.append(
                f"line {lineno}: could not parse {raw.strip()!r} "
                f"(expected 'lora_name : strength' or "
                f"'lora_name : model_strength : clip_strength')"
            )
            continue

        name = m.group("name").strip()
        if not name:
            errors.append(f"line {lineno}: empty lora name")
            continue

        try:
            sm = float(m.group("sm"))
            sc = float(m.group("sc")) if m.group("sc") is not None else sm
        except ValueError:
            errors.append(f"line {lineno}: bad strength in {raw.strip()!r}")
            continue

        entries.append((name, sm, sc, enabled, ann, stamp))
    return entries, errors


def resolve_lora(name, available):
    """Resolve a user-supplied name against folder_paths' lora list.

    Returns (resolved_name, error). Exactly one of the two is None.
    Matching order: exact -> exact+ext -> basename/stem (case-insensitive)
    -> unique case-insensitive substring.
    """
    if name in available:
        return name, None

    for ext in _EXTS:
        if name + ext in available:
            return name + ext, None

    lowered = name.lower().replace("\\", "/")
    for f in available:
        fl = f.lower().replace("\\", "/")
        base = os.path.basename(fl)
        stem = os.path.splitext(base)[0]
        if lowered in (fl, base, stem, os.path.splitext(fl)[0]):
            return f, None

    matches = [f for f in available if lowered in f.lower().replace("\\", "/")]
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        return None, (
            f"'{name}' is ambiguous, matches: {', '.join(matches[:6])}"
            + (" ..." if len(matches) > 6 else "")
        )
    return None, f"'{name}' not found in loras folder"


def _fmt_strengths(sm, sc, with_clip):
    if with_clip and f"{sm:g}" != f"{sc:g}":
        return f"{sm:g}/{sc:g}"
    return f"{sm:g}"


def _read_safetensors_metadata(path):
    """Read only the JSON header of a .safetensors file (cheap, no tensors)."""
    try:
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            if n <= 0 or n > 100_000_000:
                return {}
            header = json.loads(f.read(n))
        md = header.get("__metadata__") or {}
        return md if isinstance(md, dict) else {}
    except Exception:
        return {}


def _triggers_from_tag_frequency(raw):
    """Kohya ss_tag_frequency: triggers are the tags that appear in
    (nearly) every caption. Take the top tag plus anything within 85% of
    its count, capped at 3 — exact-tie-only was too brittle for datasets
    where the trigger misses a couple of captions."""
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        counts = {}
        for tags in data.values():          # {dataset_dir: {tag: count}}
            for tag, c in tags.items():
                tag = tag.strip()
                if tag:
                    counts[tag] = counts.get(tag, 0) + int(c)
        if not counts:
            return []
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        top = ranked[0][1]
        return [t for t, c in ranked if c >= 0.85 * top][:3]
    except Exception:
        return []


def _sidecar_paths(lora_path):
    stem = os.path.splitext(lora_path)[0]
    return (stem + ".txt", stem + ".json", stem + ".civitai.info",
            lora_path + ".json")


def resolve_declared_src(lora_path):
    """DECLARED trigger words only, local, no network, no guessing.

    Priority: sidecar .txt -> sidecar .json/.civitai.info (trainedWords /
    activation text) -> modelspec.trigger_phrase. Returns (list, source)
    with source in {"sidecar-txt", "sidecar-json", "modelspec"} or
    ([], None).
    """
    txt, js, info, js2 = _sidecar_paths(lora_path)
    try:
        if os.path.isfile(txt):
            with open(txt, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read()
            parts = [t.strip() for chunk in raw.splitlines()
                     for t in chunk.split(",")]
            parts = [t for t in parts if t and not t.startswith("#")]
            if parts:
                return parts[:8], "sidecar-txt"
    except Exception:
        pass
    for cand in (js, info, js2):
        try:
            if not os.path.isfile(cand):
                continue
            with open(cand, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
            tw = data.get("trainedWords") or data.get("triggerWords")
            if isinstance(tw, list) and tw:
                out = []
                for t in tw:
                    out += [x.strip() for x in str(t).split(",")]
                out = [t for t in out if t]
                if out:
                    return out[:8], "sidecar-json"
            at = data.get("activation text")
            if isinstance(at, str) and at.strip():
                got = [t.strip() for t in at.split(",") if t.strip()][:8]
                if got:
                    return got, "sidecar-json"
        except Exception:
            continue
    if lora_path.lower().endswith((".safetensors", ".sft", ".st")):
        md = _read_safetensors_metadata(lora_path)
        tp = md.get("modelspec.trigger_phrase")
        if isinstance(tp, str) and tp.strip():
            got = [t.strip() for t in tp.split(",") if t.strip()][:8]
            if got:
                return got, "modelspec"
    return [], None


def infer_dataset_tags(lora_path):
    """INFERRED trigger candidates from kohya ss_tag_frequency — a guess,
    used only when nothing anywhere declares triggers."""
    if lora_path.lower().endswith((".safetensors", ".sft", ".st")):
        md = _read_safetensors_metadata(lora_path)
        tf = md.get("ss_tag_frequency")
        if tf:
            return _triggers_from_tag_frequency(tf)
    return []


def resolve_triggers_src(lora_path):
    """Local-only resolution (declared first, inference last).
    Network-aware ordering lives in the callers, which slot civitai
    between these two."""
    trigs, source = resolve_declared_src(lora_path)
    if trigs:
        return trigs, source
    trigs = infer_dataset_tags(lora_path)
    return trigs, ("dataset-tags" if trigs else None)


def resolve_triggers(lora_path):
    """List-only wrapper around resolve_triggers_src."""
    return resolve_triggers_src(lora_path)[0]


# Hashing, the persistent civitai cache and the by-hash lookup live in
# lora_resolver (shared with relocation and download).
_get_fetch_cache = _lr.get_cache
_sha256_of = _lr.sha256_of
_split_trained_words = _lr.split_trained_words


def _write_info_sidecar(path, ent):
    """Minimal .civitai.info next to the file so future runs never touch the
    network and other machines can identify it. Skipped when any sidecar
    exists already or the share is read-only."""
    info = os.path.splitext(path)[0] + ".civitai.info"
    if os.path.exists(info):
        return
    try:
        with open(info, "w", encoding="utf-8") as f:
            json.dump({"trainedWords": ent.get("words") or [],
                       "modelId": ent.get("model_id"),
                       "modelVersionId": ent.get("version_id"),
                       "name": ent.get("version_name"),
                       "baseModel": ent.get("base_model"),
                       "sha256": ent.get("sha256"),
                       "source": "comfyui-lora-wrangler"}, f)
    except OSError:
        pass  # read-only share etc.; json cache still covers us


def fetch_civitai_triggers(path):
    """Look up trainedWords on Civitai by file SHA256.

    Persistent caching (hash results and 404s), sidecar write-back on
    success so future runs never touch the network, short back-off after
    network failures. Never raises.
    """
    try:
        sha = _sha256_of(path)
        ent = lookup_by_hash(sha, allow_network=True, need_ids=True)
        if not ent:
            return []
        if ent.get("missing"):
            log.info(f"civitai: no entry for {os.path.basename(path)}")
            return []
        words = list(ent.get("words") or [])
        if words or ent.get("version_id"):
            _write_info_sidecar(path, ent)
        if words:
            log.info(f"civitai triggers for {os.path.basename(path)}: "
                     f"{', '.join(words)}")
        return words
    except Exception as e:
        log.warning(f"civitai trigger fetch failed for "
                    f"{os.path.basename(path)}: {e}")
        return []


def cached_civitai_triggers(path):
    """Previously-fetched civitai triggers from the persistent JSON cache,
    without touching the network. Covers files whose sidecar couldn't be
    written (read-only share, foreign .civitai.info already present)."""
    try:
        sha = _lr.cached_sha256(path)
        if not sha:
            return []
        ent = lookup_by_hash(sha, allow_network=False)
        return list((ent or {}).get("words") or [])
    except Exception:
        return []


# civitai.red is the same site with NSFW models visible; civitai.com
# hides many of them.
CIVITAI_SITE = "https://civitai.red"


def civitai_page_url(model_id, version_id):
    if model_id and version_id:
        return f"{CIVITAI_SITE}/models/{model_id}?modelVersionId={version_id}"
    if model_id:
        return f"{CIVITAI_SITE}/models/{model_id}"
    if version_id:
        return f"{CIVITAI_SITE}/model-versions/{version_id}"
    return None


def cached_civitai_page(path):
    """Civitai page URL for a local file from the hash cache or a sidecar,
    or None. Never hashes or touches the network."""
    sha = _lr.cached_sha256(path)
    ent = lookup_by_hash(sha, allow_network=False) if sha else None
    if not (ent and (ent.get("model_id") or ent.get("version_id"))):
        ent = _lr.sidecar_identity(path)
    return civitai_page_url(ent.get("model_id"), ent.get("version_id"))


_RANGE_KEYWORDS = ("strength", "weight", "range", "recommend", "slider",
                   "use", "works", "value", "scale", "between")
_RANGE_SAFE_WORDS = ("safe", "recommend", "sweet", "normal", "default", "best")
_RANGE_MAX_WORDS = ("max", "extreme", "limit", "beyond", "absolute", "hard")
_RANGE_NUM = r"[-+]?\d+(?:\.\d+)?"
_RANGE_PAIR_RE = re.compile(
    rf"({_RANGE_NUM})\s*(?:to|and|~|\u2013|\u2014|\.\.|/|-)\s*({_RANGE_NUM})",
    re.IGNORECASE)
_RANGE_PM_RE = re.compile(rf"[\u00b1]\s*({_RANGE_NUM})")


def _strip_html(text):
    text = re.sub(r"<[^>]+>", " ", text or "")
    return _html.unescape(text)


def parse_range_candidates(text):
    """All plausible strength ranges in free text (civitai descriptions),
    ordered best-first: the safe pick leads, wider/max ranges follow as
    hints. Keyword proximity, safe-vs-max wording, zero-straddling shape,
    and nesting decide the order; magnitude bounds filter out years,
    resolutions, and step counts. Returns a list of (lo, hi), capped 4."""
    text = _strip_html(text)
    candidates = []

    def score_ctx(start, base):
        ctx = text[max(0, start - 60):start].lower()
        sc = base
        if any(k in ctx for k in _RANGE_KEYWORDS):
            sc += 2
        # authors often list a SAFE and a MAX range; prefer safe
        if any(k in ctx for k in _RANGE_SAFE_WORDS):
            sc += 2
        if any(k in ctx for k in _RANGE_MAX_WORDS):
            sc -= 1
        return sc

    for m in _RANGE_PAIR_RE.finditer(text):
        try:
            a, b = float(m.group(1)), float(m.group(2))
        except ValueError:
            continue
        lo, hi = min(a, b), max(a, b)
        if lo == hi or abs(lo) > 30 or abs(hi) > 30 or hi - lo > 60:
            continue
        sc = score_ctx(m.start(), 0)
        if lo < 0 <= hi:
            sc += 1
        candidates.append((sc, m.start(), (lo, hi)))

    for m in _RANGE_PM_RE.finditer(text):
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        if 0 < v <= 30:
            candidates.append((score_ctx(m.start(), 1) + 1, m.start(), (-v, v)))

    if not candidates:
        return []
    best = max(c[0] for c in candidates)
    if best <= 0:
        return []        # a bare unscored number pair is too weak a signal
    pool = [c for c in candidates if c[0] >= best - 1]

    # containment rule for the WINNER: among similarly-scored ranges, a
    # range strictly nested inside another IS the safe range by geometry
    # (-3..3 inside -6..6), independent of wording.
    def _contains(outer, inner):
        return (outer != inner
                and outer[0] <= inner[0] and outer[1] >= inner[1])

    kept = [c for c in pool
            if not any(_contains(c[2], d[2]) for d in pool if d[2] != c[2])]
    winner_pool = sorted(kept or pool, key=lambda c: (-c[0], c[1]))
    out = [winner_pool[0][2]]
    # the losers (e.g. a MAX range) are kept as ordered hints, not tossed —
    # anything positively scored qualifies, even below the winner pool cut
    for c in sorted(candidates, key=lambda c: (-c[0], c[1])):
        if c[0] > 0 and c[2] not in out:
            out.append(c[2])
    return out[:4]


def parse_range_from_text(text):
    """Best single range, or None — the head of parse_range_candidates."""
    cands = parse_range_candidates(text)
    return cands[0] if cands else None


_PREFS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "trigger_prefs.json")
_trigger_prefs = None


def _pref_key(resolved_name):
    return resolved_name.replace("\\", "/").lower()


def _get_trigger_prefs():
    global _trigger_prefs
    if _trigger_prefs is None:
        try:
            with open(_PREFS_FILE, "r", encoding="utf-8") as f:
                _trigger_prefs = json.load(f)
        except Exception:
            _trigger_prefs = {}
        if not isinstance(_trigger_prefs, dict):
            _trigger_prefs = {}
    return _trigger_prefs


def _save_trigger_prefs(prefs):
    try:
        tmp = _PREFS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(prefs, f, indent=1)
        os.replace(tmp, _PREFS_FILE)
    except Exception as e:
        log.warning(f"could not save trigger prefs: {e}")


def trigger_pref_for(resolved_name):
    """The user's saved trigger selection for a lora, or None if they have
    never customized it (= use everything resolvable). Toggled-off customs
    live in the entry's "customs" list and are deliberately NOT returned
    here — this is what executes."""
    ent = _get_trigger_prefs().get(_pref_key(resolved_name))
    if isinstance(ent, dict) and isinstance(ent.get("triggers"), list):
        return [str(t) for t in ent["triggers"]]
    return None


def trigger_pref_customs_for(resolved_name):
    """The saved full custom-trigger roster (on and off), or None."""
    ent = _get_trigger_prefs().get(_pref_key(resolved_name))
    if isinstance(ent, dict) and isinstance(ent.get("customs"), list):
        return [str(t) for t in ent["customs"]]
    return None


def set_trigger_pref_payload(name, triggers, customs=None):
    """Set (list) or clear (None) the saved selection for a lora by fuzzy
    name. customs optionally records the user's full custom-trigger roster
    (including toggled-off ones) so they survive deselection. Pure function
    backing the POST endpoint; returns a payload."""
    try:
        available = folder_paths.get_filename_list("loras")
        resolved, err = resolve_lora(name, available)
        if err is not None:
            return {"ok": False, "error": err}
        prefs = _get_trigger_prefs()
        key = _pref_key(resolved)
        ent = dict(prefs.get(key) or {})
        if triggers is None:
            ent.pop("triggers", None)
            ent.pop("customs", None)
        else:
            clean = [str(t).strip() for t in triggers]
            ent["triggers"] = [t for t in clean if t]
            cc = [str(t).strip() for t in (customs or [])]
            cc = [t for t in cc if t]
            if cc:
                ent["customs"] = cc
            else:
                ent.pop("customs", None)
        if ent:
            prefs[key] = ent
        else:
            prefs.pop(key, None)
        _save_trigger_prefs(prefs)
        if triggers is None:
            log.info(f"trigger prefs cleared for {resolved}")
        else:
            log.info(f"trigger prefs saved for {resolved}: "
                     f"{', '.join(prefs[key]['triggers']) or '(none)'}")
        return {"ok": True, "resolved": resolved, "key": key,
                "triggers": None if triggers is None else prefs[key]["triggers"]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _from_this_machine(request):
    """ComfyUI has no auth: the key and the download mode may only be
    changed by a browser or script running on the server itself."""
    try:
        ip = ipaddress.ip_address((request.remote or "").split("%")[0])
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_loopback


def _send_event(event, data):
    """Push a custom websocket event to the UI; silently no-op outside the
    server (tests, tooling)."""
    try:
        from server import PromptServer
        PromptServer.instance.send_sync(event, data)
    except Exception:
        pass


def _push_rows(node_id, updates):
    """Tell the row UI about relocations / identity stamps:
    {"node_id", "updates": [{"name", "resolved", "how", "fields"}]}."""
    _send_event("lora_wrangler.rows", {"node_id": node_id, "updates": updates})


_dl_last_push = {}


def _push_download(node_id, name, done, total, state="progress", **extra):
    """Download status for the UI header; progress is throttled to ~2/s."""
    now = time.time()
    if state == "progress" and now - _dl_last_push.get(name, 0) < 0.5 \
            and done < total:
        return
    _dl_last_push[name] = now
    data = {"node_id": node_id, "name": name, "done": done, "total": total,
            "state": state}
    data.update(extra)
    _send_event("lora_wrangler.download", data)


class _LoraStackBase:
    """Shared engine. Subclasses set _WITH_CLIP and node metadata."""

    _WITH_CLIP = False
    CATEGORY = "loaders"
    FUNCTION = "apply"

    def __init__(self):
        # path -> (mtime, state_dict). RAM-side cache so editing one
        # strength doesn't re-read every active file from disk.
        self._cache = {}
        # path -> (mtime, [triggers]). Header/sidecar reads are cheap but
        # not free; cache them the same way.
        self._trig_cache = {}

    def _triggers_for(self, path, allow_fetch):
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return []
        key = (path, allow_fetch)
        hit = self._trig_cache.get(key)
        if hit is not None and hit[0] == mtime:
            return hit[1]
        # declared sources first; civitai's declared words beat local
        # dataset-tag inference, which is a guess of last resort
        trigs, src = resolve_declared_src(path)
        if not trigs:
            trigs = cached_civitai_triggers(path)
            if trigs:
                src = "civitai"
        if not trigs and allow_fetch:
            trigs = fetch_civitai_triggers(path)
            if trigs:
                src = "civitai"
        if not trigs:
            # a manifest may declare triggers for this file (matched by hash)
            try:
                ent = _lr.manifest_by_sha(_lr.sha256_of(path))
            except OSError:
                ent = None
            if ent and ent.get("triggers"):
                trigs, src = list(ent["triggers"]), "manifest"
        if not trigs:
            trigs = infer_dataset_tags(path)
            src = "dataset-tags" if trigs else None
        # memoize only declared results: never empties, and never
        # inferences (a later successful fetch must be able to replace an
        # inference without the lora file's mtime changing)
        if trigs and src != "dataset-tags":
            self._trig_cache[key] = (mtime, trigs)
        return trigs

    def _load_lora_file(self, path):
        mtime = os.path.getmtime(path)
        hit = self._cache.get(path)
        if hit is not None and hit[0] == mtime:
            return hit[1]
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        self._cache[path] = (mtime, sd)
        return sd

    def _is_active(self, sm, sc, enabled):
        if not enabled:
            return False
        if self._WITH_CLIP:
            return sm != 0 or sc != 0
        return sm != 0

    def _locate(self, name, stamp, available, node_id=None):
        """Resolution ladder: name -> identity (same file elsewhere) ->
        download. Returns (resolved, how, err); how is "name",
        "relocated by <method>" or "downloaded"; exactly one of resolved
        and err is None."""
        resolved, err = resolve_lora(name, available)
        how = "name"
        stamp = stamp or {}
        if err is None and stamp.get("size"):
            # identity beats name: a same-named file of another size is
            # somebody else's lora
            p = folder_paths.get_full_path("loras", resolved)
            try:
                if p and os.path.getsize(p) != int(stamp["size"]):
                    alt, alt_how = find_local_by_stamp(stamp, available)
                    if alt and alt != resolved:
                        return alt, f"relocated by {alt_how}", None
                    how = "name; size differs from stamp"
            except OSError:
                pass
        if err is None:
            return resolved, how, None

        if not stamp:
            # a name this server lacks may still be known to a manifest
            ent = _lr.manifest_lookup(name)
            stamp = _lr.manifest_stamp(ent) if ent else {}
            if stamp:
                log.info(f"'{name}' not found locally; using its manifest entry")
        if not stamp:
            return None, None, err + (" (no identity stamp on this row and "
                                      "no manifest entry, so it can't be "
                                      "located or downloaded)")
        alt, alt_how = find_local_by_stamp(stamp, available)
        if alt:
            return alt, f"relocated by {alt_how}", None
        if not stamp_downloadable(stamp):
            return None, None, err
        fetch, refusal = download_policy(stamp)
        if fetch is None:
            return None, None, f"{err} ({refusal}; see Settings > LoRA Wrangler)"
        try:
            got = self._download(name, fetch, node_id)
            return got, "downloaded", None
        except DownloadError as e:
            return None, None, (f"'{name}' not found locally and the "
                                f"download failed: {e}")

    def _download(self, name, stamp, node_id=None):
        """Fetch a missing lora during execution, with the node's progress
        bar and a header status in the row UI. Cancel in the UI aborts."""
        pbar = comfy.utils.ProgressBar(1, node_id=node_id)

        def progress(done, total):
            pbar.update_absolute(done, max(total, 1))
            _push_download(node_id, name, done, total)

        def interrupt():
            comfy.model_management.throw_exception_if_processing_interrupted()

        log.info(f"'{name}' is missing; fetching it from Civitai")
        _push_download(node_id, name, 0, 0, state="start")
        try:
            got = download_lora(stamp, target_name=name,
                                progress=progress, interrupt=interrupt)
        except DownloadError as e:
            _push_download(node_id, name, 0, 0, state="error", error=str(e))
            raise
        except BaseException:
            _push_download(node_id, name, 0, 0, state="error",
                           error="cancelled")
            raise
        _push_download(node_id, name, 1, 1, state="done", resolved=got)
        return got

    def _stamp_update(self, name, resolved, how, stamp, path, allow_network):
        """What the row UI should absorb after this row ran: a rename when
        the file was found elsewhere / downloaded, and any identity fields
        the row lacks (hash is cached; civitai ids are fetched once per
        file and only when civitai lookups are allowed)."""
        try:
            ident = identity_for_path(path, allow_network=allow_network)
        except Exception as e:
            log.debug(f"identity for {resolved} failed: {e}")
            ident = None
        fields = stamp_fields(ident) if ident else {}
        stamp = stamp or {}
        # a same-named file whose size disagrees with the stamp and has no
        # identity twin elsewhere is a retrained/replaced local file: the
        # local file wins and the stamp is rewritten to match it
        replace = how.startswith("name;")
        out_fields = {}
        if fields.get("sha") and (replace or not stamp.get("sha")):
            out_fields["sha"] = fields["sha"]
        if fields.get("size") and (replace or not stamp.get("size")):
            out_fields["size"] = fields["size"]
        if fields.get("civitai") and (replace or not stamp.get("version_id")):
            out_fields["civitai"] = fields["civitai"]
        renamed = (resolved != name
                   and (how == "downloaded" or how.startswith("relocated")))
        if not out_fields and not renamed:
            return None
        return {"name": name, "resolved": resolved, "how": how,
                "fields": out_fields, "replace": replace}

    def _apply_stack(self, model, clip, loras, on_error, exclusive,
                     civitai=True, incoming_triggers="", randomize=False,
                     node_id=None):
        entries, errors = parse_stack_text(loras)
        available = folder_paths.get_filename_list("loras")
        wc = self._WITH_CLIP

        # randomize: pick exactly one entry from the enabled pool each run
        # (the toggles define the pool). Supersedes exclusive.
        random_note = ""
        if randomize:
            pool = [i for i, e in enumerate(entries) if e[3]]
            if pool:
                pick = random.choice(pool)
                entries = [(n, sm, sc, i == pick, ann, st)
                           for i, (n, sm, sc, en, ann, st) in enumerate(entries)]
                random_note = f" (random pick: {entries[pick][0]})"
            else:
                random_note = " (random: empty pool, nothing enabled)"
            exclusive = False

        # exclusive: only the first active entry applies
        if exclusive:
            actives = [n for n, sm, sc, en, _ann, _st in entries
                       if self._is_active(sm, sc, en)]
            if len(actives) > 1:
                errors.append(
                    "exclusive mode but multiple entries active: "
                    + ", ".join(actives)
                    + " (only the first applies)"
                )

        pre_errors = list(errors)  # parse errors + exclusive notice:
                                   # these get no inline row below
        report_lines = []
        applied = 0
        inactive = 0
        # seed with triggers chained in from an upstream stack node
        triggers = []
        for t in (incoming_triggers or "").split(","):
            t = t.strip()
            if t and t not in triggers:
                triggers.append(t)

        stamp_updates = []   # relocations / identities for the row UI
        for name, sm, sc, enabled, ann, stamp in entries:
            disp = _fmt_strengths(sm, sc, wc)
            if not self._is_active(sm, sc, enabled):
                inactive += 1
                off = "" if enabled else " (off)"
                report_lines.append(f" [ ] {name} : {disp}{off}")
                continue

            if exclusive and applied >= 1:
                inactive += 1
                report_lines.append(
                    f" [ ] {name} : {disp} (exclusive: skipped)")
                continue

            resolved, how, err = self._locate(name, stamp, available, node_id)
            if how == "downloaded":
                available = folder_paths.get_filename_list("loras")
            if err is not None:
                errors.append(err)
                report_lines.append(f" [!] {name} : {disp}  <- {err}")
                continue

            path = folder_paths.get_full_path("loras", resolved)
            if path is None:
                errors.append(f"'{resolved}' resolved but has no path")
                report_lines.append(f" [!] {name} : {disp}  <- no path")
                continue

            lora_sd = self._load_lora_file(path)
            if wc:
                model, clip = comfy.sd.load_lora_for_models(
                    model, clip, lora_sd, sm, sc
                )
            else:
                # clip=None + clip strength 0 -> pure model patch
                model, _ = comfy.sd.load_lora_for_models(
                    model, None, lora_sd, sm, 0
                )
            applied += 1
            # annotation in the stack text wins; then the user's saved
            # per-lora selection (trigger_prefs.json); then metadata/
            # sidecars; civitai (if enabled) is the last resort and caches
            # so it only ever happens once per file
            if ann is not None:
                trigs = ann
            else:
                pref = trigger_pref_for(resolved)
                trigs = pref if pref is not None \
                    else self._triggers_for(path, civitai)
            for t in trigs:
                if t not in triggers:
                    triggers.append(t)
            note = ""
            if how != "name":
                note = (f"  ({how}"
                        + (f" from '{name}'" if resolved != name else "")
                        + ")")
            report_lines.append(
                f" [x] {resolved} : {disp}{note}"
                + (f"  [triggers: {', '.join(trigs)}]" if trigs else ""))
            upd = self._stamp_update(name, resolved, how, stamp, path, civitai)
            if upd:
                stamp_updates.append(upd)

        if stamp_updates:
            _push_rows(node_id, stamp_updates)
        _lr.flush_cache()

        if errors and on_error == "error":
            raise ValueError(
                "LoRA Wrangler: "
                + "; ".join(errors)
                + " (set on_error to 'skip' to ignore)"
            )

        header = (f"LoRA Wrangler: {applied} active, {inactive} inactive, "
                  f"{len(errors)} problem(s){random_note}")
        for e in pre_errors:
            report_lines.append(f" [!] {e}")
        report = "\n".join([header] + report_lines)
        log.info(header)
        return model, clip, report, ", ".join(triggers)

    # -- shared IS_CHANGED: re-execute when text changes OR when an active
    #    lora file on disk changes (retrained slider with same filename)
    @classmethod
    def IS_CHANGED(cls, loras="", **kwargs):
        try:
            if kwargs.get("randomize"):
                return float("nan")   # re-roll every queue
            sig = [loras, str(kwargs.get("exclusive", ""))]
            try:
                if os.path.isfile(_PREFS_FILE):
                    sig.append(f"prefs:{os.path.getmtime(_PREFS_FILE)}")
            except OSError:
                pass
            entries, _ = parse_stack_text(loras)
            available = folder_paths.get_filename_list("loras")
            for name, sm, sc, enabled, _ann, _stamp in entries:
                if not enabled or (sm == 0 and sc == 0):
                    continue
                resolved, err = resolve_lora(name, available)
                if resolved is None:
                    sig.append(f"{name}:missing")
                    continue
                path = folder_paths.get_full_path("loras", resolved)
                if path is None:
                    continue
                sig.append(f"{resolved}:{os.path.getmtime(path)}")
                for sp in _sidecar_paths(path):
                    if os.path.isfile(sp):
                        sig.append(f"{sp}:{os.path.getmtime(sp)}")
            return "|".join(sig)
        except Exception:
            return float("nan")  # force re-execution if anything went odd


_LORAS_INPUT = ("STRING", {
    "multiline": True,
    "default": "",
    "advanced": True,
    "tooltip": "One 'lora_name : strength' per line (or "
               "'name : model_strength : clip_strength'). '!' prefix or "
               "all-zero strengths = inactive. '#' starts a comment. "
               "Normally managed by the row UI above.",
})

_CIVITAI_INPUT = ("BOOLEAN", {
    "default": True,
    "advanced": True,
    "tooltip": "Look loras up on Civitai by file hash — once per file, "
               "then cached to a sidecar — for trigger words when none are "
               "found locally, and for the Civitai id in each row's "
               "identity stamp (what lets other machines auto-download "
               "it). Turn off for fully offline operation.",
})

_TRIGGERS_CHAIN_INPUT = ("STRING", {
    "forceInput": True,
    "default": "",
    "tooltip": "Chain from an upstream stack node's triggers output. "
               "Incoming triggers come first, this node's active triggers "
               "append after, deduplicated.",
})

_RANDOMIZE_INPUT = ("BOOLEAN", {
    "default": False,
    "advanced": True,
    "tooltip": "Each run, randomly pick exactly ONE lora from the enabled "
               "entries (toggles define the pool) and apply only it, at its "
               "own strength. Supersedes exclusive. Re-rolls every queue.",
})

_ON_ERROR_INPUT = (["error", "skip"], {
    "default": "error",
    "advanced": True,
    "tooltip": "What to do about unknown/ambiguous names, unparseable "
               "lines, or exclusive-mode violations.",
})


class SliderLoraStack(_LoraStackBase):
    """Model-only stack for slider LoRAs. CLIP is never touched."""

    _WITH_CLIP = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "loras": _LORAS_INPUT,
                "on_error": _ON_ERROR_INPUT,
                "civitai_triggers": _CIVITAI_INPUT,
                "randomize": _RANDOMIZE_INPUT,
            },
            "optional": {
                "triggers": _TRIGGERS_CHAIN_INPUT,
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MODEL", "STRING", "STRING")
    RETURN_NAMES = ("model", "report", "triggers")
    DESCRIPTION = ("Model-only multi-LoRA stack with a Power-Lora-Loader "
                   "style row UI. Nodes 2.0 safe, API-editable, zero-cost "
                   "inactive entries. CLIP is never touched. The triggers "
                   "output emits trigger words of active entries.")

    def apply(self, model, loras, on_error="error", civitai_triggers=True,
              randomize=False, triggers="", unique_id=None):
        model, _, report, out_triggers = self._apply_stack(
            model, None, loras, on_error, exclusive=False,
            civitai=civitai_triggers, incoming_triggers=triggers,
            randomize=randomize, node_id=unique_id)
        return (model, report, out_triggers)


class CallsignLoraStack(_LoraStackBase):
    """Model+CLIP stack for normal LoRAs.

    One strength drives both by default; separate_clip_strength switches
    the row UI to independent model/clip fields (the text format
    'name : m : c' works either way). exclusive: only one entry active
    at once, enforced by the backend as well as the UI.
    """

    _WITH_CLIP = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "loras": _LORAS_INPUT,
                "separate_clip_strength": ("BOOLEAN", {
                    "default": False,
                    "advanced": True,
                    "tooltip": "Show independent model and clip strength "
                               "fields per row. Off: one field drives "
                               "both. Display-only — execution always "
                               "uses the values in the text.",
                }),
                "exclusive": ("BOOLEAN", {
                    "default": False,
                    "advanced": True,
                    "tooltip": "Only one entry may be active at once. "
                               "Toggling one on in the UI turns the "
                               "others off; the backend enforces it for "
                               "API edits too.",
                }),
                "on_error": _ON_ERROR_INPUT,
                "civitai_triggers": _CIVITAI_INPUT,
                "randomize": _RANDOMIZE_INPUT,
            },
            "optional": {
                "triggers": _TRIGGERS_CHAIN_INPUT,
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "report", "triggers")
    DESCRIPTION = ("Model+CLIP multi-LoRA stack with a Power-Lora-Loader "
                   "style row UI, optional separate model/clip strengths, "
                   "and optional exclusive (one-active-only) mode. The "
                   "triggers output emits trigger words of active entries.")

    def apply(self, model, clip, loras, separate_clip_strength=False,
              exclusive=False, on_error="error", civitai_triggers=True,
              randomize=False, triggers="", unique_id=None):
        model, clip, report, out_triggers = self._apply_stack(
            model, clip, loras, on_error, exclusive=exclusive,
            civitai=civitai_triggers, incoming_triggers=triggers,
            randomize=randomize, node_id=unique_id)
        return (model, clip, report, out_triggers)


def _store_range_candidates(resolved_name, cands):
    """Persist the fetched candidate list as informational hints."""
    try:
        prefs = _get_trigger_prefs()
        key = _pref_key(resolved_name)
        ent = dict(prefs.get(key) or {})
        ent["range_candidates"] = [[float(a), float(b)] for a, b in cands]
        prefs[key] = ent
        _save_trigger_prefs(prefs)
    except Exception as e:
        log.warning(f"could not store range candidates: {e}")


def range_candidates_for(resolved_name):
    ent = _get_trigger_prefs().get(_pref_key(resolved_name))
    if isinstance(ent, dict):
        rc = ent.get("range_candidates")
        if isinstance(rc, list) and all(
                isinstance(r, list) and len(r) == 2 for r in rc):
            return [[float(r[0]), float(r[1])] for r in rc]
    return None


def range_pref_for(resolved_name):
    """(lo, hi, source) for a lora's saved recommended range, or None."""
    ent = _get_trigger_prefs().get(_pref_key(resolved_name))
    if isinstance(ent, dict):
        r = ent.get("range")
        if (isinstance(r, list) and len(r) == 2
                and all(isinstance(v, (int, float)) for v in r)):
            return float(r[0]), float(r[1]), ent.get("range_src") or "user"
    return None


def set_range_payload(name, rng, source="user"):
    """Set ([lo, hi]) or clear (None) a lora's recommended range.
    A user-set range is never overwritten by a fetched one."""
    try:
        available = folder_paths.get_filename_list("loras")
        resolved, err = resolve_lora(name, available)
        if err is not None:
            return {"ok": False, "error": err}
        prefs = _get_trigger_prefs()
        key = _pref_key(resolved)
        ent = dict(prefs.get(key) or {})
        if rng is None:
            ent.pop("range", None)
            ent.pop("range_src", None)
        else:
            lo, hi = float(rng[0]), float(rng[1])
            if lo > hi:
                lo, hi = hi, lo
            if source == "civitai" and ent.get("range_src") == "user":
                return {"ok": True, "resolved": resolved, "key": key,
                        "range": ent.get("range"), "source": "user",
                        "kept_user": True}
            ent["range"] = [lo, hi]
            ent["range_src"] = source
        if ent:
            prefs[key] = ent
        else:
            prefs.pop(key, None)
        _save_trigger_prefs(prefs)
        if rng is not None:
            log.info(f"range saved for {resolved}: {ent['range']} ({source})")
        return {"ok": True, "resolved": resolved, "key": key,
                "range": ent.get("range"), "source": ent.get("range_src")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def fetch_civitai_range(path, resolved_name):
    """Look for a recommended strength range in the civitai version and
    model descriptions. Saves on success (unless a user-set range exists).
    Returns (lo, hi) or None. Never raises."""
    try:
        sha = _sha256_of(path)
        req = urllib.request.Request(
            "https://civitai.com/api/v1/model-versions/by-hash/" + sha,
            headers={"User-Agent": _lr.USER_AGENT})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
        texts = [data.get("description") or ""]
        mid = data.get("modelId")
        if mid:
            try:
                req2 = urllib.request.Request(
                    f"https://civitai.com/api/v1/models/{mid}",
                    headers={"User-Agent": _lr.USER_AGENT})
                with urllib.request.urlopen(req2, timeout=10) as r2:
                    mdata = json.loads(r2.read().decode("utf-8", "replace"))
                texts.append(mdata.get("description") or "")
            except Exception:
                pass
        cands = parse_range_candidates("\n".join(texts))
        if cands:
            _store_range_candidates(resolved_name, cands)
            set_range_payload(resolved_name, list(cands[0]),
                              source="civitai")
            return cands[0]
        return None
    except Exception as e:
        log.warning(f"civitai range fetch failed for "
                    f"{os.path.basename(path)}: {e}")
        return None


def range_endpoint_payload(name, do_fetch):
    """Saved (and optionally freshly-fetched) recommended range by name."""
    try:
        available = folder_paths.get_filename_list("loras")
        resolved, err = resolve_lora(name, available)
        if err is not None:
            ent = _lr.manifest_lookup(name)
            if ent and ent.get("range"):
                return {"range": list(ent["range"]), "source": "manifest",
                        "resolved": None, "candidates": None}
            return {"range": None, "error": err}
        cur = range_pref_for(resolved)
        if cur is not None and not do_fetch:
            return {"range": [cur[0], cur[1]], "source": cur[2],
                    "resolved": resolved,
                    "candidates": range_candidates_for(resolved)}
        if do_fetch:
            path = folder_paths.get_full_path("loras", resolved)
            if path is not None:
                fetch_civitai_range(path, resolved)
        cur = range_pref_for(resolved)
        if cur is not None:
            return {"range": [cur[0], cur[1]], "source": cur[2],
                    "resolved": resolved,
                    "candidates": range_candidates_for(resolved)}
        return {"range": None, "source": None, "resolved": resolved,
                "candidates": range_candidates_for(resolved)}
    except Exception as e:
        return {"range": None, "error": str(e)}


def triggers_endpoint_payload(name, do_fetch):
    """Known triggers for a lora by (fuzzy) name — used by the row UI's
    trigger popup. Pure function so it's testable without the server."""
    try:
        available = folder_paths.get_filename_list("loras")
        resolved, err = resolve_lora(name, available)
        if err is not None:
            ent = _lr.manifest_lookup(name)
            if ent:
                # not here yet: the manifest's triggers feed the popup
                st = _lr.manifest_stamp(ent)
                return {"triggers": list(ent.get("triggers") or []),
                        "resolved": None, "fetched": False, "selected": None,
                        "customs": None, "source": "manifest",
                        "civitai_url": civitai_page_url(st.get("model_id"),
                                                        st.get("version_id")),
                        "virtual": True, "version": __version__}
            return {"triggers": [], "error": err}
        path = folder_paths.get_full_path("loras", resolved)
        if path is None:
            return {"triggers": [], "error": f"'{resolved}' has no path"}
        trigs, source = resolve_declared_src(path)
        if not trigs:
            trigs = cached_civitai_triggers(path)
            if trigs:
                source = "civitai"
        fetched = False
        if not trigs and do_fetch:
            trigs = fetch_civitai_triggers(path)
            fetched = True
            if trigs:
                source = "civitai"
        if not trigs:
            trigs = infer_dataset_tags(path)
            source = "dataset-tags" if trigs else None
        return {"triggers": trigs, "resolved": resolved, "fetched": fetched,
                "selected": trigger_pref_for(resolved),
                "customs": trigger_pref_customs_for(resolved),
                "source": source, "civitai_url": cached_civitai_page(path),
                "version": __version__}
    except Exception as e:
        return {"triggers": [], "error": str(e)}


def _effective_triggers(n, path):
    """(triggers, source) for a listed lora: saved selection if any, else
    declared local, else cached civitai, else inference. Local only —
    never hashes or touches the network."""
    pref = trigger_pref_for(n)
    if pref is not None:
        return pref, "pref"
    if path is None:
        return [], None
    trigs, src = resolve_declared_src(path)
    if not trigs:
        trigs = cached_civitai_triggers(path)
        if trigs:
            src = "civitai"
    if not trigs:
        trigs = infer_dataset_tags(path)
        src = "dataset-tags" if trigs else None
    return trigs, src


def list_loras_payload(folder=None, search=None, include_triggers=False,
                       include_manifest=False):
    """Filtered listing of the loras folder for API consumers.

    folder: matches a path prefix or any directory segment,
    case-insensitive, / and \\ both fine ("characters" matches
    "Characters\\sub\\x.safetensors").
    search: multi-term AND substring match on the full name.
    include_triggers: adds each lora's EFFECTIVE triggers (saved selection
    if any, else declared local, else cached civitai, else inference) —
    local only, never hashes or touches the network.
    """
    try:
        names = folder_paths.get_filename_list("loras")
        f = (folder or "").replace("\\", "/").strip("/").lower()
        terms = [t for t in (search or "").lower().split() if t]
        out = []
        for n in names:
            nn = n.replace("\\", "/").lower()
            if f:
                dirp = os.path.dirname(nn)
                parts = dirp.split("/") if dirp else []
                if not (nn.startswith(f + "/") or dirp == f or f in parts):
                    continue
            if terms and not all(t in nn for t in terms):
                continue
            out.append(n)
        virtual = []
        if include_manifest:
            # manifest loras this server lacks (by name), same filters
            for ent in _lr.load_manifests().values():
                mn = ent["name"]
                if resolve_lora(mn, names)[1] is None:
                    continue
                nn = mn.replace("\\", "/").lower()
                if f:
                    dirp = os.path.dirname(nn)
                    parts = dirp.split("/") if dirp else []
                    if not (nn.startswith(f + "/") or dirp == f or f in parts):
                        continue
                if terms and not all(t in nn for t in terms):
                    continue
                virtual.append(ent)
        if not include_triggers:
            return {"loras": out + [e["name"] for e in virtual],
                    "virtual": [e["name"] for e in virtual],
                    "version": __version__}
        detailed = []
        for n in out:
            path = folder_paths.get_full_path("loras", n)
            trigs, src = _effective_triggers(n, path)
            rng = range_pref_for(n)
            detailed.append({
                "name": n, "triggers": trigs, "source": src,
                "range": [rng[0], rng[1]] if rng else None,
                "range_source": rng[2] if rng else None,
                "range_candidates": range_candidates_for(n),
            })
        for ent in virtual:
            detailed.append({
                "name": ent["name"],
                "triggers": list(ent.get("triggers") or []),
                "source": "manifest" if ent.get("triggers") else None,
                "range": list(ent["range"]) if ent.get("range") else None,
                "range_candidates": None, "virtual": True,
            })
        return {"loras": detailed, "version": __version__}
    except Exception as e:
        return {"loras": [], "error": str(e), "version": __version__}


def identity_payload(names, allow_network=True):
    """Identity stamp fields for loras by (fuzzy) name — the row UI calls
    this when rows are added and for 'stamp all'. Hashes on first contact
    (cached afterwards); civitai ids need the network once per file."""
    results = {}
    try:
        available = folder_paths.get_filename_list("loras")
    except Exception as e:
        return {"results": {}, "error": str(e), "version": __version__}
    for name in names or []:
        name = str(name)
        try:
            resolved, err = resolve_lora(name, available)
            if err is not None:
                ent = _lr.manifest_lookup(name)
                if ent:
                    fields = _lr.manifest_fields(ent)
                    results[name] = {"resolved": None, "fields": fields,
                                     "on_civitai": "civitai" in fields,
                                     "source": "manifest"}
                else:
                    results[name] = {"error": err}
                continue
            path = folder_paths.get_full_path("loras", resolved)
            if path is None:
                results[name] = {"error": f"'{resolved}' has no path"}
                continue
            ident = identity_for_path(path, allow_network=allow_network)
            results[name] = {"resolved": resolved,
                             "fields": stamp_fields(ident),
                             "on_civitai": bool(ident.get("version_id"))}
        except Exception as e:
            results[name] = {"error": str(e)}
    _lr.flush_cache()
    return {"results": results, "version": __version__}


def check_rows_payload(rows):
    """Pre-flight for the UI: per row, found / relocated / downloadable /
    missing. Local only — never hashes unstamped files or hits the
    network — so it is cheap enough to run before every queue."""
    try:
        available = folder_paths.get_filename_list("loras")
        status = key_status()
        out = []
        for r in rows or []:
            name = str((r or {}).get("name") or "")
            fields, _ = parse_comment_fields(str((r or {}).get("comment") or ""))
            stamp = parse_stamp(fields) or {}
            resolved, err = resolve_lora(name, available)
            if err is None:
                out.append({"name": name, "status": "found",
                            "resolved": resolved})
                continue
            if not stamp:
                # same fallback the run uses: an unstamped row named after
                # a manifest entry borrows the manifest's identity
                ent = _lr.manifest_lookup(name)
                stamp = _lr.manifest_stamp(ent) if ent else {}
            alt, how = (find_local_by_stamp(stamp, available)
                        if stamp else (None, None))
            if alt:
                out.append({"name": name, "status": "relocated",
                            "resolved": alt, "how": how})
                continue
            if stamp_downloadable(stamp):
                fetch, refusal = download_policy(stamp)
                out.append({"name": name, "status": "downloadable",
                            "allowed": fetch is not None, "reason": refusal,
                            "auto": status["auto_download"],
                            "key": status["set"]})
                continue
            out.append({"name": name, "status": "missing", "error": err,
                        "stamped": bool(stamp)})
        return {"rows": out, "settings": status, "version": __version__}
    except Exception as e:
        return {"rows": [], "error": str(e), "version": __version__}


def manifest_payload():
    """Every manifest entry plus whether this server has it by name. The
    row UI merges entries with local=False into its picker as cloud
    items; picking one adds a stamped row that downloads on first run."""
    try:
        available = folder_paths.get_filename_list("loras")
        entries = []
        for ent in _lr.load_manifests().values():
            resolved, err = resolve_lora(ent["name"], available)
            e = dict(ent)
            e["local"] = err is None
            e["resolved"] = resolved
            entries.append(e)
        entries.sort(key=lambda e: e["name"].lower())
        return {"entries": entries, "sources": _lr.manifest_sources(),
                "version": __version__}
    except Exception as e:
        return {"entries": [], "error": str(e), "version": __version__}


def manifest_save_payload(name, data):
    if not _lr.manifest_push_allowed():
        return {"ok": False, "disabled": True,
                "error": "manifest push is disabled on this server; copy "
                         "the file into the node's manifests/ folder, or set "
                         "LORA_WRANGLER_ALLOW_MANIFEST_PUSH=1 in the server's "
                         "environment to allow it"}
    try:
        path, count = _lr.save_manifest(name, data)
        log.info(f"manifest saved: {path} ({count} loras)")
        return {"ok": True, "path": path, "entries": count}
    except Exception as e:
        return {"ok": False, "error": str(e)}


_export_state = {"running": False, "scanned": 0, "total": 0, "kept": 0,
                 "skipped": 0, "cancel": False, "started": None}


def export_status_payload():
    """Progress of the export in flight (or the last one)."""
    s = dict(_export_state)
    s["elapsed"] = round(time.time() - s["started"], 1) if s["started"] else None
    return s


def export_cancel_payload():
    """Ask a running export to stop; it returns what it has so far."""
    _export_state["cancel"] = True
    return {"ok": True, "running": _export_state["running"]}


EXPORT_UNFILTERED_LIMIT = 500   # bigger banks must opt in with all=1


def manifest_export_payload(folder=None, search=None, do_hash=True,
                            network=True, only_info=False, only_known=False,
                            export_all=False):
    """Build a manifest from THIS server's loras: names relative to the
    loras root, identity (hash / Civitai id / size), effective triggers
    and saved ranges. do_hash hashes files not yet in the cache (slow on
    first contact, cached afterwards); network fetches Civitai ids once
    per file. only_info keeps just loras with a .civitai.info sidecar (the
    ones that have been looked up / used); only_known keeps just loras
    already in the hash cache. Both are cheap pre-filters applied before
    any per-file work. Progress goes to the console and to
    GET /lora_wrangler/manifest/export/status; POST .../export/cancel
    stops it. The result is a valid manifest file as-is."""
    names = []
    loras = []
    skipped = 0
    cancelled = False
    try:
        all_names = list_loras_payload(folder, search).get("loras") or []
        # the filters are cheap (an isfile / cache check per lora), so apply
        # them up front: the first console line then says how many will
        # actually be processed
        if only_info or only_known:
            for n in all_names:
                path = folder_paths.get_full_path("loras", n)
                has_info = path is not None and os.path.isfile(
                    os.path.splitext(path)[0] + ".civitai.info")
                known = path is not None and _lr.cached_sha256(path) is not None
                if (only_info and has_info) or (only_known and known):
                    names.append(n)
                else:
                    skipped += 1
        else:
            names = list(all_names)
            if len(names) > EXPORT_UNFILTERED_LIMIT and not export_all:
                msg = (f"refusing to export {len(names)} loras unfiltered "
                       f"(more than {EXPORT_UNFILTERED_LIMIT}): add info=1 "
                       f"(loras with a .civitai.info sidecar) or known=1 "
                       f"(already hashed), or all=1 to really process "
                       f"every file")
                log.warning(f"manifest export: {msg}")
                return {"loras": [], "error": msg, "total": len(names),
                        "version": __version__}
        _export_state.update(running=True, scanned=0, total=len(names),
                             kept=0, skipped=skipped, cancel=False,
                             started=time.time())
        log.info(f"manifest export: {len(names)} of {len(all_names)} loras"
                 + (" (info sidecar only)" if only_info else "")
                 + (" (hashed only)" if only_known else "")
                 + (" (unfiltered)" if not (only_info or only_known) else "")
                 + f" [hash={'1' if do_hash else '0'} "
                   f"network={'1' if network else '0'}]")
        for i, n in enumerate(names, 1):
            _export_state["scanned"] = i
            if _export_state["cancel"]:
                cancelled = True
                log.info("manifest export: cancelled")
                break
            path = folder_paths.get_full_path("loras", n)
            rec = {"name": n.replace("\\", "/")}
            if path is not None:
                try:
                    if do_hash:
                        ident = identity_for_path(path, allow_network=network)
                    else:
                        sha = _lr.cached_sha256(path)
                        ident = {"sha": sha, "size": os.path.getsize(path),
                                 "version_id": None, "model_id": None}
                        ent = (lookup_by_hash(sha, allow_network=False)
                               if sha else None)
                        if ent and ent.get("version_id"):
                            ident["version_id"] = ent["version_id"]
                            ident["model_id"] = ent.get("model_id")
                        else:
                            sc = _lr.sidecar_identity(path)
                            if sc.get("version_id"):
                                ident["version_id"] = sc["version_id"]
                                ident["model_id"] = sc.get("model_id")
                    rec.update(stamp_fields(ident))
                    if rec.get("size"):
                        rec["size"] = int(rec["size"])
                except Exception as e:
                    rec["error"] = str(e)
                trigs, _src = _effective_triggers(n, path)
                if trigs:
                    rec["triggers"] = trigs
                rng = range_pref_for(n)
                if rng:
                    rec["range"] = [rng[0], rng[1]]
            loras.append(rec)
            _export_state["kept"] = len(loras)
            if len(loras) % 25 == 0:
                log.info(f"manifest export: {i}/{len(names)} done")
        log.info(f"manifest export: {len(loras)} loras"
                 + (f", {skipped} skipped by filter" if skipped else "")
                 + (" (cancelled, partial)" if cancelled else ""))
        return {"name": "export",
                "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "source": "comfyui-lora-wrangler", "loras": loras,
                "skipped": skipped, "cancelled": cancelled,
                "version": __version__}
    except Exception as e:
        return {"loras": loras, "error": str(e), "version": __version__}
    finally:
        _export_state["running"] = False
        _lr.flush_cache()


def download_payload(name, comment):
    """Download one stamped row on demand (API). Same policy as a run: off
    downloads nothing, manifest mode only what the manifests list."""
    try:
        fields, _ = parse_comment_fields(comment or "")
        stamp = parse_stamp(fields) or {}
        if not stamp:
            ent = _lr.manifest_lookup(name)
            stamp = _lr.manifest_stamp(ent) if ent else {}
        if not stamp_downloadable(stamp):
            return {"ok": False, "error": "row has no identity stamp"}
        fetch, refusal = download_policy(stamp)
        if fetch is None:
            return {"ok": False, "disabled": True, "error": refusal}
        _push_download(None, name, 0, 0, state="start")
        got = download_lora(
            fetch, target_name=name,
            progress=lambda d, t: _push_download(None, name, d, t))
        _push_download(None, name, 1, 1, state="done", resolved=got)
        return {"ok": True, "resolved": got}
    except DownloadError as e:
        _push_download(None, name, 0, 0, state="error", error=str(e))
        return {"ok": False, "error": str(e)}
    except Exception as e:
        _push_download(None, name, 0, 0, state="error", error=str(e))
        return {"ok": False, "error": str(e)}


try:
    import asyncio
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/lora_wrangler/triggers")
    async def _lw_triggers(request):
        name = request.query.get("name", "")
        do_fetch = request.query.get("fetch", "0") == "1"
        # hashing + the civitai round-trip are blocking; keep them off
        # the server's event loop
        payload = await asyncio.get_event_loop().run_in_executor(
            None, triggers_endpoint_payload, name, do_fetch)
        return web.json_response(payload)

    @PromptServer.instance.routes.post("/lora_wrangler/triggers")
    async def _lw_set_triggers(request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "bad json"})
        payload = await asyncio.get_event_loop().run_in_executor(
            None, set_trigger_pref_payload,
            data.get("name", ""), data.get("triggers", None),
            data.get("customs", None))
        return web.json_response(payload)

    @PromptServer.instance.routes.get("/lora_wrangler/loras")
    async def _lw_loras(request):
        folder = request.query.get("folder") or None
        search = request.query.get("search") or None
        include = request.query.get("triggers", "0") == "1"
        manifest = request.query.get("manifest", "0") == "1"
        payload = await asyncio.get_event_loop().run_in_executor(
            None, list_loras_payload, folder, search, include, manifest)
        return web.json_response(payload)

    @PromptServer.instance.routes.get("/lora_wrangler/prefs")
    async def _lw_prefs(request):
        prefs = _get_trigger_prefs()
        ranges = {}
        for k, ent in prefs.items():
            if isinstance(ent, dict):
                r = ent.get("range")
                if isinstance(r, list) and len(r) == 2:
                    ranges[k] = r
        keys = [k for k, ent in prefs.items()
                if isinstance(ent, dict) and "triggers" in ent]
        return web.json_response({"keys": keys, "ranges": ranges})

    @PromptServer.instance.routes.get("/lora_wrangler/range")
    async def _lw_range(request):
        name = request.query.get("name", "")
        do_fetch = request.query.get("fetch", "0") == "1"
        payload = await asyncio.get_event_loop().run_in_executor(
            None, range_endpoint_payload, name, do_fetch)
        return web.json_response(payload)

    @PromptServer.instance.routes.post("/lora_wrangler/range")
    async def _lw_set_range(request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "bad json"})
        payload = await asyncio.get_event_loop().run_in_executor(
            None, set_range_payload, data.get("name", ""),
            data.get("range", None))
        return web.json_response(payload)

    @PromptServer.instance.routes.get("/lora_wrangler/civitai_key")
    async def _lw_key_status(request):
        # never returns the key itself — only whether one is set
        return web.json_response(key_status())

    @PromptServer.instance.routes.post("/lora_wrangler/civitai_key")
    async def _lw_set_key(request):
        if not _from_this_machine(request):
            return web.json_response({
                "ok": False, "local_only": True,
                "error": "the Civitai key and auto-download mode can only be "
                         "changed from the server machine itself, or with "
                         "the CIVITAI_API_KEY / LORA_WRANGLER_AUTO_DOWNLOAD "
                         "environment variables"})
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "bad json"})
        try:
            if "auto_download" in data:
                payload = set_auto_download(str(data["auto_download"]))
            else:
                payload = set_api_key(data.get("key"))
        except OSError as e:
            payload = {"ok": False, "error": str(e)}
        return web.json_response(payload)

    @PromptServer.instance.routes.post("/lora_wrangler/identity")
    async def _lw_identity(request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"results": {}, "error": "bad json"})
        payload = await asyncio.get_event_loop().run_in_executor(
            None, identity_payload, data.get("names") or [],
            bool(data.get("network", True)))
        return web.json_response(payload)

    @PromptServer.instance.routes.post("/lora_wrangler/check")
    async def _lw_check(request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"rows": [], "error": "bad json"})
        payload = await asyncio.get_event_loop().run_in_executor(
            None, check_rows_payload, data.get("rows") or [])
        return web.json_response(payload)

    @PromptServer.instance.routes.post("/lora_wrangler/download")
    async def _lw_download(request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "bad json"})
        payload = await asyncio.get_event_loop().run_in_executor(
            None, download_payload, str(data.get("name") or ""),
            str(data.get("comment") or ""))
        return web.json_response(payload)

    @PromptServer.instance.routes.get("/lora_wrangler/manifest")
    async def _lw_manifest(request):
        payload = await asyncio.get_event_loop().run_in_executor(
            None, manifest_payload)
        return web.json_response(payload)

    @PromptServer.instance.routes.post("/lora_wrangler/manifest")
    async def _lw_manifest_save(request):
        # {"name": "mygame", "manifest": {...}} or the manifest itself
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "bad json"})
        name, body = "default", data
        if isinstance(data, dict) and "manifest" in data:
            name = data.get("name") or "default"
            body = data["manifest"]
        elif isinstance(data, dict) and data.get("name") and "loras" in data:
            name = data["name"]
        payload = await asyncio.get_event_loop().run_in_executor(
            None, manifest_save_payload, name, body)
        return web.json_response(payload)

    @PromptServer.instance.routes.get("/lora_wrangler/manifest/export")
    async def _lw_manifest_export(request):
        folder = request.query.get("folder") or None
        search = request.query.get("search") or None
        do_hash = request.query.get("hash", "1") == "1"
        network = request.query.get("network", "1") == "1"
        only_info = request.query.get("info", "0") == "1"
        only_known = request.query.get("known", "0") == "1"
        export_all = request.query.get("all", "0") == "1"
        log.info(f"manifest export requested: {dict(request.query)}")
        payload = await asyncio.get_event_loop().run_in_executor(
            None, manifest_export_payload, folder, search, do_hash, network,
            only_info, only_known, export_all)
        return web.json_response(payload)

    @PromptServer.instance.routes.get("/lora_wrangler/manifest/export/status")
    async def _lw_manifest_export_status(request):
        return web.json_response(export_status_payload())

    @PromptServer.instance.routes.post("/lora_wrangler/manifest/export/cancel")
    async def _lw_manifest_export_cancel(request):
        return web.json_response(export_cancel_payload())
except Exception:
    pass  # not running inside the ComfyUI server (tests, tooling)


NODE_CLASS_MAPPINGS = {
    "SliderLoraStack": SliderLoraStack,
    "CallsignLoraStack": CallsignLoraStack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SliderLoraStack": "LoRA Wrangler: Slider Stack (model only)",
    "CallsignLoraStack": "LoRA Wrangler: LoRA Stack (model+clip)",
}
