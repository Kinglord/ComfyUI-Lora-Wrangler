// LoRA Wrangler — the stack nodes' row UI, mounted as a DOM widget.
//
// HTML via addDOMWidget (not litegraph canvas drawing), which is why it
// works under Nodes 2.0. Source of truth stays the backend "loras" STRING
// widget; every UI change re-serializes into it, so the API surface is
// unchanged and if this file fails the node falls back to the text box.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASSES = new Set(["SliderLoraStack", "CallsignLoraStack"]);
const UI_VERSION = "1.0.0";
console.log("[LoRA Wrangler] UI v" + UI_VERSION);
const NUM = "[-+]?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][-+]?\\d+)?";
const LINE_RE = new RegExp(
  "^(.+?)\\s*[:=@]\\s*(" + NUM + ")(?:\\s*[:=@]\\s*(" + NUM + "))?\\s*$");
const TAG_RE = /^<\s*lora\s*:\s*(.+?)\s*>$/i;
const MAX_LIST_ITEMS = 400;
const CIVITAI_SITE = "https://civitai.red";   // civitai.com hides NSFW models

// ---------------------------------------------------------------- text <-> rows

function parseText(text) {
  // Comments are preserved: trailing comments stay attached to their row,
  // and standalone comment/blank-adjacent lines survive as passthrough rows
  // (trigger annotations like "# triggers: x, y" live in comments, so
  // dropping them on a UI edit would eat data).
  const rows = [];
  for (let raw of (text || "").split("\n")) {
    const hashIdx = raw.indexOf("#");
    const comment = hashIdx >= 0 ? raw.slice(hashIdx) : "";
    let line = (hashIdx >= 0 ? raw.slice(0, hashIdx) : raw).trim();
    if (!line || line.startsWith("//")) {
      if (comment || line.startsWith("//"))
        rows.push({ passthrough: comment ? comment : line });
      continue;
    }
    let on = true;
    if (line.startsWith("!")) { on = false; line = line.slice(1).trim(); }
    const tag = line.match(TAG_RE);
    if (tag) line = tag[1];
    const m = line.match(LINE_RE);
    if (!m) { rows.push({ passthrough: raw }); continue; }
    const strength = parseFloat(m[2]);
    if (!isFinite(strength)) { rows.push({ passthrough: raw }); continue; }
    let strengthClip = m[3] !== undefined ? parseFloat(m[3]) : strength;
    if (!isFinite(strengthClip)) strengthClip = strength;
    rows.push({ name: m[1].trim(), strength, strengthClip, on, comment });
  }
  return rows;
}

function renderText(rows) {
  return rows
    .filter((r) => r.name || r.passthrough !== undefined)
    .map((r) => {
      if (r.passthrough !== undefined) return r.passthrough;
      const sc = r.strengthClip !== undefined ? r.strengthClip : r.strength;
      const tail = fmt(sc) !== fmt(r.strength)
        ? `${fmt(r.strength)} : ${fmt(sc)}`
        : fmt(r.strength);
      return `${r.on ? "" : "!"}${r.name} : ${tail}` +
        (r.comment ? `  ${r.comment}` : "");
    })
    .join("\n");
}

// Row comments are ';'-separated "key: value" fields plus free text
// (mirrors parse_comment_fields in lora_wrangler.py):
//   # triggers: a, b; sha: 0123456789ab; civitai: 123456@654321; size: 144703488
// 'triggers' is the per-workflow override; sha/civitai/size/url are the
// identity stamp that lets another machine locate or download the file.
const FIELD_RE = /^\s*([A-Za-z_][\w-]*)\s*:\s*(.*?)\s*$/;
const FIELD_ORDER = ["triggers", "sha", "civitai", "size", "url"];
const STAMP_KEYS = ["sha", "civitai", "size", "url"];
const TRIG_KEYS = new Set(["triggers", "trigger", "trig"]);

function parseCommentFields(comment) {
  const text = (comment || "").replace(/^\s*#/, "");
  const fields = {};
  const free = [];
  for (const seg of text.split(";")) {
    if (!seg.trim()) continue;
    const m = seg.match(FIELD_RE);
    if (m && !m[2].startsWith("//")) {         // "https://x" is prose
      let k = m[1].toLowerCase();
      if (TRIG_KEYS.has(k)) k = "triggers";
      fields[k] = m[2];
    } else {
      free.push(seg.trim());
    }
  }
  return { fields, free };
}

function renderCommentFields(fields, free) {
  const parts = [];
  for (const k of FIELD_ORDER) {
    const v = fields[k];
    if (v === undefined || v === null) continue;
    const s = String(v).trim();
    if (k === "triggers" || s) parts.push((k + ": " + s).replace(/\s+$/, ""));
  }
  for (const k in fields) {
    if (FIELD_ORDER.includes(k)) continue;
    const v = fields[k];
    if (v !== undefined && v !== null && String(v).trim())
      parts.push(k + ": " + String(v).trim());
  }
  for (const s of free || []) if (s) parts.push(s);
  return parts.length ? "# " + parts.join("; ") : "";
}

// null = no annotation (use all known); [] = explicitly none
function parseTriggerAnn(comment) {
  const { fields } = parseCommentFields(comment);
  if (!("triggers" in fields)) return null;
  return fields.triggers.split(",").map((t) => t.trim()).filter(Boolean);
}

// Set (list) or clear (null) the triggers field; every other field and any
// free text survive.
function withTriggerAnn(comment, list) {
  const { fields, free } = parseCommentFields(comment);
  if (list === null) delete fields.triggers;
  else fields.triggers = list.map((t) => String(t).replace(/;/g, " ")).join(", ");
  return renderCommentFields(fields, free);
}

// Fill in identity fields the row lacks; with replace=true (a retrained
// local file) the given fields overwrite the old stamp.
function withStampFields(comment, stamp, replace = false) {
  const { fields, free } = parseCommentFields(comment);
  let changed = false;
  for (const k of STAMP_KEYS) {
    if (!stamp || !stamp[k]) continue;
    if ((replace && fields[k] !== stamp[k]) || !fields[k]) {
      fields[k] = stamp[k];
      changed = true;
    }
  }
  return changed ? renderCommentFields(fields, free) : (comment || "");
}

function withoutStampFields(comment) {
  const { fields, free } = parseCommentFields(comment);
  for (const k of STAMP_KEYS) delete fields[k];
  return renderCommentFields(fields, free);
}

function hasStamp(comment) {
  const f = parseCommentFields(comment).fields;
  return !!(f.sha || f.civitai || f.url);
}

// Civitai page for a stamp-format id ("modelId@versionId", a bare
// version id, or an AIR urn), or null.
function civitaiPageUrl(id) {
  const m = /^(?:urn:air:[^:]*:[^:]*:civitai:)?(?:(\d+)@)?(\d+)$/i
    .exec((id || "").trim());
  if (!m) return null;
  return m[1]
    ? `${CIVITAI_SITE}/models/${m[1]}?modelVersionId=${m[2]}`
    : `${CIVITAI_SITE}/model-versions/${m[2]}`;
}

function baseName(n) {
  return (n || "").replace(/\\/g, "/").split("/").pop();
}

function fmt(v) {
  const s = v.toFixed(2);
  return s === "-0.00" ? "0.00" : s;
}

// One strength step (0.05, or 0.25 for big steps), kept inside bounds.
function stepStrength(v, dir, big, bounds) {
  let n = Math.round((v + dir * (big ? 0.25 : 0.05)) * 100) / 100;
  if (bounds) n = Math.min(bounds[1], Math.max(bounds[0], n));
  return n;
}

// The wheel steps a value only while its field has focus. Unfocused, the
// canvas keeps the wheel; the frontend honours data-capture-wheel for that,
// and still zooms on ctrl/cmd+wheel. Mouse wheels step once per notch,
// trackpads once per ~50px of scrolling.
function attachWheelStep(el, onStep) {
  el.dataset.captureWheel = "true";
  let acc = 0;
  el.addEventListener("wheel", (e) => {
    if (document.activeElement !== el || e.ctrlKey || e.metaKey) return;
    if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) return;
    e.preventDefault();
    e.stopPropagation();
    let dy = e.deltaY;
    if (e.deltaMode === 0 && Math.abs(dy) < 50) {
      acc += dy;
      if (Math.abs(acc) < 50) return;
      dy = acc;
    }
    acc = 0;
    onStep(dy < 0 ? 1 : -1);
  }, { passive: false });
}

// A tick at 0 on slider bars (neutral for most slider LoRAs); Chrome also
// snaps to it when you drag close.
const ZERO_TICK_ID = "lw-zero-tick";
function ensureZeroTick() {
  if (document.getElementById(ZERO_TICK_ID)) return;
  const dl = document.createElement("datalist");
  dl.id = ZERO_TICK_ID;
  const opt = document.createElement("option");
  opt.value = "0";
  dl.appendChild(opt);
  document.body.appendChild(dl);
}

// ---------------------------------------------------------------- trigger prefs

let prefKeys = new Set();
let prefRanges = {};
let prefKeysPromise = null;
function loadPrefKeys() {
  if (!prefKeysPromise) {
    prefKeysPromise = api
      .fetchApi("/lora_wrangler/prefs")
      .then((r) => r.json())
      .then((d) => {
        prefKeys = new Set(d.keys || []);
        prefRanges = d.ranges || {};
      })
      .catch(() => {});
  }
  return prefKeysPromise;
}
function rangeFor(name) {
  const n = normName(name);
  const b = n.split("/").pop();
  for (const k in prefRanges) {
    const kk = normName(k);
    if (kk === n || kk.split("/").pop() === b) return prefRanges[k];
  }
  return null;
}
function normName(n) {
  n = (n || "").toLowerCase().replace(/\\/g, "/");
  return n.replace(/\.(safetensors|sft|st|pt|pth|ckpt)$/, "");
}
function hasPref(name) {
  const n = normName(name);
  const b = n.split("/").pop();
  for (const k of prefKeys) {
    const kk = normName(k);
    if (kk === n || kk.split("/").pop() === b) return true;
  }
  return false;
}

// ---------------------------------------------------------------- civitai / auto-download

const SETTING_AUTO = "LoraWrangler.AutoDownloadMode";
const SETTING_KEY = "LoraWrangler.CivitaiKey";
const SETTING_FOLDER = "LoraWrangler.DownloadFolder";
const SETTING_MATCH = "LoraWrangler.MatchFolders";
const SETTING_SLIDERS = "LoraWrangler.SliderBars";
const DEFAULT_SLIDER_RANGE = [-3, 3];   // slider bar ends when no range is set
const KEY_MASK = "\u2022\u2022\u2022\u2022\u2022\u2022\u2022\u2022";

function toast(detail, severity = "info", life = 5000) {
  try {
    app.extensionManager?.toast?.add?.({
      severity, summary: "LoRA Wrangler", detail, life });
  } catch (e) {}
  console.log("[LoRA Wrangler] " + detail);
}

// The key and the auto-download mode are server-side state: the settings
// panel only mirrors them. Changes are POSTed to the node, which accepts
// them only from the server machine itself; anything the server refuses is
// reverted in the panel. The key is shown as a mask (dots + last 4).
let syncing = false;
async function setQuietly(id, value) {
  syncing = true;
  try { await app.ui.settings.setSettingValue(id, value); }
  finally { syncing = false; }
}

async function postServerSetting(body) {
  const r = await api.fetchApi("/lora_wrangler/civitai_key", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return r.json();
}

async function onKeyChange(value, oldValue) {
  if (syncing || value === undefined || value === null) return;
  const v = String(value).trim();
  if (v.startsWith(KEY_MASK)) return;                 // already masked
  if (v === "" && !(oldValue && String(oldValue).trim())) return;
  try {
    const d = await postServerSetting({ key: v });
    if (d.ok) {
      await setQuietly(SETTING_KEY, d.set ? KEY_MASK + " " + (d.hint || "") : "");
      toast(d.set ? "Civitai API key saved" : "Civitai API key cleared",
        "success");
      return;
    }
    toast("Civitai key not saved: " + (d.error || "unknown error"),
      "error", 12000);
  } catch (e) {
    toast("Civitai key not saved: " + e, "error", 8000);
  }
  await setQuietly(SETTING_KEY, oldValue ?? "");
}

async function onAutoChange(value, oldValue) {
  if (syncing || value === undefined || value === null) return;
  try {
    const d = await postServerSetting({ auto_download: value });
    if (d.ok) {
      toast("Auto-download: " + AUTO_LABELS[d.auto_download_mode], "success");
      return;
    }
    toast("Auto-download not changed: " + (d.error || "unknown error"),
      "error", 12000);
  } catch (e) {
    toast("Auto-download not changed: " + e, "error", 8000);
  }
  await setQuietly(SETTING_AUTO, oldValue ?? "off");
}

// show what the server actually uses, including env var overrides
async function syncServerSettings() {
  try {
    const r = await api.fetchApi("/lora_wrangler/civitai_key");
    if (!r.ok) return;
    const d = await r.json();
    await setQuietly(SETTING_AUTO, d.auto_download_mode || "off");
    await setQuietly(SETTING_KEY, d.set ? KEY_MASK + " " + (d.hint || "") : "");
  } catch (e) {}
}

const AUTO_LABELS = {
  off: "Off",
  manifest: "Manifest LoRAs only",
  any: "Any stamped LoRA",
};

const SETTINGS = [
  {
    id: SETTING_AUTO,
    name: "Auto-download missing LoRAs",
    type: "combo",
    options: Object.entries(AUTO_LABELS).map(([value, text]) => ({ text, value })),
    defaultValue: "off",
    category: ["LoRA Wrangler", "Civitai", "Auto-download"],
    tooltip: "When a row's LoRA isn't on this server, fetch it during the " +
      "run. 'Manifest LoRAs only' fetches only LoRAs listed in this " +
      "server's manifests, which is the safe choice for any server others " +
      "can reach. 'Any stamped LoRA' fetches whatever a workflow names. " +
      "Can only be changed from the server machine itself.",
    onChange: onAutoChange,
  },
  {
    id: SETTING_KEY,
    name: "Civitai API key",
    type: "text",
    defaultValue: "",
    category: ["LoRA Wrangler", "Civitai", "API key"],
    tooltip: "Create one under Account Settings > API Keys on civitai.com. " +
      "Stored on the server, never in workflows; shown masked once saved. " +
      "Clear the field to remove it. Can only be changed from the server " +
      "machine itself.",
    onChange: onKeyChange,
  },
  {
    id: SETTING_FOLDER,
    name: "Download folder (under models/loras)",
    type: "text",
    defaultValue: "auto_download",
    category: ["LoRA Wrangler", "Civitai", "Download folder"],
    tooltip: "Downloads land here so you can sort them later. The node " +
      "finds them by hash wherever you move them afterwards.",
  },
  {
    id: SETTING_MATCH,
    name: "Match workflow folders",
    type: "boolean",
    defaultValue: true,
    category: ["LoRA Wrangler", "Civitai", "Match folders"],
    tooltip: "If the workflow names the lora with a subfolder (e.g. " +
      "NSFW/foo) and that folder already exists in your loras folder, " +
      "download straight into it instead of the download folder - the " +
      "two machines evidently share the same organisation.",
  },
  {
    id: SETTING_SLIDERS,
    name: "Show slider bars on the Slider Stack",
    type: "boolean",
    defaultValue: true,
    category: ["LoRA Wrangler", "Slider Stack", "Slider bars"],
    tooltip: "A drag bar under each slider row. Its ends are the LoRA's " +
      "recommended range (set or fetch one with the T button), or -3 to 3 " +
      "when none is known.",
    onChange: () => controllers.forEach((c) => c.refresh()),
  },
];

// node id -> {applyRowUpdates, setStatus}; the backend addresses the row
// UI by the executing node's id
const controllers = new Map();

api.addEventListener("lora_wrangler.rows", (ev) => {
  const d = ev.detail || {};
  const c = controllers.get(String(d.node_id));
  if (c) c.applyRowUpdates(d.updates || []);
});

api.addEventListener("lora_wrangler.download", (ev) => {
  const d = ev.detail || {};
  const c = d.node_id != null ? controllers.get(String(d.node_id)) : null;
  const mb = (b) => Math.round((b || 0) / 1048576);
  let text = "";
  if (d.state === "start") {
    text = "downloading " + baseName(d.name) + "\u2026";
  } else if (d.state === "progress") {
    const pct = d.total ? Math.round((100 * d.done) / d.total) : 0;
    text = "downloading " + baseName(d.name) + " " + pct + "%" +
      (d.total ? " of " + mb(d.total) + " MB" : "");
  } else if (d.state === "done") {
    toast("Downloaded " + baseName(d.name) + " \u2192 " + d.resolved, "success");
  } else if (d.state === "error") {
    toast("Download failed for " + baseName(d.name) + ": " + d.error,
      "error", 12000);
  }
  if (c) c.setStatus(text);
});

// Before every queue: a cheap local check of every active row in every
// stack node, so the user hears about missing loras (and what will be
// downloaded) up front instead of from a failed run.
async function preflight() {
  try {
    const rowsAll = [];
    for (const node of app.graph?._nodes || []) {
      const cls = node.comfyClass ?? node.constructor?.comfyClass;
      if (!NODE_CLASSES.has(cls)) continue;
      if (node.mode === 2 || node.mode === 4) continue;   // muted / bypassed
      const w = (node.widgets || []).find((x) => x.name === "loras");
      if (!w) continue;
      for (const r of parseText(w.value)) {
        if (r.passthrough !== undefined || !r.on) continue;
        const sc = r.strengthClip !== undefined ? r.strengthClip : r.strength;
        if (r.strength === 0 && sc === 0) continue;
        rowsAll.push({ name: r.name, comment: r.comment || "" });
      }
    }
    if (!rowsAll.length) return;
    const r = await api.fetchApi("/lora_wrangler/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rows: rowsAll }),
    });
    if (!r.ok) return;
    const d = await r.json();
    const list = (xs) => xs.map((x) => baseName(x.name)).join(", ");
    const rows = d.rows || [];
    const dl = rows.filter((x) => x.status === "downloadable" && x.allowed);
    const refused = rows.filter((x) => x.status === "downloadable" && !x.allowed);
    const missing = rows.filter((x) => x.status === "missing");
    const reloc = rows.filter((x) => x.status === "relocated");
    const s = d.settings || {};
    if (refused.length) {
      toast(refused.length + " lora(s) missing and not downloadable here (" +
        refused[0].reason + "): " + list(refused), "warn", 12000);
    }
    if (dl.length) {
      if (!s.set) {
        toast(dl.length + " lora(s) missing \u2014 add a Civitai API key in " +
          "Settings > LoRA Wrangler if the download is refused: " + list(dl),
          "warn", 12000);
      } else {
        toast(dl.length + " lora(s) will be downloaded during this run: " +
          list(dl), "info", 8000);
      }
    }
    if (missing.length) {
      toast(missing.length + " lora(s) missing and without an identity " +
        "stamp, so they cannot be located or downloaded: " + list(missing),
        "warn", 12000);
    }
    if (reloc.length) {
      toast(reloc.length + " lora(s) found under another name; the rows " +
        "will be updated after the run: " + list(reloc), "info", 6000);
    }
  } catch (e) {
    console.warn("[LoRA Wrangler] preflight failed", e);
  }
}

// ---------------------------------------------------------------- lora list

let loraListPromise = null;
function getLoraList() {
  if (!loraListPromise) {
    loraListPromise = api
      .fetchApi("/models/loras")
      .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then((d) => (Array.isArray(d) ? d : Object.values(d)))
      .catch(() =>
        api
          .fetchApi("/object_info/LoraLoaderModelOnly")
          .then((r) => r.json())
          .then((d) => d.LoraLoaderModelOnly.input.required.lora_name[0])
      )
      .catch(() => []);
  }
  return loraListPromise;
}

// Manifest: loras a workflow may use that this server doesn't have. They
// show in the picker as cloud items; a picked one is stamped from the
// manifest and downloads on first run.
let manifestPromise = null;
function getManifest() {
  if (!manifestPromise) {
    manifestPromise = api
      .fetchApi("/lora_wrangler/manifest")
      .then((r) => (r.ok ? r.json() : { entries: [] }))
      .then((d) => d.entries || [])
      .catch(() => []);
  }
  return manifestPromise;
}

const VIRTUAL_NOTE = "not on this server — downloaded (or found by " +
  "hash) when the workflow runs";

// ---------------------------------------------------------------- styles

const CSS = `
/* The Vue renderer rings every advanced input. In our nodes the show/hide
   advanced bar already marks that section, and stacked toggles' rings
   overlap, so drop them here only: other nodes keep theirs. */
[data-node-id]:has(.lw-root) .ring-component-node-widget-advanced {
  --tw-ring-color: transparent; }
.lw-root { display:flex; flex-direction:column; gap:4px; width:100%;
  font-family: var(--comfy-font-family, sans-serif); font-size:12px;
  color: var(--input-text, #ccc); box-sizing:border-box; padding:2px 0; }
.lw-root, .lw-root * { box-sizing:border-box; }
.lw-row { display:flex; align-items:center; gap:5px; height:24px; }
.lw-toggle { width:34px; height:16px; border-radius:8px; flex:none;
  background: var(--comfy-input-bg, #222); border:1px solid var(--border-color, #4e4e4e);
  position:relative; cursor:pointer; transition: background .1s; }
.lw-toggle::after { content:""; position:absolute; top:1px; left:1px;
  width:12px; height:12px; border-radius:50%; background:#8a8a9a; transition: left .1s; }
.lw-row.on .lw-toggle, .lw-header.on .lw-toggle { background:#5a5aad; }
.lw-row.on .lw-toggle::after, .lw-header.on .lw-toggle::after {
  left:19px; background:#e6e6f0; }
.lw-header { display:flex; align-items:center; gap:6px; height:22px;
  padding-bottom:3px; margin-bottom:1px;
  border-bottom:1px solid var(--border-color, #3a3a3a); }
.lw-header-label { flex:1 1 auto; color:#9a9aa8; cursor:pointer;
  user-select:none; }
.lw-status { flex:0 1 auto; min-width:0; color:#9a9aff; font-style:italic;
  font-size:11px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.lw-idbtn { flex:none; height:18px; padding:0 7px; border-radius:4px;
  border:1px solid var(--border-color, #4e4e4e); background:transparent;
  color:#8a8a98; font-size:10px; font-weight:700; white-space:nowrap; }
.lw-idbtn.ok { color:#7cc47c; border-color:#3f6b3f; cursor:default; }
.lw-idbtn.todo { color:#e2b54a; border-color:#7a6128; cursor:pointer; }
.lw-idbtn.todo:hover { background:rgba(226,181,74,.14); }
.lw-idbtn.busy { cursor:progress; }
.lw-name { flex:1 1 auto; min-width:0; height:22px; line-height:20px;
  background: var(--comfy-input-bg, #222); color: var(--input-text, #ccc);
  border:1px solid var(--border-color, #4e4e4e); border-radius:6px;
  padding:0 6px; cursor:pointer; overflow:hidden; white-space:nowrap;
  text-overflow:ellipsis; }
.lw-name:hover { filter:brightness(1.25); }
.lw-row:not(.on) .lw-name, .lw-row:not(.on) .lw-num { opacity:.45; }
.lw-arrow { flex:none; width:18px; height:22px; border:none; border-radius:4px;
  background:transparent; color: var(--input-text, #ccc); cursor:pointer; padding:0; }
.lw-arrow:hover { background: var(--comfy-input-bg, #333); }
.lw-strength-label { flex:none; color:#8a8a98; font-size:10px; width:8px;
  text-align:center; user-select:none; }
.lw-num { flex:none; width:52px; height:22px; text-align:center;
  background: var(--comfy-input-bg, #222); color: var(--input-text, #ccc);
  border:1px solid var(--border-color, #4e4e4e); border-radius:6px;
  -moz-appearance:textfield; appearance:textfield; }
.lw-num::-webkit-inner-spin-button, .lw-num::-webkit-outer-spin-button {
  -webkit-appearance:none; margin:0; }
/* slider bar spans the name and arrows: left of it the toggle (34px + 5px
   gap), right of it the T and remove buttons (5 + 20 + 5 + 18px) */
.lw-sliderbar { display:flex; align-items:center; height:14px;
  padding:0 48px 0 39px; }
.lw-sliderbar:not(.on) { opacity:.45; }
.lw-range { width:100%; height:14px; margin:0; cursor:pointer;
  accent-color:#5a5aad; }
.lw-trig { flex:none; width:20px; height:22px; border:none; border-radius:4px;
  background:transparent; color:#777; cursor:pointer; padding:0;
  font-weight:700; font-size:12px; }
.lw-trig:hover { color: var(--input-text, #ccc);
  background: var(--comfy-input-bg, #333); }
.lw-trig.ann { color:#9a9aff; }
.lw-trigpop { padding:8px 10px 10px 10px; min-width:280px; max-width:480px; }
.lw-trigpop-head { display:flex; align-items:center; gap:8px;
  margin-bottom:7px; }
.lw-trigpop-title { flex:1 1 auto; min-width:0; font-weight:600;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.lw-trigpop-link { flex:none; height:22px; padding:0 8px; border-radius:6px;
  cursor:pointer; font-size:11px; background:transparent; color:#9a9aff;
  border:1px solid var(--border-color, #4e4e4e); }
.lw-trigpop-link:hover { background: var(--comfy-input-bg, #333); }
.lw-trigpop-link.off { color:#666; cursor:default; }
.lw-trigpop-link.off:hover { background:transparent; }
.lw-pills { display:flex; flex-wrap:wrap; gap:5px; max-height:40vh;
  overflow-y:auto; }
.lw-pill { padding:2px 10px; border-radius:11px; cursor:pointer;
  border:1px solid var(--border-color, #4e4e4e); user-select:none;
  color:#9a9aa8; background:transparent; font-size:12px; }
.lw-pill.on { background:#5a5aad; border-color:#5a5aad; color:#fff; }
.lw-pill:hover { filter:brightness(1.2); }
.lw-pill.custom { border-style:dashed; }
.lw-pill-x { margin-left:7px; opacity:.55; cursor:pointer; }
.lw-pill-x:hover { opacity:1; color:#e66; }
.lw-trigpop-row { display:flex; gap:6px; margin-top:8px; }
.lw-trigpop-input { flex:1 1 auto; height:24px; padding:0 8px;
  background: var(--comfy-input-bg, #222); color: var(--input-text, #eee);
  border:1px solid var(--border-color, #4e4e4e); border-radius:6px;
  outline:none; }
.lw-trigpop-btn { height:24px; padding:0 10px; border-radius:6px;
  cursor:pointer; background:#5a5aad; color:#fff; border:none; }
.lw-trigpop-note { color:#888; font-style:italic; padding:2px 0; }
.lw-range-label { flex:none; color:#9a9aa8; align-self:center; }
.lw-range-num { flex:none; width:58px; height:24px; text-align:center;
  background: var(--comfy-input-bg, #222); color: var(--input-text, #eee);
  border:1px solid var(--border-color, #4e4e4e); border-radius:6px;
  -moz-appearance:textfield; appearance:textfield; }
.lw-range-src { flex:1 1 auto; align-self:center; color:#777;
  font-style:italic; text-align:right; }
.lw-del { flex:none; width:18px; height:22px; border:none; background:transparent;
  color:#777; cursor:pointer; padding:0; border-radius:4px; }
.lw-del:hover { color:#e66; background: var(--comfy-input-bg, #333); }
.lw-add { height:24px; margin-top:2px; border-radius:6px; cursor:pointer;
  background: var(--comfy-input-bg, #222); color: var(--input-text, #ccc);
  border:1px solid var(--border-color, #4e4e4e); }
.lw-add:hover { filter:brightness(1.25); }

.lw-backdrop { position:fixed; inset:0; z-index:9999; background:transparent; }
.lw-picker { position:fixed; z-index:10000; display:flex; flex-direction:column;
  background: var(--comfy-menu-bg, #1b1b23); color: var(--input-text, #ccc);
  border:1px solid var(--border-color, #4e4e4e); border-radius:8px;
  box-shadow:0 8px 28px rgba(0,0,0,.55);
  font-family: var(--comfy-font-family, sans-serif); font-size:13px; }
.lw-picker-filter { margin:7px 7px 4px 7px; height:26px; padding:0 8px;
  background: var(--comfy-input-bg, #222); color: var(--input-text, #eee);
  border:1px solid var(--border-color, #4e4e4e); border-radius:6px; outline:none; }
.lw-picker-filter:focus { border-color:#5a5aad; }
.lw-picker-list { overflow-y:auto; max-height:45vh; padding:2px 0 6px 0; }
.lw-item { padding:3px 12px; cursor:pointer; white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis; }
.lw-item.hl { background:#5a5aad; color:#fff; }
.lw-item.cur { font-weight:600; }
.lw-item.virtual { color:#9a9aff; font-style:italic; }
.lw-item.virtual.hl { color:#fff; }
.lw-name.virtual { color:#9a9aff; font-style:italic; }
.lw-picker-note { padding:3px 12px; color:#888; font-style:italic; }
.lw-check { margin:0 7px 0 0; vertical-align:middle; cursor:pointer;
  accent-color:#5a5aad; }
.lw-picker-foot { display:flex; align-items:center; gap:8px;
  padding:6px 8px; border-top:1px solid var(--border-color, #3a3a3a); }
.lw-foot-btn { height:24px; padding:0 10px; border-radius:6px; cursor:pointer;
  background:#5a5aad; color:#fff; border:none; }
.lw-foot-btn:disabled { background: var(--comfy-input-bg, #333);
  color:#777; cursor:default; }
.lw-foot-link { color:#9a9aff; cursor:pointer; user-select:none;
  background:none; border:none; padding:0; font-size:12px; }
.lw-foot-link:hover { text-decoration:underline; }
`;

let cssInjected = false;
function injectCSS() {
  if (cssInjected) return;
  cssInjected = true;
  const el = document.createElement("style");
  el.textContent = CSS;
  document.head.appendChild(el);
}

// ---------------------------------------------------------------- lora picker

let activePicker = null;

function closePicker() {
  if (!activePicker) return;
  document.removeEventListener("pointerdown", activePicker.onDocDown, true);
  document.removeEventListener("mousedown", activePicker.onDocDown, true);
  document.removeEventListener("keydown", activePicker.onDocKey, true);
  window.removeEventListener("keydown", activePicker.onDocKey, true);
  window.removeEventListener("resize", activePicker.close);
  activePicker.backdrop?.remove();
  activePicker.panel.remove();
  activePicker = null;
}

// Shared modal shell: transparent backdrop + fixed panel + Escape, torn
// down through the same activePicker slot the lora picker uses (opening
// either closes the other).
function openModalPanel(anchor, className) {
  closePicker();
  const rect = anchor.getBoundingClientRect();
  const panel = document.createElement("div");
  panel.className = "lw-picker " + (className || "");
  const width = Math.max(rect.width, 280);
  panel.style.minWidth = width + "px";
  panel.style.left = Math.min(rect.left, window.innerWidth - width - 12) + "px";
  const spaceBelow = window.innerHeight - rect.bottom;
  if (spaceBelow > 220 || rect.top < 220) {
    panel.style.top = rect.bottom + 2 + "px";
    panel.style.maxHeight = Math.max(spaceBelow - 12, 200) + "px";
  } else {
    panel.style.bottom = window.innerHeight - rect.top + 2 + "px";
    panel.style.maxHeight = rect.top - 12 + "px";
  }
  const backdrop = document.createElement("div");
  backdrop.className = "lw-backdrop";
  backdrop.addEventListener("pointerdown", (e) => {
    e.preventDefault(); e.stopPropagation(); closePicker();
  });
  backdrop.addEventListener("contextmenu", (e) => e.preventDefault());
  document.body.appendChild(backdrop);
  document.body.appendChild(panel);
  panel.addEventListener("mousedown", (e) => e.stopPropagation());
  panel.addEventListener("pointerdown", (e) => e.stopPropagation());
  const onDocDown = (e) => { if (!panel.contains(e.target)) closePicker(); };
  document.addEventListener("pointerdown", onDocDown, true);
  document.addEventListener("mousedown", onDocDown, true);
  const onDocKey = (e) => {
    if (e.key === "Escape") { e.stopPropagation(); closePicker(); }
  };
  document.addEventListener("keydown", onDocKey, true);
  window.addEventListener("keydown", onDocKey, true);
  const close = () => closePicker();
  window.addEventListener("resize", close);
  activePicker = { panel, backdrop, onDocDown, onDocKey, close };
  return panel;
}

// Trigger management popup: pills for each known trigger; selection is
// written into the row's "# triggers: ..." annotation (absent = all,
// empty = none), which is what the backend's triggers output honors.
async function openTriggerPopup(anchor, row, sync, onAnnChanged, onRangeChanged) {
  const panel = openModalPanel(anchor, "lw-trigpop");

  const head = document.createElement("div");
  head.className = "lw-trigpop-head";
  const title = document.createElement("div");
  title.className = "lw-trigpop-title";
  title.textContent = row.name;
  title.title = row.name;
  // .off + aria-disabled rather than disabled, so the tooltip still shows
  const link = document.createElement("button");
  link.className = "lw-trigpop-link";
  link.textContent = "Civitai page \u2197";
  let pageUrl = null;
  const setPage = (url) => {
    pageUrl = url || null;
    link.classList.toggle("off", !pageUrl);
    link.setAttribute("aria-disabled", String(!pageUrl));
    link.title = pageUrl || "No Civitai link";
  };
  link.addEventListener("click", (e) => {
    e.stopPropagation();
    if (pageUrl) window.open(pageUrl, "_blank", "noopener");
  });
  setPage(civitaiPageUrl(parseCommentFields(row.comment).fields.civitai));
  head.append(title, link);
  const body = document.createElement("div");
  body.className = "lw-trigpop-note";
  body.textContent = "loading\u2026";
  panel.append(head, body);

  let known = [];
  let customs = [];
  let saved = null;         // globally saved selection from the backend
  let savedCustoms = null;  // full saved custom roster (on and off)
  let source = null;        // where the known triggers came from
  let sel = null;           // Set
  let curRange = null;      // [lo, hi] or null
  let curRangeSrc = null;
  let curCandidates = [];   // fetched range hints, safe-first

  const saveRange = async (rng) => {
    try {
      const r = await api.fetchApi("/lora_wrangler/range", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: row.name, range: rng }),
      });
      const d = await r.json();
      if (d.ok) {
        curRange = d.range || null;
        curRangeSrc = d.source || null;
        if (d.key) {
          if (curRange) prefRanges[d.key] = curRange;
          else delete prefRanges[d.key];
        }
        onRangeChanged?.();
      }
    } catch (err) {
      console.error("[LoRA Wrangler] range save failed", err);
    }
  };

  const commit = async () => {
    const selKnown = known.filter((t) => sel.has(t));
    const selCustom = customs.filter((t) => sel.has(t));
    // a toggled-off custom must keep the pref alive, or it would vanish
    const isDefaultAll = customs.length === 0 &&
      selKnown.length === known.length;
    const triggers = isDefaultAll ? null : selKnown.concat(selCustom);

    // Save globally; only claim success when the backend confirms it.
    let savedOk = false;
    try {
      const r = await api.fetchApi("/lora_wrangler/triggers", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: row.name,
          triggers,
          customs: isDefaultAll || !customs.length ? null : customs.slice(),
        }),
      });
      if (r.ok) {
        const d = await r.json();
        if (d.ok) {
          savedOk = true;
          if (d.key) {
            if (triggers === null) prefKeys.delete(d.key);
            else prefKeys.add(d.key);
          }
        } else {
          console.error("[LoRA Wrangler] trigger save rejected:", d.error);
        }
      } else {
        console.error(
          "[LoRA Wrangler] trigger save HTTP " + r.status +
          " \u2014 the running backend looks older than this UI " +
          "(v" + UI_VERSION + "). Restart ComfyUI after updating.");
      }
    } catch (err) {
      console.error("[LoRA Wrangler] saving trigger selection failed", err);
    }

    if (savedOk) {
      // global store is authoritative; migrate any per-line annotation away
      // (the other comment fields — the identity stamp — stay put)
      if (parseTriggerAnn(row.comment) !== null) {
        row.comment = withTriggerAnn(row.comment, null);
        sync();
      }
    } else {
      // graceful degradation: keep the selection in the row's annotation so
      // it still takes effect and serializes with the workflow
      const newComment = withTriggerAnn(row.comment, triggers);
      if (newComment !== (row.comment || "")) {
        row.comment = newComment;
        sync();
      }
    }
    onAnnChanged?.(savedOk
      ? triggers !== null
      : parseTriggerAnn(row.comment) !== null);
  };

  const renderPills = () => {
    body.replaceChildren();
    body.className = "";
    const pills = document.createElement("div");
    pills.className = "lw-pills";
    const mk = (t, isCustom) => {
      const pill = document.createElement("div");
      pill.className = "lw-pill" + (sel.has(t) ? " on" : "") +
        (isCustom ? " custom" : "");
      pill.textContent = t;
      pill.title = isCustom
        ? "custom \u2014 click to toggle, \u2715 to remove"
        : "click to toggle";
      pill.addEventListener("click", (e) => {
        e.stopPropagation();
        if (sel.has(t)) sel.delete(t);
        else sel.add(t);
        commit();
        renderPills();
      });
      if (isCustom) {
        const x = document.createElement("span");
        x.className = "lw-pill-x";
        x.textContent = "\u2715";
        x.title = "Remove custom trigger";
        x.addEventListener("click", (e) => {
          e.stopPropagation();
          customs = customs.filter((v) => v !== t);
          sel.delete(t);
          commit();
          renderPills();
        });
        pill.appendChild(x);
      }
      return pill;
    };
    known.forEach((t) => pills.appendChild(mk(t, false)));
    customs.forEach((t) => pills.appendChild(mk(t, true)));
    if (!known.length && !customs.length) {
      const note = document.createElement("div");
      note.className = "lw-trigpop-note";
      note.textContent = "no triggers known for this lora";
      pills.appendChild(note);
    }
    body.appendChild(pills);
    if (source === "dataset-tags") {
      const srcNote = document.createElement("div");
      srcNote.className = "lw-trigpop-note";
      srcNote.textContent =
        "inferred from dataset tag frequency \u2014 not declared triggers";
      body.appendChild(srcNote);
    }

    const rowEl = document.createElement("div");
    rowEl.className = "lw-trigpop-row";
    const input = document.createElement("input");
    input.className = "lw-trigpop-input";
    input.placeholder = "add custom trigger\u2026";
    input.addEventListener("keydown", (e) => {
      e.stopPropagation();
      if (e.key === "Enter") addCustom();
      if (e.key === "Escape") closePicker();
    });
    const addBtn = document.createElement("button");
    addBtn.className = "lw-trigpop-btn";
    addBtn.textContent = "Add";
    const addCustom = () => {
      const t = input.value.trim().replace(/,/g, " ");
      if (!t || known.includes(t) || customs.includes(t)) return;
      customs.push(t);
      sel.add(t);
      input.value = "";
      commit();
      renderPills();
    };
    addBtn.addEventListener("click", (e) => { e.stopPropagation(); addCustom(); });
    rowEl.append(input, addBtn);
    body.appendChild(rowEl);

    // ---- recommended range row (useful for sliders especially) ---------
    const rRow = document.createElement("div");
    rRow.className = "lw-trigpop-row";
    const rLabel = document.createElement("div");
    rLabel.className = "lw-range-label";
    rLabel.textContent = "Range";
    const mkRangeNum = (idx, ph) => {
      const inp = document.createElement("input");
      inp.className = "lw-range-num";
      inp.type = "number";
      inp.step = "0.5";
      inp.placeholder = ph;
      if (curRange) inp.value = curRange[idx];
      inp.addEventListener("keydown", (e) => e.stopPropagation());
      inp.addEventListener("change", async () => {
        const lo = parseFloat(rLo.value);
        const hi = parseFloat(rHi.value);
        if (isFinite(lo) && isFinite(hi)) await saveRange([lo, hi]);
        else if (rLo.value === "" && rHi.value === "") await saveRange(null);
        else return; // half-filled: wait for the other field
        renderPills();
      });
      return inp;
    };
    const rLo = mkRangeNum(0, "min");
    const rHi = mkRangeNum(1, "max");
    const rFetch = document.createElement("button");
    rFetch.className = "lw-trigpop-btn";
    rFetch.textContent = "Fetch range";
    rFetch.title = "Search this lora's Civitai description for a recommended range";
    rFetch.addEventListener("click", async (e) => {
      e.stopPropagation();
      rFetch.disabled = true;
      rFetch.textContent = "fetching\u2026";
      try {
        const r = await api.fetchApi(
          "/lora_wrangler/range?fetch=1&name=" + encodeURIComponent(row.name));
        const d = await r.json();
        curCandidates = d.candidates || [];
        if (d.range) {
          curRange = d.range;
          curRangeSrc = d.source || "civitai";
          if (d.resolved) prefRanges[d.resolved.replace(/\\/g, "/").toLowerCase()] = d.range;
          onRangeChanged?.();
          renderPills();
          return;
        } else {
          rSrc.textContent = "(nothing found in description)";
        }
      } catch (err) {
        console.error("[LoRA Wrangler] range fetch failed", err);
        rSrc.textContent = "(fetch failed)";
      }
      rFetch.disabled = false;
      rFetch.textContent = "Fetch range";
    });
    const rSrc = document.createElement("div");
    rSrc.className = "lw-range-src";
    rSrc.textContent = curRangeSrc ? `(${curRangeSrc})` : "";
    rRow.append(rLabel, rLo, rHi, rFetch, rSrc);
    body.appendChild(rRow);

    if (curCandidates.length) {
      const cRow = document.createElement("div");
      cRow.className = "lw-pills";
      cRow.style.marginTop = "4px";
      curCandidates.forEach((c, i) => {
        const chip = document.createElement("div");
        const active = curRange &&
          curRange[0] === c[0] && curRange[1] === c[1];
        chip.className = "lw-pill" + (active ? " on" : "");
        chip.textContent = `${c[0]}..${c[1]}` + (i === 0 ? " (safe)" : "");
        chip.title = "Use this range";
        chip.addEventListener("click", async (e) => {
          e.stopPropagation();
          await saveRange([c[0], c[1]]);
          renderPills();
        });
        cRow.appendChild(chip);
      });
      body.appendChild(cRow);
    }

    if (!known.length || source === "dataset-tags") {
      const fRow = document.createElement("div");
      fRow.className = "lw-trigpop-row";
      const fBtn = document.createElement("button");
      fBtn.className = "lw-trigpop-btn";
      fBtn.textContent = source === "dataset-tags"
        ? "Check Civitai for declared triggers"
        : "Fetch from Civitai";
      fBtn.addEventListener("click", async (e) => {
        e.stopPropagation();
        fBtn.disabled = true;
        fBtn.textContent = "fetching\u2026";
        try {
          const r = await api.fetchApi(
            "/lora_wrangler/triggers?fetch=1&name=" +
            encodeURIComponent(row.name));
          const d = await r.json();
          known = d.triggers || [];
          if (d.civitai_url) setPage(d.civitai_url);
          const newSource = d.source || null;
          if (newSource === "civitai") {
            // declared triggers replace the inference wholesale; stale
            // inferred tags must not survive into the selection as customs
            sel = new Set(known);
            customs = [];
          } else {
            known.forEach((t) => sel.add(t));
          }
          source = newSource;
        } catch (err) { console.error("[LoRA Wrangler] fetch", err); }
        commit();
        renderPills();
      });
      fRow.appendChild(fBtn);
      body.appendChild(fRow);
    }
  };

  try {
    const r = await api.fetchApi(
      "/lora_wrangler/triggers?fetch=0&name=" + encodeURIComponent(row.name));
    const d = await r.json();
    known = (d.triggers || []).slice();
    saved = d.selected != null ? d.selected.slice() : null;
    savedCustoms = d.customs != null ? d.customs.slice() : null;
    source = d.source || null;
    if (d.civitai_url) setPage(d.civitai_url);
    if (d.version === undefined) {
      console.warn(
        "[LoRA Wrangler] backend has no version field \u2014 it predates " +
        "this UI (v" + UI_VERSION + "). Saved trigger selections will fall " +
        "back to per-workflow annotations until ComfyUI is restarted with " +
        "the updated backend.");
    }
  } catch (err) {
    console.error("[LoRA Wrangler] triggers endpoint", err);
  }
  try {
    const rr = await api.fetchApi(
      "/lora_wrangler/range?fetch=0&name=" + encodeURIComponent(row.name));
    const rd = await rr.json();
    curRange = rd.range || null;
    curRangeSrc = rd.source || null;
    curCandidates = rd.candidates || [];
  } catch (err) {}

  const ann = parseTriggerAnn(row.comment);
  if (ann !== null) {
    sel = new Set(ann);                       // per-workflow override
    customs = ann.filter((t) => !known.includes(t));
  } else if (saved != null) {
    sel = new Set(saved);                     // the user's saved selection
    customs = savedCustoms != null
      ? savedCustoms.slice()                  // full roster incl. off ones
      : saved.filter((t) => !known.includes(t));   // legacy prefs: derive
    // any selected phrase not in known must appear as a custom pill
    saved.forEach((t) => {
      if (!known.includes(t) && !customs.includes(t)) customs.push(t);
    });
  } else {
    sel = new Set(known);                     // never customized: all
  }
  renderPills();
}

// PLL-style chooser: filter input on top, scrollable list, keyboard nav.
// opts.multi: checkbox multi-select with an Add-selected footer;
// opts.existing: Set of names already in the stack (deduped on commit);
// opts.onPickMany(names[]): commit callback for multi mode.
function openLoraPicker(anchor, items, current, onPick, opts = {}) {
  closePicker();
  const multi = !!opts.multi;
  const selected = new Set();

  const rect = anchor.getBoundingClientRect();
  const panel = document.createElement("div");
  panel.className = "lw-picker";
  const width = Math.max(rect.width, 320);
  panel.style.minWidth = width + "px";
  panel.style.maxWidth = "560px";
  panel.style.left = Math.min(rect.left, window.innerWidth - width - 12) + "px";

  const spaceBelow = window.innerHeight - rect.bottom;
  if (spaceBelow > 220 || rect.top < 220) {
    panel.style.top = rect.bottom + 2 + "px";
    panel.style.maxHeight = Math.max(spaceBelow - 12, 200) + "px";
  } else {
    panel.style.bottom = window.innerHeight - rect.top + 2 + "px";
    panel.style.maxHeight = rect.top - 12 + "px";
  }

  const filter = document.createElement("input");
  filter.className = "lw-picker-filter";
  filter.placeholder = "Filter list";
  filter.type = "text";

  const list = document.createElement("div");
  list.className = "lw-picker-list";

  panel.append(filter, list);

  const backdrop = document.createElement("div");
  backdrop.className = "lw-backdrop";
  backdrop.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    e.stopPropagation();
    closePicker();
  });
  backdrop.addEventListener("contextmenu", (e) => e.preventDefault());
  document.body.appendChild(backdrop);

  let foot = null, addBtn = null, allBtn = null;
  if (multi) {
    foot = document.createElement("div");
    foot.className = "lw-picker-foot";
    addBtn = document.createElement("button");
    addBtn.className = "lw-foot-btn";
    allBtn = document.createElement("button");
    allBtn.className = "lw-foot-link";
    foot.append(addBtn, allBtn);
    panel.appendChild(foot);
  }
  document.body.appendChild(panel);

  let filtered = items;
  let hl = 0;

  const commitMany = () => {
    const names = items.filter((n) => selected.has(n));
    closePicker();
    if (names.length) opts.onPickMany?.(names);
  };

  const pick = (name) => {
    if (multi && selected.size) { commitMany(); return; }
    closePicker();
    onPick(name);
  };

  const updateFoot = () => {
    if (!multi) return;
    addBtn.textContent = selected.size
      ? `Add selected (${selected.size})`
      : "Add selected";
    addBtn.disabled = !selected.size;
    const allSel = filtered.length &&
      filtered.every((n) => selected.has(n));
    allBtn.textContent = allSel
      ? `Deselect all (${filtered.length})`
      : `Select all (${filtered.length})`;
  };

  const renderList = () => {
    list.replaceChildren();
    const shown = filtered.slice(0, MAX_LIST_ITEMS);
    shown.forEach((name, i) => {
      const div = document.createElement("div");
      const isVirtual = !!opts.virtual?.has(name);
      div.className = "lw-item" + (i === hl ? " hl" : "") +
        (name === current ? " cur" : "") + (isVirtual ? " virtual" : "");
      const label = (isVirtual ? "☁ " : "") + name;
      if (multi) {
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.className = "lw-check";
        cb.checked = selected.has(name);
        div.appendChild(cb);
        div.appendChild(document.createTextNode(label));
      } else {
        div.textContent = label;
      }
      div.title = isVirtual ? name + " — " + VIRTUAL_NOTE : name;
      div.addEventListener("mouseenter", () => {
        hl = i;
        list.querySelectorAll(".lw-item.hl").forEach((e) => e.classList.remove("hl"));
        div.classList.add("hl");
      });
      const toggleSel = () => {
        if (selected.has(name)) selected.delete(name);
        else selected.add(name);
        const cb = div.querySelector(".lw-check");
        if (cb) cb.checked = selected.has(name);
        updateFoot();
      };
      div.addEventListener("mousedown", (e) => { e.preventDefault(); e.stopPropagation(); });
      div.addEventListener("click", (e) => {
        e.stopPropagation();
        if (!multi) { pick(name); return; }
        // fast path: plain click with no selection going = add this one now.
        // ctrl/cmd-click or clicking the checkbox = start/extend a selection;
        // once anything is checked, plain clicks toggle too.
        const onCheckbox = e.target && e.target.classList &&
          e.target.classList.contains("lw-check");
        if (e.ctrlKey || e.metaKey || onCheckbox || selected.size > 0) {
          toggleSel();
        } else {
          closePicker();
          opts.onPickMany?.([name]);
        }
      });
      div.addEventListener("dblclick", (e) => {
        e.stopPropagation();
        if (!multi) return;
        // commit everything checked plus this one (the two clicks of the
        // dblclick toggle the item off and back on, so state is unchanged)
        selected.add(name);
        commitMany();
      });
      list.appendChild(div);
    });
    if (filtered.length > MAX_LIST_ITEMS) {
      const note = document.createElement("div");
      note.className = "lw-picker-note";
      note.textContent = `\u2026 ${filtered.length - MAX_LIST_ITEMS} more, keep typing`;
      list.appendChild(note);
    }
    if (!filtered.length) {
      const note = document.createElement("div");
      note.className = "lw-picker-note";
      note.textContent = "no matches";
      list.appendChild(note);
    }
  };

  const applyFilter = () => {
    const terms = filter.value.toLowerCase().split(/\s+/).filter(Boolean);
    filtered = terms.length
      ? items.filter((n) => { const ln = n.toLowerCase(); return terms.every((t) => ln.includes(t)); })
      : items;
    hl = 0;
    renderList();
    updateFoot();
  };

  const scrollToHl = () => {
    const el = list.children[hl];
    if (el) el.scrollIntoView({ block: "nearest" });
  };

  filter.addEventListener("input", applyFilter);
  filter.addEventListener("keydown", (e) => {
    e.stopPropagation();
    if (e.key === "ArrowDown") {
      e.preventDefault();
      hl = Math.min(hl + 1, Math.min(filtered.length, MAX_LIST_ITEMS) - 1);
      renderList(); scrollToHl();
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      hl = Math.max(hl - 1, 0);
      renderList(); scrollToHl();
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (multi && selected.size) commitMany();
      else if (filtered[hl] !== undefined) {
        if (multi) { closePicker(); opts.onPickMany?.([filtered[hl]]); }
        else pick(filtered[hl]);
      }
    } else if (e.key === "Escape") {
      e.preventDefault();
      closePicker();
    }
  });
  panel.addEventListener("mousedown", (e) => e.stopPropagation());
  panel.addEventListener("pointerdown", (e) => e.stopPropagation());

  // Backup to the backdrop: the canvas preventDefaults pointerdown (which
  // suppresses synthesized mousedown) and may stop propagation early, so
  // these are best-effort — the backdrop is the mechanism that guarantees
  // outside clicks dismiss. Escape is registered on document AND window
  // capture because focus can end up on the canvas.
  const onDocDown = (e) => {
    if (!panel.contains(e.target)) closePicker();
  };
  document.addEventListener("pointerdown", onDocDown, true);
  document.addEventListener("mousedown", onDocDown, true);
  const onDocKey = (e) => {
    if (e.key === "Escape") { e.stopPropagation(); closePicker(); }
  };
  document.addEventListener("keydown", onDocKey, true);
  window.addEventListener("keydown", onDocKey, true);
  const close = () => closePicker();
  window.addEventListener("resize", close);

  if (multi) {
    addBtn.addEventListener("click", (e) => { e.stopPropagation(); commitMany(); });
    allBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      const allSel = filtered.length && filtered.every((n) => selected.has(n));
      filtered.forEach((n) => { if (allSel) selected.delete(n); else selected.add(n); });
      renderList();
      updateFoot();
    });
  }

  activePicker = { panel, backdrop, onDocDown, onDocKey, close };
  applyFilter();
  const curIdx = filtered.indexOf(current);
  if (curIdx >= 0 && curIdx < MAX_LIST_ITEMS) { hl = curIdx; renderList(); scrollToHl(); }
  setTimeout(() => filter.focus(), 0);
}

// ---------------------------------------------------------------- widget hiding

function hideWidget(w) {
  if (!w) return;
  try { w.hidden = true; } catch (e) {}
  try { if (w.options) w.options.hidden = true; } catch (e) {}
  try { w.computeSize = () => [0, -4]; } catch (e) {}
  try { if (w.element) w.element.style.display = "none"; } catch (e) {}
  try { if (w.inputEl) w.inputEl.style.display = "none"; } catch (e) {}
}

// ---------------------------------------------------------------- main

app.registerExtension({
  name: "callsign.LoraWrangler",

  settings: SETTINGS,

  async setup() {
    await syncServerSettings();
  },

  async beforeQueued() {
    await preflight();
  },

  async nodeCreated(node) {
    const cls = node.comfyClass ?? node.constructor?.comfyClass;
    if (!NODE_CLASSES.has(cls)) return;

    const lorasWidget = (node.widgets || []).find((w) => w.name === "loras");
    if (!lorasWidget) return;
    const isExclusive = () => {
      const w = (node.widgets || []).find((x) => x.name === "exclusive");
      return !!(w && (w.value === true || w.value === "true"));
    };
    const sepWidget = (node.widgets || []).find(
      (x) => x.name === "separate_clip_strength");
    const isSeparate = () => !!(sepWidget &&
      (sepWidget.value === true || sepWidget.value === "true"));

    try {
      injectCSS();

      let rows = parseText(lorasWidget.value);
      let loraList = [];
      let virtualNames = [];      // manifest loras this server lacks
      const pickerItems = () => loraList.concat(
        virtualNames.filter((n) => !loraList.includes(n)));
      const isLocal = (n) => {
        const nn = normName(n);
        return loraList.some((l) => normName(l) === nn);
      };
      // name cell text: cloud-marked when the lora isn't on this server
      const setNameCell = (el, n) => {
        const virtual = !!n && loraList.length > 0 && !isLocal(n);
        el.textContent = (virtual ? "☁ " : "") + (n || "Choose a lora");
        el.title = n ? (virtual ? n + " — " + VIRTUAL_NOTE : n) : "";
        el.classList.toggle("virtual", virtual);
      };

      const root = document.createElement("div");
      root.className = "lw-root";

      const visRows = () =>
        rows.filter((r) => r.passthrough === undefined).length;
      const sliderBars = () => cls === "SliderLoraStack" &&
        app.ui.settings.getSettingValue(SETTING_SLIDERS) !== false;
      // each row is 24px + 4px gap; a slider bar adds 14px + 4px gap
      const uiHeight = () => 28 /*header*/ +
        visRows() * (28 + (sliderBars() ? 18 : 0)) + 30 /*add*/ + 8;
      if (cls === "SliderLoraStack") ensureZeroTick();

      const domWidget = node.addDOMWidget("lora_wrangler_ui", "LW_UI", root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: uiHeight,
      });
      domWidget.computeSize = (width) => [width || 320, uiHeight()];

      hideWidget(lorasWidget);

      const sync = () => {
        lorasWidget.value = renderText(rows);
        try { lorasWidget.callback?.(lorasWidget.value); } catch (e) {}
        try { node.graph?.setDirtyCanvas?.(true, true); } catch (e) {}
      };

      // Identity stamps: ask the backend for hash + Civitai id of the named
      // rows and write them into the row comments (only fills gaps).
      const stampRows = async (names, verbose = false) => {
        names = (names || []).filter(Boolean);
        if (!names.length) return;
        try {
          const r = await api.fetchApi("/lora_wrangler/identity", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ names, network: true }),
          });
          if (!r.ok) throw new Error("HTTP " + r.status);
          const d = await r.json();
          let changed = 0, onCivitai = 0, failed = [];
          for (const n of names) {
            const res = d.results?.[n];
            if (!res || !res.fields) { failed.push(n); continue; }
            if (res.on_civitai) onCivitai++;
            const row = rows.find((x) => x.passthrough === undefined && x.name === n);
            if (!row) continue;
            const nc = withStampFields(row.comment, res.fields);
            if (nc !== (row.comment || "")) { row.comment = nc; changed++; }
          }
          if (changed) { sync(); render(); }
          if (verbose) {
            toast("Stamped " + names.length + " row(s): " + onCivitai +
              " known to Civitai" + (failed.length
                ? "; could not stamp " + failed.map(baseName).join(", ")
                : ""), failed.length ? "warn" : "success");
          }
        } catch (e) {
          console.warn("[LoRA Wrangler] stamping failed", e);
          if (verbose) toast("Stamping failed: " + e, "error");
        }
      };

      let statusText = "";
      let stamping = false;    // ID button shows "Stamping…" meanwhile
      const ctl = {
        applyRowUpdates(updates) {
          let changed = false;
          for (const u of updates || []) {
            const row = rows.find((x) => x.passthrough === undefined && x.name === u.name);
            if (!row) continue;
            const how = String(u.how || "");
            if (u.resolved && u.resolved !== row.name &&
                (how === "downloaded" || how.startsWith("relocated"))) {
              row.name = u.resolved;
              changed = true;
            }
            if (u.fields && Object.keys(u.fields).length) {
              const nc = withStampFields(row.comment, u.fields, !!u.replace);
              if (nc !== (row.comment || "")) { row.comment = nc; changed = true; }
            }
          }
          if (changed) { render(); sync(); }
        },
        setStatus(text) {
          statusText = text || "";
          const el = root.querySelector(".lw-status");
          if (el) { el.textContent = statusText; el.title = statusText; }
        },
        refresh() { render(); grow(); },
      };
      const registerCtl = () => controllers.set(String(node.id), ctl);
      registerCtl();
      const onRemoved = node.onRemoved;
      node.onRemoved = function (...args) {
        if (controllers.get(String(node.id)) === ctl)
          controllers.delete(String(node.id));
        return onRemoved?.apply(this, args);
      };

      const grow = () => {
        try {
          const sz = node.computeSize();
          node.setSize([Math.max(node.size[0], sz[0]), sz[1]]);
        } catch (e) {}
      };

      const render = () => {
        root.replaceChildren();

        // ---- header: master toggle --------------------------------------
        const header = document.createElement("div");
        const real = rows.filter((r) => r.passthrough === undefined);
        const allOn = real.length > 0 && real.every((r) => r.on);
        header.className = "lw-header" + (allOn ? " on" : "");

        const master = document.createElement("div");
        master.className = "lw-toggle";
        master.title = "Toggle all on/off";

        const label = document.createElement("div");
        label.className = "lw-header-label";
        label.textContent = "Toggle All";

        const toggleAll = () => {
          const real = rows.filter((r) => r.passthrough === undefined);
          if (isExclusive()) {
            const anyOn = real.some((r) => r.on);
            real.forEach((r) => { r.on = false; });
            if (!anyOn && real[0]) real[0].on = true; // all-off <-> solo first
          } else {
            const anyOff = real.some((r) => !r.on);
            real.forEach((r) => { r.on = anyOff; });
          }
          render();
          sync();
        };
        master.addEventListener("click", toggleAll);
        label.addEventListener("click", toggleAll);

        const status = document.createElement("span");
        status.className = "lw-status";
        status.textContent = statusText;
        status.title = statusText;

        header.append(master, label, status);

        // Identity stamps let other machines find or download each row's
        // LoRA. Green = nothing to do, amber = some rows still need one.
        const unstamped = real.filter((r) => !hasStamp(r.comment));
        if (real.length) {
          const idBtn = document.createElement("button");
          if (stamping) {
            idBtn.className = "lw-idbtn busy";
            idBtn.textContent = "Stamping\u2026";
            idBtn.disabled = true;
          } else if (!unstamped.length) {
            idBtn.className = "lw-idbtn ok";
            idBtn.textContent = "\u2713 IDs OK";
            idBtn.disabled = true;
            idBtn.title = "Every row has an identity stamp, so this " +
              "workflow can find its LoRAs on other machines, or download " +
              "them there.";
          } else {
            const n = unstamped.length;
            idBtn.className = "lw-idbtn todo";
            idBtn.textContent = `Stamp ${n} ${n === 1 ? "ID" : "IDs"}`;
            idBtn.title = `${n} row(s) have no identity stamp yet: ` +
              unstamped.map((r) => baseName(r.name)).join(", ") +
              ". Click to add them, so other machines can find or " +
              "download these LoRAs. Rows are also stamped automatically " +
              "when added and when the workflow runs.";
            idBtn.addEventListener("click", (e) => {
              e.stopPropagation();
              stamping = true;
              render();
              stampRows(unstamped.map((r) => r.name), true)
                .finally(() => { stamping = false; render(); });
            });
          }
          header.appendChild(idBtn);
        }
        root.appendChild(header);

        // ---- lora rows ---------------------------------------------------
        rows.forEach((row, idx) => {
          if (row.passthrough !== undefined) return; // comment line, kept in text
          const div = document.createElement("div");
          div.className = "lw-row" + (row.on ? " on" : "");

          const toggle = document.createElement("div");
          toggle.className = "lw-toggle";
          toggle.title = "Toggle on/off";
          toggle.addEventListener("click", () => {
            const turningOn = !row.on;
            if (turningOn && isExclusive())
              rows.forEach((r) => { if (r.passthrough === undefined) r.on = false; });
            row.on = turningOn;
            render(); // keeps the master toggle state honest
            sync();
          });

          const name = document.createElement("div");
          name.className = "lw-name";
          setNameCell(name, row.name);
          name.addEventListener("click", (e) => {
            e.stopPropagation();
            openLoraPicker(name, pickerItems(), row.name, (picked) => {
              if (picked !== row.name)
                row.comment = withoutStampFields(row.comment);
              row.name = picked;
              setNameCell(name, picked);
              sync();
              stampRows([picked]);
            }, { virtual: new Set(virtualNames) });
          });

          // one field drives both strengths in simple mode; separate mode
          // shows independent M / C fields (no arrows, labels instead)
          const rng = rangeFor(row.name);
          const rngText = rng ? ` (rec. ${rng[0]}..${rng[1]})` : "";
          let bar = null;      // this row's slider bar, if shown
          const mkNum = (get, set, title) => {
            const inp = document.createElement("input");
            inp.className = "lw-num";
            inp.type = "number";
            inp.step = "0.05";
            if (rng) { inp.min = rng[0]; inp.max = rng[1]; }
            inp.value = fmt(get());
            inp.title = title + rngText + " — click, then scroll to adjust";
            inp.addEventListener("keydown", (e) => e.stopPropagation());
            inp.addEventListener("change", () => {
              const v = parseFloat(inp.value);
              if (isFinite(v)) set(v);
              inp.value = fmt(get());
              sync();
            });
            attachWheelStep(inp, (dir) => {
              set(stepStrength(get(), dir, false, rng));
              inp.value = fmt(get());
              sync();
            });
            return inp;
          };

          const strengthEls = [];
          if (isSeparate()) {
            const lm = document.createElement("span");
            lm.className = "lw-strength-label";
            lm.textContent = "M";
            const numM = mkNum(
              () => row.strength,
              (v) => { row.strength = v; },
              "model strength");
            const lc = document.createElement("span");
            lc.className = "lw-strength-label";
            lc.textContent = "C";
            const numC = mkNum(
              () => (row.strengthClip !== undefined ? row.strengthClip : row.strength),
              (v) => { row.strengthClip = v; },
              "clip strength");
            strengthEls.push(lm, numM, lc, numC);
          } else {
            // the number field, arrows and slider bar all go through here
            const setBoth = (v) => {
              row.strength = v;
              row.strengthClip = v;
              if (bar) {
                // a typed value outside the bar's ends widens the bar
                if (v < parseFloat(bar.min)) bar.min = v;
                if (v > parseFloat(bar.max)) bar.max = v;
                bar.value = v;
              }
            };
            const both = mkNum(
              () => row.strength,
              setBoth,
              row.strengthClip !== undefined &&
              fmt(row.strengthClip) !== fmt(row.strength)
                ? `M ${fmt(row.strength)} / C ${fmt(row.strengthClip)} \u2014 editing sets both`
                : "strength (model & clip)");
            const mkArrow = (dir) => {
              const b = document.createElement("button");
              b.className = "lw-arrow";
              b.textContent = dir < 0 ? "\u25C0" : "\u25B6";
              b.title = "step 0.05 (shift: 0.25)";
              b.addEventListener("click", (e) => {
                setBoth(stepStrength(row.strength, dir, e.shiftKey, rng));
                both.value = fmt(row.strength);
                both.title = "strength (model & clip)" + rngText;
                sync();
              });
              return b;
            };
            strengthEls.push(mkArrow(-1), both, mkArrow(1));

            if (sliderBars()) {
              const [lo, hi] = rng || DEFAULT_SLIDER_RANGE;
              bar = document.createElement("input");
              bar.type = "range";
              bar.className = "lw-range";
              bar.step = "0.05";
              bar.min = Math.min(lo, row.strength);
              bar.max = Math.max(hi, row.strength);
              bar.value = row.strength;
              bar.setAttribute("list", ZERO_TICK_ID);
              bar.title = (rng
                ? `recommended range ${lo}..${hi}`
                : `no range known, showing ${lo}..${hi}; set one with T`) +
                " \u2014 drag, or click then scroll";
              bar.addEventListener("keydown", (e) => e.stopPropagation());
              bar.addEventListener("input", () => {
                const v = Math.round(parseFloat(bar.value) * 100) / 100;
                row.strength = v;
                row.strengthClip = v;
                both.value = fmt(v);
              });
              bar.addEventListener("change", () => sync());
              attachWheelStep(bar, (dir) => {
                setBoth(stepStrength(row.strength, dir, false,
                  [parseFloat(bar.min), parseFloat(bar.max)]));
                both.value = fmt(row.strength);
                sync();
              });
            }
          }

          const trigBtn = document.createElement("button");
          trigBtn.className = "lw-trig" +
            (parseTriggerAnn(row.comment) !== null || hasPref(row.name)
              ? " ann" : "");
          trigBtn.textContent = "T";
          trigBtn.title = "Manage trigger words";
          trigBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            openTriggerPopup(trigBtn, row, sync, (hasAnn) => {
              trigBtn.classList.toggle("ann", hasAnn);
            }, () => { render(); grow(); });
          });

          const del = document.createElement("button");
          del.className = "lw-del";
          del.textContent = "\u2715";
          del.title = "Remove";
          del.addEventListener("click", () => {
            rows.splice(idx, 1);
            render();
            sync();
            grow();
          });

          div.append(toggle, name, ...strengthEls, trigBtn, del);
          root.appendChild(div);
          if (bar) {
            const wrap = document.createElement("div");
            wrap.className = "lw-sliderbar" + (row.on ? " on" : "");
            wrap.appendChild(bar);
            root.appendChild(wrap);
          }
        });

        // ---- add button --------------------------------------------------
        const add = document.createElement("button");
        add.className = "lw-add";
        add.textContent = "\uFF0B Add Lora";
        add.addEventListener("click", (e) => {
          e.stopPropagation();
          const isSlider = cls === "SliderLoraStack";
          openLoraPicker(add, pickerItems(), null, () => {}, {
            multi: true,
            virtual: new Set(virtualNames),
            onPickMany: (names) => {
              const existing = new Set(
                rows.filter((r) => r.passthrough === undefined).map((r) => r.name));
              const fresh = names.filter((n) => !existing.has(n));
              if (!fresh.length) return;
              const single = fresh.length === 1;
              if (isExclusive() && single) {
                rows.forEach((r) => { if (r.passthrough === undefined) r.on = false; });
              }
              for (const n of fresh) {
                // slider banks roster at 0 (inactive = zero cost);
                // normal loras arrive at 1.0. Batch-adds in exclusive
                // mode arrive off so the invariant holds.
                const v = isSlider ? (single ? 1.0 : 0.0) : 1.0;
                rows.push({
                  name: n, strength: v, strengthClip: v,
                  on: isExclusive() ? single : true,
                });
              }
              render();
              sync();
              grow();
              stampRows(fresh);
            },
          });
        });
        root.appendChild(add);
      };

      if (sepWidget) {
        const prevCb = sepWidget.callback;
        sepWidget.callback = function (...args) {
          const r = prevCb?.apply(this, args);
          try { render(); grow(); } catch (e) {}
          return r;
        };
      }

      getLoraList().then((list) => {
        loraList = list || [];
        render();
        grow();
      });
      getManifest().then((entries) => {
        virtualNames = (entries || []).filter((e) => !e.local).map((e) => e.name);
        if (virtualNames.length) { try { render(); } catch (e) {} }
      });
      loadPrefKeys().then(() => { try { render(); } catch (e) {} });
      render();

      const onConfigure = node.onConfigure;
      node.onConfigure = function (...args) {
        onConfigure?.apply(this, args);
        try {
          rows = parseText(lorasWidget.value);
          registerCtl();
          render();
          grow();
        } catch (e) {}
      };
    } catch (err) {
      console.error("[LoRA Wrangler] row UI failed, falling back to text widget:", err);
      try {
        lorasWidget.hidden = false;
        if (lorasWidget.options) lorasWidget.options.hidden = false;
        delete lorasWidget.computeSize;
        if (lorasWidget.element) lorasWidget.element.style.display = "";
        if (lorasWidget.inputEl) lorasWidget.inputEl.style.display = "";
      } catch (e) {}
    }
  },
});
