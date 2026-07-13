// AI Sport Rep-Count Labeler - frontend logic.

const STAGE_COLORS = {
  up: '#2da44e',
  down: '#cf222e',
  transition: '#d29922',
  background: '#6e7681',
  '': '#444c56',
};

const STAGES = ['up', 'down', 'transition', 'background'];

const state = {
  clips: [],
  currentVideoId: null,
  frames: [],
  manifest: {},
  deviceCount: null,
  countsEntryRaw: null,
  trueCount: null,
  sport: 'push_up',
  currentIdx: 0,
  playing: false,
  speed: 1.0,
  showPose: true,
  labelResult: null,        // last /label response (algorithm output)
  editedStages: null,       // List[str]|null - user's working copy (null = no labels yet)
  manualEdited: false,     // does samples.json have manual_edited=true?
  timelineMethod: 'edit',  // 'edit' | 'primary' | 'pv' | 'thr'
  editMode: false,
  selection: null,         // {start, end} during drag
  unsavedChanges: false,
  playTimer: null,
};

// --- DOM refs ---------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const clipsUl = $('clips');
const emptyState = $('empty-state');
const clipDetail = $('clip-detail');
const frameImg = $('frame-img');
const btnPrev = $('btn-prev');
const btnPlay = $('btn-play');
const btnNext = $('btn-next');
const seek = $('seek');
const speedSel = $('speed');
const showPose = $('show-pose');
const frameCounter = $('frame-counter');
const trueCountInput = $('true-count');
const sportRadios = document.querySelectorAll('input[name="sport"]');
const btnLabel = $('btn-label');
const btnSave = $('btn-save');
const btnSaveLabels = $('btn-save-labels');
const btnRerun = $('btn-rerun');
const editedIndicator = $('edited-indicator');
const timelineCanvas = $('timeline-canvas');
const timelineMethodSel = $('timeline-method');
const btnEditMode = $('btn-edit-mode');
const stagePicker = $('stage-picker');
const editHint = $('edit-hint');
const angleCanvas = $('angle-canvas');
const countPvEl = $('count-pv');
const countThrEl = $('count-thr');
const countManualEl = $('count-manual');
const countTrueEl = $('count-true');
const countStatusEl = $('count-status');
const messageEl = $('message');

// --- helpers ----------------------------------------------------------------
function showMessage(text, kind = 'info') {
  messageEl.textContent = text || '';
  messageEl.className = `message ${kind}`;
  if (text) setTimeout(() => { if (messageEl.textContent === text) messageEl.className = 'message'; }, 3500);
}

function currentStageArray() {
  // What the timeline draws right now.
  if (state.timelineMethod === 'edit') return state.editedStages;
  if (!state.labelResult) return null;
  if (state.timelineMethod === 'pv') return state.labelResult.stage_labels_pv;
  if (state.timelineMethod === 'thr') return state.labelResult.stage_labels_thr;
  return state.labelResult.stage_labels_primary;
}

function countFromStages(stages) {
  if (!stages || !stages.length) return 0;
  let count = 0, inDown = false;
  for (const s of stages) {
    if (s === 'down') {
      if (!inDown) { count++; inDown = true; }
    } else {
      inDown = false;
    }
  }
  return count;
}

function currentFrameUrl() {
  const f = state.frames[state.currentIdx];
  if (!f) return '';
  return state.showPose ? f.pose_url : f.frame_url;
}

function setPlaying(v) {
  state.playing = v;
  btnPlay.textContent = v ? '⏸' : '▶';
  if (!v && state.playTimer) {
    clearTimeout(state.playTimer);
    state.playTimer = null;
  }
  if (v) scheduleNext();
}

function scheduleNext() {
  if (!state.playing) return;
  if (state.currentIdx >= state.frames.length - 1) {
    setPlaying(false);
    return;
  }
  const dt = (state.frames[state.currentIdx + 1].time_ms - state.frames[state.currentIdx].time_ms) / state.speed;
  state.playTimer = setTimeout(() => {
    state.currentIdx += 1;
    renderFrame();
    scheduleNext();
  }, Math.max(16, dt));
}

function renderFrame() {
  const f = state.frames[state.currentIdx];
  if (!f) return;
  frameImg.src = currentFrameUrl();
  seek.value = String(state.currentIdx);
  frameCounter.textContent = `${state.currentIdx + 1} / ${state.frames.length}  (${f.time_ms}ms)`;
  drawTimeline();
  drawAngle();
}

// --- clip list --------------------------------------------------------------
async function loadClips() {
  const resp = await fetch('/api/clips');
  const data = await resp.json();
  state.clips = data.clips;
  renderClipList();
}

function renderClipList() {
  clipsUl.innerHTML = '';
  for (const c of state.clips) {
    const li = document.createElement('li');
    li.dataset.videoId = c.video_id;
    if (c.video_id === state.currentVideoId) li.classList.add('selected');
    const title = document.createElement('div');
    title.className = 'clip-title';
    title.textContent = c.video_id;
    const meta = document.createElement('div');
    meta.className = 'clip-meta';
    const trueTxt = c.true_count === null || c.true_count === undefined ? '?' : c.true_count;
    const editedMark = c.manual_edited ? ' · ✎' : '';
    meta.textContent = `${c.frame_count}f · dev=${c.device_count ?? '?'} · true=${trueTxt}${editedMark}${c.labeled ? ' · ✓' : ''}`;
    li.appendChild(title);
    li.appendChild(meta);
    li.addEventListener('click', () => selectClip(c.video_id));
    clipsUl.appendChild(li);
  }
}

// --- clip selection ---------------------------------------------------------
async function selectClip(videoId) {
  if (state.unsavedChanges && !confirm('You have unsaved label edits. Discard them and switch clips?')) {
    return;
  }
  state.currentVideoId = videoId;
  setPlaying(false);
  renderClipList();

  const resp = await fetch(`/api/clips/${encodeURIComponent(videoId)}`);
  if (!resp.ok) {
    showMessage('Failed to load clip', 'error');
    return;
  }
  const data = await resp.json();
  state.frames = data.frames;
  state.manifest = data.manifest || {};
  state.deviceCount = data.device_count;
  state.countsEntryRaw = data.counts_entry_raw;
  state.trueCount = data.true_count;
  state.sport = data.sport || state.manifest.inferred_sport_type || 'push_up';
  state.currentIdx = 0;
  state.labelResult = null;
  state.manualEdited = !!data.manual_edited;
  state.editedStages = state.manualEdited ? [...data.saved_stage_labels] : null;
  state.timelineMethod = state.manualEdited ? 'edit' : 'edit';
  state.editMode = false;
  state.selection = null;
  state.unsavedChanges = false;

  emptyState.hidden = true;
  clipDetail.hidden = false;

  seek.max = String(Math.max(0, state.frames.length - 1));
  seek.value = '0';

  document.querySelector(`input[name="sport"][value="${state.sport}"]`).checked = true;
  updateCountInputEnabled();
  if (state.sport === 'background') {
    trueCountInput.value = '0';
  } else if (state.trueCount !== null && state.trueCount !== undefined) {
    trueCountInput.value = String(state.trueCount);
  } else if (state.deviceCount !== null) {
    trueCountInput.value = String(state.deviceCount);
  } else {
    trueCountInput.value = '0';
  }

  editedIndicator.hidden = !state.manualEdited;
  updateEditModeButton();
  updateSaveLabelsButton();
  resetCountsPanel();
  drawTimeline();
  drawAngle();
  renderFrame();
}

function updateCountInputEnabled() {
  trueCountInput.disabled = (state.sport === 'background');
}

function resetCountsPanel() {
  countPvEl.textContent = '?';
  countThrEl.textContent = '?';
  countManualEl.textContent = state.editedStages ? String(countFromStages(state.editedStages)) : '?';
  countTrueEl.textContent = state.sport === 'background' ? 'n/a' : (trueCountInput.value || '?');
  countStatusEl.textContent = 'no label yet';
  countStatusEl.className = 'count-val muted';
}

// --- label / save -----------------------------------------------------------
async function runLabel(force = false) {
  if (!state.currentVideoId) return;
  btnLabel.disabled = true;
  try {
    const body = {
      sport: state.sport,
      true_count: state.sport === 'background' ? null : Number(trueCountInput.value),
      force: force,
    };
    const resp = await fetch(`/api/clips/${encodeURIComponent(state.currentVideoId)}/label`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${resp.status}`);
    }
    state.labelResult = await resp.json();
    // Algorithm labels become the starting point for editing.
    state.editedStages = [...state.labelResult.stage_labels_primary];
    state.timelineMethod = 'edit';
    timelineMethodSel.value = 'edit';
    state.unsavedChanges = false;
    drawTimeline();
    drawAngle();
    updateCountsPanel();
    updateSaveLabelsButton();
  } catch (e) {
    showMessage(`Label failed: ${e.message}`, 'error');
  } finally {
    btnLabel.disabled = false;
  }
}

async function runSaveLabels() {
  if (!state.currentVideoId || !state.editedStages) return;
  btnSaveLabels.disabled = true;
  try {
    const resp = await fetch(`/api/clips/${encodeURIComponent(state.currentVideoId)}/save-labels`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stage_labels: state.editedStages, sport: state.sport }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    state.manualEdited = true;
    state.unsavedChanges = false;
    editedIndicator.hidden = false;
    updateSaveLabelsButton();
    showMessage(`Saved ${data.frame_count} frame labels (count=${data.saved_count})`, 'ok');
  } catch (e) {
    showMessage(`Save labels failed: ${e.message}`, 'error');
  } finally {
    btnSaveLabels.disabled = false;
  }
}

async function runSaveCount() {
  if (!state.currentVideoId) return;
  btnSave.disabled = true;
  try {
    const body = {
      sport: state.sport,
      true_count: state.sport === 'background' ? null : Number(trueCountInput.value),
    };
    const resp = await fetch(`/api/clips/${encodeURIComponent(state.currentVideoId)}/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    showMessage(`Saved count to counts.json (entry #${data.total_entries})`, 'ok');
    await loadClips();
  } catch (e) {
    showMessage(`Save failed: ${e.message}`, 'error');
  } finally {
    btnSave.disabled = false;
  }
}

function updateCountsPanel() {
  const r = state.labelResult;
  if (!r) {
    resetCountsPanel();
    return;
  }
  countPvEl.textContent = String(r.count_pv);
  countThrEl.textContent = String(r.count_thr);
  const manualCount = state.editedStages ? countFromStages(state.editedStages) : 0;
  countManualEl.textContent = String(manualCount);
  countTrueEl.textContent = r.sport === 'background' ? 'n/a' : (r.true_count === null || r.true_count === undefined ? '?' : String(r.true_count));

  if (r.sport === 'background') {
    countStatusEl.textContent = 'OK (background)';
    countStatusEl.className = 'count-val ok';
  } else if (r.true_count === null || r.true_count === undefined) {
    countStatusEl.textContent = 'NEEDS CHECK (no true)';
    countStatusEl.className = 'count-val warn';
  } else if (manualCount === r.true_count) {
    countStatusEl.textContent = 'OK (your labels match true)';
    countStatusEl.className = 'count-val ok';
  } else {
    countStatusEl.textContent = `NEEDS CHECK (your ${manualCount} vs true ${r.true_count})`;
    countStatusEl.className = 'count-val warn';
  }
}

function updateSaveLabelsButton() {
  btnSaveLabels.disabled = !state.editedStages;
}

function updateEditModeButton() {
  btnEditMode.textContent = `Edit: ${state.editMode ? 'on' : 'off'}`;
  btnEditMode.classList.toggle('on', state.editMode);
  timelineCanvas.classList.toggle('editing', state.editMode && state.timelineMethod === 'edit');
  editHint.hidden = !state.editMode;
}

// --- canvas drawing ---------------------------------------------------------
function drawTimeline() {
  const ctx = timelineCanvas.getContext('2d');
  const W = timelineCanvas.width;
  const H = timelineCanvas.height;
  ctx.clearRect(0, 0, W, H);

  const stages = currentStageArray();
  const N = state.frames.length;
  if (!N) return;

  const colW = W / N;
  if (stages) {
    for (let i = 0; i < N; i++) {
      ctx.fillStyle = STAGE_COLORS[stages[i]] || STAGE_COLORS[''];
      ctx.fillRect(i * colW, 0, Math.ceil(colW) + 1, H);
    }
  } else {
    ctx.fillStyle = '#21262d';
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = '#8b949e';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('click "Auto Label" to detect stages', W / 2, H / 2 + 4);
  }

  // Selection overlay (during drag).
  if (state.selection) {
    const s = Math.min(state.selection.start, state.selection.end);
    const e = Math.max(state.selection.start, state.selection.end);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.25)';
    ctx.fillRect(s * colW, 0, (e - s + 1) * colW, H);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1;
    ctx.strokeRect(s * colW, 0, (e - s + 1) * colW, H);
  }

  // Playhead.
  const x = state.currentIdx * colW + colW / 2;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(x, 0);
  ctx.lineTo(x, H);
  ctx.stroke();
}

function drawAngle() {
  const ctx = angleCanvas.getContext('2d');
  const W = angleCanvas.width;
  const H = angleCanvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#0d1117';
  ctx.fillRect(0, 0, W, H);

  const r = state.labelResult;
  const N = state.frames.length;
  if (!N) return;

  const padL = 40, padR = 20, padT = 16, padB = 24;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const raw = r ? r.raw_angle : state.frames.map(f => f.pushup_angle);
  const smooth = r ? r.smooth_angle : null;

  let yMin = Math.min(...raw);
  let yMax = Math.max(...raw);
  if (r) {
    yMin = Math.min(yMin, r.down_thr);
    yMax = Math.max(yMax, r.up_thr);
  }
  yMin -= 10;
  yMax += 10;
  if (yMax - yMin < 30) {
    const mid = (yMax + yMin) / 2;
    yMin = mid - 20; yMax = mid + 20;
  }

  const xAt = (i) => padL + (i / Math.max(1, N - 1)) * plotW;
  const yAt = (v) => padT + (1 - (v - yMin) / (yMax - yMin)) * plotH;

  ctx.fillStyle = '#8b949e';
  ctx.font = '10px monospace';
  ctx.textAlign = 'right';
  for (let v = Math.ceil(yMin / 20) * 20; v <= yMax; v += 20) {
    if (v < yMin || v > yMax) continue;
    const y = yAt(v);
    ctx.fillText(String(v), padL - 4, y + 3);
    ctx.strokeStyle = '#21262d';
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
  }

  if (r) {
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = '#2da44e'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, yAt(r.up_thr)); ctx.lineTo(W - padR, yAt(r.up_thr)); ctx.stroke();
    ctx.fillStyle = '#2da44e'; ctx.textAlign = 'left';
    ctx.fillText(`up ${r.up_thr.toFixed(0)}`, padL + 4, yAt(r.up_thr) - 3);

    ctx.strokeStyle = '#cf222e';
    ctx.beginPath(); ctx.moveTo(padL, yAt(r.down_thr)); ctx.lineTo(W - padR, yAt(r.down_thr)); ctx.stroke();
    ctx.fillStyle = '#cf222e';
    ctx.fillText(`down ${r.down_thr.toFixed(0)}`, padL + 4, yAt(r.down_thr) + 11);
    ctx.setLineDash([]);
  }

  ctx.strokeStyle = '#c9d1d9';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < N; i++) {
    const x = xAt(i), y = yAt(raw[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  if (smooth) {
    ctx.strokeStyle = '#1f6feb';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const x = xAt(i), y = yAt(smooth[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    for (const [idx, kind] of r.extrema) {
      const x = xAt(idx), y = yAt(smooth[idx]);
      ctx.fillStyle = kind === 'peak' ? '#2da44e' : '#cf222e';
      ctx.beginPath();
      if (kind === 'peak') {
        ctx.moveTo(x, y - 5); ctx.lineTo(x - 4, y + 3); ctx.lineTo(x + 4, y + 3);
      } else {
        ctx.moveTo(x, y + 5); ctx.lineTo(x - 4, y - 3); ctx.lineTo(x + 4, y - 3);
      }
      ctx.closePath();
      ctx.fill();
    }
  }

  const px = xAt(state.currentIdx);
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(px, padT);
  ctx.lineTo(px, padT + plotH);
  ctx.stroke();
}

// --- timeline interaction ---------------------------------------------------
function frameAtX(clientX) {
  const rect = timelineCanvas.getBoundingClientRect();
  const x = (clientX - rect.left) / rect.width * timelineCanvas.width;
  const colW = timelineCanvas.width / state.frames.length;
  return Math.min(state.frames.length - 1, Math.max(0, Math.floor(x / colW)));
}

function showStagePicker(clientX, clientY) {
  const wrap = timelineCanvas.parentElement;
  const rect = wrap.getBoundingClientRect();
  const left = Math.min(rect.width - 250, clientX - rect.left);
  const top = clientY - rect.top + 4;
  stagePicker.style.left = `${left}px`;
  stagePicker.style.top = `${top}px`;
  stagePicker.hidden = false;
}

function hideStagePicker() {
  stagePicker.hidden = true;
}

function applyStageToSelection(stage) {
  if (!state.selection || !state.editedStages) {
    hideStagePicker();
    return;
  }
  const s = Math.min(state.selection.start, state.selection.end);
  const e = Math.max(state.selection.start, state.selection.end);
  for (let i = s; i <= e; i++) {
    state.editedStages[i] = stage;
  }
  state.selection = null;
  hideStagePicker();
  state.unsavedChanges = true;
  drawTimeline();
  updateCountsPanel();
  updateSaveLabelsButton();
}

timelineCanvas.addEventListener('mousedown', (e) => {
  if (state.frames.length === 0) return;
  if (state.editMode && state.timelineMethod === 'edit' && state.editedStages) {
    const idx = frameAtX(e.clientX);
    state.selection = { start: idx, end: idx };
    drawTimeline();
  } else {
    // Click to seek.
    setPlaying(false);
    state.currentIdx = frameAtX(e.clientX);
    renderFrame();
  }
});

timelineCanvas.addEventListener('mousemove', (e) => {
  if (!state.selection) return;
  state.selection.end = frameAtX(e.clientX);
  drawTimeline();
});

window.addEventListener('mouseup', (e) => {
  if (!state.selection) return;
  showStagePicker(e.clientX, e.clientY);
});

stagePicker.addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-stage]');
  if (!btn) return;
  const stage = btn.dataset.stage;
  if (stage === 'cancel') {
    state.selection = null;
    hideStagePicker();
    drawTimeline();
  } else {
    applyStageToSelection(stage);
  }
});

// --- event wiring -----------------------------------------------------------
btnPlay.addEventListener('click', () => {
  if (state.frames.length === 0) return;
  if (state.playing) setPlaying(false);
  else {
    if (state.currentIdx >= state.frames.length - 1) {
      state.currentIdx = 0;
      renderFrame();
    }
    setPlaying(true);
  }
});

btnPrev.addEventListener('click', () => {
  if (state.frames.length === 0) return;
  setPlaying(false);
  state.currentIdx = Math.max(0, state.currentIdx - 1);
  renderFrame();
});

btnNext.addEventListener('click', () => {
  if (state.frames.length === 0) return;
  setPlaying(false);
  state.currentIdx = Math.min(state.frames.length - 1, state.currentIdx + 1);
  renderFrame();
});

seek.addEventListener('input', () => {
  if (state.frames.length === 0) return;
  setPlaying(false);
  state.currentIdx = Number(seek.value);
  renderFrame();
});

speedSel.addEventListener('change', () => {
  state.speed = Number(speedSel.value);
});

showPose.addEventListener('change', () => {
  state.showPose = showPose.checked;
  if (state.frames.length > 0) renderFrame();
});

timelineMethodSel.addEventListener('change', () => {
  state.timelineMethod = timelineMethodSel.value;
  // Non-edit views are read-only; turn off edit mode to avoid confusion.
  if (state.timelineMethod !== 'edit') {
    state.editMode = false;
  }
  updateEditModeButton();
  drawTimeline();
});

btnEditMode.addEventListener('click', () => {
  if (state.timelineMethod !== 'edit') {
    state.timelineMethod = 'edit';
    timelineMethodSel.value = 'edit';
  }
  if (!state.editedStages) {
    showMessage('Run "Auto Label" first to get a starting point for editing.', 'warn');
    return;
  }
  state.editMode = !state.editMode;
  state.selection = null;
  hideStagePicker();
  updateEditModeButton();
  drawTimeline();
});

for (const radio of sportRadios) {
  radio.addEventListener('change', () => {
    state.sport = radio.value;
    updateCountInputEnabled();
    if (!state.labelResult) {
      countTrueEl.textContent = state.sport === 'background' ? 'n/a' : (trueCountInput.value || '?');
    }
  });
}

btnLabel.addEventListener('click', () => runLabel(false));
btnRerun.addEventListener('click', () => {
  if (!state.currentVideoId) return;
  if (state.manualEdited || state.unsavedChanges) {
    if (!confirm('Re-run autolabel will overwrite your current labels in the view (saved labels on disk are kept until you click "Save Labels"). Continue?')) {
      return;
    }
  }
  runLabel(true);
});
btnSave.addEventListener('click', runSaveCount);
btnSaveLabels.addEventListener('click', runSaveLabels);

trueCountInput.addEventListener('input', () => {
  if (state.labelResult) updateCountsPanel();
});

window.addEventListener('beforeunload', (e) => {
  if (state.unsavedChanges) {
    e.preventDefault();
    e.returnValue = '';
  }
});

// --- init -------------------------------------------------------------------
loadClips().catch((e) => showMessage(`Init failed: ${e.message}`, 'error'));
