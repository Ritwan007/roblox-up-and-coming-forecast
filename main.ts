// roblox-up-and-coming-forecast-01 — entry
//
// Forecasts Roblox's Up-and-Coming top 10 as ONE probability distribution per place,
// scored against every hourly read of the 72 hours after the history ends.
//
// Model (fitted entirely from /input/history.ndjson at run time):
//   * Each game gets a latent strength on a log-rank scale, y = -ln(rank), tracked with
//     two time-decayed averages over the reads: a fast one (where the game is now) and a
//     slow one (its long-run level; reads where the game is absent pull both down).
//   * A shrunk, capped momentum term from the recent slope of y pushes games that are
//     climbing (and drags games that are falling) for the first hours of the window,
//     and a capped log-playing-growth term (recent level vs the day before) shifts
//     strength toward games whose player counts are surging — the chart sorts by
//     user growth, and in the history such games reach the top 10 far more often.
//   * The future is simulated hour by hour: strengths mean-revert (Ornstein-Uhlenbeck)
//     toward the long-run level while Gaussian noise accumulates with horizon, so the
//     chart is near-frozen in hour 1 and well shuffled by hour 72. Games whose Roblox
//     `created` date (from the reads' details, when present) is 28+ days before an hour
//     are removed from that hour's chart, as the real chart drops them.
//   * Ranking each simulated hour and counting places gives, per place, a probability
//     for every game. A small share of each place is blended with the recent empirical
//     place-occupancy distribution, tiny probabilities are dropped, and a slice of each
//     place is left unnamed on purpose: the scorer shares it over the read's chart, so
//     it is what a brand-new game (never in the history) scores if it breaks in.
//   * top10 orders the games by their expected place under the same distribution.
//
// Deterministic: a fixed-seed PRNG only; no clock, no network, no Math.random.
// Reads $INPUT_DIR (else $MAKEFUN_INPUT_DIR, else /input), writes $OUTPUT_DIR
// (else $MAKEFUN_OUTPUT_DIR, else /output). Node standard library only.

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.INPUT_DIR ?? process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.OUTPUT_DIR ?? process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Tunable constants (defaults are the values calibrated by backtesting on the history).
const P = Object.assign(
  {
    hlFast: 4, // hours: half-life of the "current level" average
    hlSlow: 160, // hours: half-life of the long-run level average
    phi: 0.982, // hourly mean-reversion persistence of strength
    sigmaInf: 2.2, // long-horizon strength noise (log-rank units)
    sigma0: 0.12, // immediate noise floor
    momWindow: 36, // hours of reads used for the momentum slope
    momShrink: 3, // pseudo-count shrinking the slope toward 0
    momCap: 0.035, // max |momentum| per hour
    momWeight: 3.0, // how much of the capped slope is applied
    momTau: 35, // hours: momentum fades with this time constant
    pgWeight: 0.6, // weight of the playing-growth term (log playing change, ~48 h)
    pgTau: 999, // hours: playing-growth term fades with this time constant
    volPow: 0, // per-game volatility scaling exponent for the noise (0 = off)
    kernBlend: 0, // weight of the empirical rank-transition kernel forecast
    renorm: 0, // rescale each place's kept rows back to the (1 - unnamed) target
    occBlend: 0.06, // weight of the empirical occupancy distribution
    occHalfLife: 30, // hours: decay for the occupancy distribution
    unnamed: 0.03, // share of each place left unnamed (covers brand-new games)
    pMin: 0.0006, // drop named probabilities below this
    sims: 1600, // Monte Carlo rankings per hour
    seed: 20261113,
    candY: -4.9, // candidates must have fast or slow level above this (about rank 130)
  },
  process.env.MAKEFUN_PARAMS ? JSON.parse(process.env.MAKEFUN_PARAMS) : {},
);

// ---------- deterministic PRNG (mulberry32 + Box-Muller) ----------
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(P.seed);
let spare = null;
function randn() {
  if (spare !== null) {
    const v = spare;
    spare = null;
    return v;
  }
  let u = 0;
  let v = 0;
  do {
    u = rand();
  } while (u <= 1e-12);
  v = rand();
  const r = Math.sqrt(-2 * Math.log(u));
  const t = 2 * Math.PI * v;
  spare = r * Math.sin(t);
  return r * Math.cos(t);
}

// ---------- input ----------
const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8'));
const places = task.places ?? 10;
const lines = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
  .split('\n')
  .filter((l) => l.trim() !== '');
const snaps = lines.map((l) => JSON.parse(l));
snaps.sort((a, b) => Date.parse(a.at) - Date.parse(b.at));
const HOUR = 3600000;
const times = snaps.map((s) => Date.parse(s.at));
const tEnd = times[times.length - 1];
const windowFrom = task.window?.from ? Date.parse(task.window.from) : tEnd;
const slots = task.window?.slots ?? 72;
const slotMinutes = task.window?.slot_minutes ?? 60;

// Per-game observations and creation dates.
const games = new Map(); // id -> {obs: [{t, rank, playing}], created: ms|null}
for (let i = 0; i < snaps.length; i++) {
  const s = snaps[i];
  const det = s.details;
  for (const row of s.up_and_coming ?? []) {
    let g = games.get(row.universe_id);
    if (!g) {
      g = { obs: [], created: null };
      games.set(row.universe_id, g);
    }
    g.obs.push({ t: times[i], rank: row.rank, playing: row.playing ?? null });
    if (g.created === null && det) {
      const d = Array.isArray(det) ? det.find((x) => x && (x.universeId === row.universe_id || x.id === row.universe_id)) : det[String(row.universe_id)] ?? det[row.universe_id];
      if (d && d.created) {
        const c = Date.parse(d.created);
        if (Number.isFinite(c)) g.created = c;
      }
    }
  }
}

// ---------- per-game strength state ----------
// y of a rank; absent reads score as a rank well past the chart's end.
function yOf(rank) {
  return -Math.log(rank);
}
const ABSENT_Y = -Math.log(140);

function ewmaSeries(g, halfLife) {
  // Time-decayed average of y over all reads from the game's first appearance on,
  // counting reads where it is absent at ABSENT_Y.
  let val = null;
  let prevT = null;
  let oi = 0;
  const firstT = g.obs[0].t;
  for (let i = 0; i < snaps.length; i++) {
    const t = times[i];
    if (t < firstT) continue;
    const present = oi < g.obs.length && g.obs[oi].t === t;
    const y = present ? yOf(g.obs[oi].rank) : ABSENT_Y;
    if (present) oi++;
    if (val === null) {
      val = y;
    } else {
      const alpha = 1 - Math.pow(0.5, (t - prevT) / HOUR / halfLife);
      val += alpha * (y - val);
    }
    prevT = t;
  }
  return val;
}

function momentum(g) {
  // Weighted least-squares slope of y vs time over the recent window (present reads only).
  const t0 = tEnd - P.momWindow * HOUR;
  let sw = 0;
  let st = 0;
  let sy = 0;
  let stt = 0;
  let sty = 0;
  let n = 0;
  for (const o of g.obs) {
    if (o.t < t0) continue;
    const w = 0.5 ** ((tEnd - o.t) / HOUR / 24);
    const x = (o.t - tEnd) / HOUR;
    const y = yOf(o.rank);
    sw += w;
    st += w * x;
    sy += w * y;
    stt += w * x * x;
    sty += w * x * y;
    n++;
  }
  if (n < 3 || sw === 0) return 0;
  const denom = sw * stt - st * st;
  if (denom <= 0) return 0;
  let slope = (sw * sty - st * sy) / denom;
  slope *= n / (n + P.momShrink);
  if (slope > P.momCap) slope = P.momCap;
  if (slope < -P.momCap) slope = -P.momCap;
  return slope * P.momWeight;
}

function playingGrowth(g) {
  // Log-playing change: recent level (last ~14 h) minus the level before it (~14-62 h ago).
  let swR = 0;
  let svR = 0;
  let swO = 0;
  let svO = 0;
  for (const o of g.obs) {
    if (o.playing === null || o.playing <= 0) continue;
    const age = (tEnd - o.t) / HOUR;
    const lp = Math.log(o.playing);
    if (age <= 14) {
      const w = 0.5 ** (age / 10);
      swR += w;
      svR += w * lp;
    } else if (age <= 62) {
      const w = 0.5 ** ((age - 14) / 24);
      swO += w;
      svO += w * lp;
    }
  }
  if (swR === 0 || swO === 0) return 0;
  let d = svR / swR - svO / swO;
  if (d > 1.5) d = 1.5;
  if (d < -1.5) d = -1.5;
  return d;
}

function volatility(g) {
  // Root-mean-square step of y between consecutive reads (present or absent), per sqrt(hour).
  let sw = 0;
  let sv = 0;
  let prev = null;
  let oi = 0;
  const firstT = g.obs[0].t;
  for (let i = 0; i < snaps.length; i++) {
    const t = times[i];
    if (t < firstT) continue;
    const present = oi < g.obs.length && g.obs[oi].t === t;
    const y = present ? yOf(g.obs[oi].rank) : ABSENT_Y;
    if (present) oi++;
    if (prev !== null) {
      const dt = (t - prev.t) / HOUR;
      if (dt > 0) {
        const w = 0.5 ** ((tEnd - t) / HOUR / 72);
        sw += w;
        sv += w * ((y - prev.y) ** 2 / dt);
      }
    }
    prev = { t, y };
  }
  if (sw === 0) return 1;
  return Math.sqrt(sv / sw);
}

const cands = [];
for (const [id, g] of games) {
  const fast = ewmaSeries(g, P.hlFast);
  const slow = ewmaSeries(g, P.hlSlow);
  if (fast <= P.candY && slow <= P.candY) continue;
  const lastObs = g.obs[g.obs.length - 1];
  cands.push({
    id,
    fast,
    slow,
    mom: momentum(g),
    pg: playingGrowth(g),
    vol: volatility(g),
    created: g.created,
    lastRank: lastObs.rank,
    lastT: lastObs.t,
  });
}
// Volatility multipliers relative to the candidate median, clipped.
{
  const vols = cands.map((c) => c.vol).sort((a, b) => a - b);
  const med = vols.length ? vols[Math.floor(vols.length / 2)] : 1;
  for (const c of cands) {
    let m = Math.pow(c.vol / (med || 1), P.volPow);
    if (m > 1.6) m = 1.6;
    if (m < 0.65) m = 0.65;
    c.volMul = m;
  }
}
// Deterministic candidate order (stable sort by id) so runs are byte-identical.
cands.sort((a, b) => a.id - b.id);

// ---------- empirical occupancy distribution (recent, decayed, place-smoothed) ----------
const occ = new Map(); // "id|place" -> weight
{
  const kern = [1, 0.45, 0.18, 0.06];
  for (let i = 0; i < snaps.length; i++) {
    const w = 0.5 ** ((tEnd - times[i]) / HOUR / P.occHalfLife);
    for (const row of snaps[i].up_and_coming) {
      if (row.rank > places + 5) continue;
      for (let k = 1; k <= places; k++) {
        const kw = kern[Math.abs(row.rank - k)] ?? 0;
        if (kw > 0) occ.set(row.universe_id + '|' + k, (occ.get(row.universe_id + '|' + k) ?? 0) + w * kw);
      }
    }
  }
}

// ---------- empirical rank-transition kernel forecast ----------
// From every pair of reads (i, j), j later by at most ~80 h: how a game at rank r in
// read i is placed in read j, as a function of the lag. Binned by starting rank and
// smoothed; weighted toward each target horizon with a lognormal lag kernel.
const RANK_BINS = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 45, 70, 1e9];
function rankBin(r) {
  for (let b = 0; b < RANK_BINS.length; b++) if (r <= RANK_BINS[b]) return b;
  return RANK_BINS.length - 1;
}
const kernPairs = []; // {r0bin, lagH, r1 (0 = not in top places / off chart)}
for (let i = 0; i < snaps.length && P.kernBlend > 0; i++) {
  const chartI = new Map();
  for (const row of snaps[i].up_and_coming) chartI.set(row.universe_id, row.rank);
  for (let j = i + 1; j < snaps.length; j++) {
    const lagH = (times[j] - times[i]) / HOUR;
    if (lagH > 80) break;
    const chartJ = new Map();
    for (const row of snaps[j].up_and_coming) chartJ.set(row.universe_id, row.rank);
    for (const [id, r0] of chartI) {
      if (r0 > 70) continue;
      const r1 = chartJ.get(id);
      kernPairs.push({ r0bin: rankBin(r0), lagH, r1: r1 !== undefined && r1 <= places ? r1 : 0 });
    }
  }
}
function kernelForecast(hourWeights) {
  // hourWeights: [{h, w}] mixture over the window's horizons.
  // Returns Map "id|place" -> probability (each place sums to 1 over candidates).
  const out = new Map();
  if (kernPairs.length === 0) return out;
  // K[bin][k] for k = 0..places (0 = not placed), averaged over the horizon mixture.
  const K = [];
  for (let b = 0; b < RANK_BINS.length; b++) {
    const row = new Float64Array(places + 1).fill(0.35); // smoothing prior
    let tot = 0.35 * (places + 1);
    for (const pr of kernPairs) {
      if (pr.r0bin !== b) continue;
      let w = 0;
      for (const hw of hourWeights) {
        const z = (Math.log(pr.lagH) - Math.log(hw.h)) / 0.55;
        w += hw.w * Math.exp(-0.5 * z * z);
      }
      if (w <= 0) continue;
      row[pr.r1] += w;
      tot += w;
    }
    for (let k = 0; k <= places; k++) row[k] /= tot;
    K.push(row);
  }
  // Apply to each candidate from its last observed rank (absent games: implied rank).
  const perPlace = new Map();
  for (const c of cands) {
    let r0;
    if (c.lastT === tEnd) r0 = c.lastRank;
    else r0 = Math.min(140, Math.exp(-c.fast));
    const brow = K[rankBin(Math.max(1, Math.round(r0)))];
    for (let k = 1; k <= places; k++) {
      const key = c.id + '|' + k;
      perPlace.set(key, (perPlace.get(key) ?? 0) + brow[k]);
    }
  }
  for (let k = 1; k <= places; k++) {
    let tot = 0;
    for (const c of cands) tot += perPlace.get(c.id + '|' + k) ?? 0;
    if (tot <= 0) continue;
    for (const c of cands) {
      const v = perPlace.get(c.id + '|' + k) ?? 0;
      if (v > 0) out.set(c.id + '|' + k, v / tot);
    }
  }
  return out;
}

// ---------- Monte Carlo over the window's hours ----------
const counts = new Map(); // "id|place" -> count
const hourList = [];
for (let i = 0; i < slots; i++) {
  const slotStart = windowFrom + i * slotMinutes * 60000;
  hourList.push((slotStart - tEnd) / HOUR + (slotMinutes / 60) * 0.5);
}
const totalDraws = P.sims * hourList.length;
const scored = new Float64Array(cands.length);
const order = cands.map((_, i) => i);
for (const h of hourList) {
  const phiH = Math.pow(P.phi, h);
  const sigH = Math.sqrt(P.sigma0 * P.sigma0 + P.sigmaInf * P.sigmaInf * (1 - phiH * phiH));
  const momH = Math.exp(-h / P.momTau) * h;
  const pgH = Math.exp(-h / P.pgTau);
  const slotTime = tEnd + h * HOUR;
  // eligible candidates this hour (28-day ageing when the creation date is known)
  const elig = [];
  for (let i = 0; i < cands.length; i++) {
    const c = cands[i];
    if (c.created !== null && slotTime > c.created + 28 * 24 * HOUR) continue;
    elig.push(i);
  }
  for (let s = 0; s < P.sims; s++) {
    for (const i of elig) {
      const c = cands[i];
      scored[i] = c.slow + (c.fast - c.slow) * phiH + c.mom * momH + P.pgWeight * c.pg * pgH + sigH * c.volMul * randn();
    }
    elig.sort((a, b) => scored[b] - scored[a]);
    const top = Math.min(places, elig.length);
    for (let k = 0; k < top; k++) {
      const key = cands[elig[k]].id + '|' + (k + 1);
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
  }
}

// ---------- assemble per-place distributions ----------
const kern = P.kernBlend > 0 ? kernelForecast(hourList.map((h) => ({ h, w: 1 / hourList.length }))) : new Map();
const probabilities = [];
const expectedPlace = new Map(); // id -> expected place score (lower is better)
for (let k = 1; k <= places; k++) {
  const col = new Map();
  for (const c of cands) {
    const mc = (counts.get(c.id + '|' + k) ?? 0) / totalDraws;
    const oc = occ.get(c.id + '|' + k) ?? 0;
    const kn = kern.get(c.id + '|' + k) ?? 0;
    col.set(c.id, { mc, oc, kn });
  }
  let occTot = 0;
  for (const v of col.values()) occTot += v.oc;
  const rows = [];
  let sum = 0;
  for (const [id, v] of col) {
    let p =
      (1 - P.occBlend - P.kernBlend) * v.mc +
      (occTot > 0 ? P.occBlend * (v.oc / occTot) : 0) +
      P.kernBlend * v.kn;
    if (p < P.pMin) p = 0;
    if (p > 0) {
      rows.push({ id, p });
      sum += p;
    }
  }
  const target = 1 - P.unnamed;
  if (sum > target) {
    const f = target / sum;
    for (const r of rows) r.p *= f;
    sum = target;
  }
  rows.sort((a, b) => b.p - a.p || a.id - b.id);
  const kept = rows.slice(0, 100);
  if (P.renorm) {
    const ks = kept.reduce((a, r) => a + r.p, 0);
    if (ks > 0 && ks < target) {
      const f = target / ks;
      for (const r of kept) r.p *= f;
    }
  }
  for (const r of kept) {
    const p = Math.floor(r.p * 1e6) / 1e6;
    if (p <= 0) continue;
    probabilities.push({ universe_id: r.id, place: k, p });
    expectedPlace.set(r.id, (expectedPlace.get(r.id) ?? 0) + p * k);
  }
}
// Games with no named probability anywhere get the worst expected place.
for (const c of cands) {
  if (!expectedPlace.has(c.id)) expectedPlace.set(c.id, 0);
}
const top10 = [...expectedPlace.entries()]
  .map(([id, ep]) => {
    const named = probabilities.filter((r) => r.universe_id === id).reduce((a, r) => a + r.p, 0);
    return { id, score: ep + (1 - named) * (places + 1.5) };
  })
  .sort((a, b) => a.score - b.score || a.id - b.id)
  .slice(0, places)
  .map((x) => x.id);

mkdirSync(OUTPUT_DIR, { recursive: true });
writeFileSync(join(OUTPUT_DIR, task.output ?? 'forecast.json'), JSON.stringify({ probabilities, top10 }) + '\n');
