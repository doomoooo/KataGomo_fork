#!/usr/bin/env python3

import argparse
from collections import deque
import datetime as dt
import json
import os
from pathlib import Path
import socket
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Deque, Dict, List, Optional

from env_defaults import load_env_sh_defaults


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KataGo 监控面板</title>
  <style>
    :root {
      --bg: #f4efe6;
      --panel: rgba(255,255,255,0.78);
      --panel-strong: rgba(255,255,255,0.92);
      --ink: #17202a;
      --muted: #5d6670;
      --line: rgba(23,32,42,0.12);
      --accent: #0f766e;
      --accent-2: #c2410c;
      --accent-3: #1d4ed8;
      --warn: #b42318;
      --shadow: 0 14px 34px rgba(25, 32, 40, 0.08);
      --radius: 16px;
      --mono: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
      --sans: "IBM Plex Sans", "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; min-width: 0; }
    html, body {
      height: 100%;
      overflow: hidden;
    }
    body {
      margin: 0;
      font-family: var(--sans);
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.16), transparent 38%),
        radial-gradient(circle at top right, rgba(194,65,12,0.12), transparent 32%),
        linear-gradient(180deg, #f8f4ec 0%, #f2ebe0 100%);
    }
    .wrap {
      width: 100vw;
      height: 100vh;
      padding: 10px;
      display: grid;
      grid-template-rows: 72px 86px minmax(0, 1fr);
      gap: 10px;
    }
    .topbar {
      display: grid;
      grid-template-columns: 1.1fr 2.4fr;
      gap: 10px;
      min-height: 0;
    }
    .panel {
      background: var(--panel);
      backdrop-filter: blur(14px);
      border: 1px solid rgba(255,255,255,0.55);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 10px 12px;
      min-height: 0;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    .brand {
      position: relative;
      justify-content: center;
    }
    .brand::after {
      content: "";
      position: absolute;
      inset: auto -34px -34px auto;
      width: 110px;
      height: 110px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(29,78,216,0.16), transparent 70%);
      pointer-events: none;
    }
    .eyebrow {
      font-size: 10px;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 4px;
    }
    .title {
      margin: 0 0 4px;
      font-size: 24px;
      line-height: 1.05;
    }
    .subtitle {
      margin: 0;
      color: var(--muted);
      max-width: 44rem;
      font-size: 12px;
      line-height: 1.35;
    }
    .status-grid {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 6px;
      height: 100%;
    }
    .status-pill {
      border-radius: 10px;
      padding: 7px 8px;
      background: var(--panel-strong);
      border: 1px solid var(--line);
      overflow: hidden;
    }
    .status-pill label, .metric-card label {
      display: block;
      font-size: 9px;
      color: var(--muted);
      margin-bottom: 4px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .status-pill strong, .metric-card strong {
      font-size: 12px;
      font-weight: 700;
      line-height: 1.25;
      display: block;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(8, minmax(0, 1fr));
      gap: 8px;
      min-height: 0;
    }
    .metric-card {
      padding: 10px 12px;
      background: var(--panel-strong);
      border-radius: 12px;
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .metric-card strong {
      display: block;
      font-size: 21px;
      line-height: 1.05;
    }
    .metric-card .sub {
      margin-top: 4px;
      color: var(--muted);
      font-size: 11px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .dashboard {
      display: grid;
      grid-template-columns: 0.94fr 1.04fr 1.16fr 1.52fr;
      grid-template-rows: minmax(0, 1.34fr) minmax(0, 0.66fr);
      grid-template-areas:
        "trend depth search infer"
        "queue batch streams streams";
      gap: 10px;
      min-height: 0;
    }
    .panel h2 {
      margin: 0;
      font-size: 14px;
      line-height: 1.1;
    }
    .panel .hint {
      margin: 2px 0 8px;
      color: var(--muted);
      font-size: 10px;
      line-height: 1.25;
      white-space: normal;
      min-height: 24px;
    }
    .tile-trend { grid-area: trend; }
    .tile-depth { grid-area: depth; }
    .tile-search { grid-area: search; }
    .tile-infer { grid-area: infer; }
    .tile-queue { grid-area: queue; }
    .tile-batch { grid-area: batch; }
    .tile-streams { grid-area: streams; }
    .chart-body {
      min-height: 0;
      flex: 1;
      overflow: hidden;
    }
    .dock-stack {
      display: grid;
      grid-template-rows: minmax(0, 1.4fr) minmax(0, 0.9fr);
      gap: 6px;
      min-height: 0;
    }
    .dock-card {
      padding: 7px 8px;
      border-radius: 10px;
      background: rgba(255,255,255,0.7);
      border: 1px solid var(--line);
      min-height: 0;
      overflow: hidden;
      display: grid;
      grid-template-rows: 14px minmax(0, 1fr);
      gap: 4px;
    }
    .dock-card h3 {
      margin: 0;
      font-size: 11px;
      line-height: 1.1;
    }
    .histogram {
      display: grid;
      gap: 4px 8px;
      align-content: start;
    }
    .plot-hist {
      height: 100%;
      min-height: 0;
      display: grid;
      grid-template-rows: 16px minmax(0, 1fr) 14px;
      gap: 3px;
    }
    .plot-meta {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: center;
      font-size: 9px;
      color: var(--muted);
      font-family: var(--mono);
      white-space: nowrap;
    }
    .plot-meta strong {
      color: var(--ink);
      font-weight: 700;
    }
    svg.plot-svg {
      width: 100%;
      height: 100%;
      display: block;
      overflow: visible;
    }
    .plot-fill {
      fill-opacity: 0.18;
    }
    .plot-stroke {
      fill: none;
      stroke-width: 2;
      stroke-linejoin: round;
      stroke-linecap: round;
    }
    .plot-grid-line {
      stroke: rgba(23,32,42,0.12);
      stroke-width: 1;
      stroke-dasharray: 2 3;
    }
    .plot-axis-line {
      stroke: rgba(23,32,42,0.26);
      stroke-width: 1;
    }
    .plot-label {
      fill: var(--muted);
      font-size: 8px;
      font-family: var(--mono);
    }
    .plot-marker {
      stroke-width: 1.2;
      stroke-dasharray: 3 2;
      opacity: 0.9;
    }
    .plot-marker-text {
      font-size: 7px;
      font-family: var(--mono);
      fill: var(--muted);
    }
    .plot-footer {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      font-size: 9px;
      color: var(--muted);
      font-family: var(--mono);
      white-space: nowrap;
    }
    .plot-footer strong {
      color: var(--ink);
      font-weight: 700;
    }
    .plot-bar-label {
      fill: var(--ink);
      font-size: 8px;
      font-family: var(--mono);
    }
    .hist-row {
      display: grid;
      grid-template-columns: 34px minmax(0, 1fr) 54px;
      gap: 6px;
      align-items: center;
      font-size: 10px;
      min-width: 0;
    }
    .hist-label {
      font-family: var(--mono);
      color: var(--muted);
      white-space: nowrap;
    }
    .bar-bg {
      height: 10px;
      border-radius: 999px;
      background: rgba(23,32,42,0.08);
      overflow: hidden;
      position: relative;
    }
    .bar-fill {
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--accent), var(--accent-3));
    }
    .bar-fill.alt {
      background: linear-gradient(90deg, #c2410c, #fb923c);
    }
    .hist-value {
      text-align: right;
      font-family: var(--mono);
      color: var(--ink);
      white-space: nowrap;
    }
    .pdf-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px;
      min-height: 0;
    }
    .pdf-card {
      padding: 7px 8px;
      border-radius: 10px;
      background: rgba(255,255,255,0.7);
      border: 1px solid var(--line);
      overflow: hidden;
      min-height: 0;
      display: grid;
      grid-template-rows: 16px minmax(0, 1fr) 14px;
      gap: 4px;
    }
    .pdf-head {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: baseline;
    }
    .pdf-title {
      font-weight: 700;
      font-size: 11px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .pdf-count {
      color: var(--muted);
      font-size: 9px;
      font-family: var(--mono);
      white-space: nowrap;
    }
    .pdf-stats {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 4px;
      font-family: var(--mono);
      font-size: 9px;
      color: var(--muted);
    }
    .pdf-stats strong {
      display: block;
      color: var(--ink);
      font-size: 10px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .spark-wrap {
      display: grid;
      grid-template-columns: 1fr;
      gap: 6px;
      min-height: 0;
    }
    .spark-card {
      padding: 8px;
      border-radius: 10px;
      background: rgba(255,255,255,0.7);
      border: 1px solid var(--line);
      min-height: 0;
      overflow: hidden;
    }
    .spark-card header {
      display: flex;
      justify-content: space-between;
      gap: 6px;
      margin-bottom: 6px;
    }
    .spark-card h3 {
      margin: 0;
      font-size: 11px;
    }
    .spark-card strong {
      font-family: var(--mono);
      font-size: 11px;
    }
    svg.spark {
      width: 100%;
      height: 34px;
      display: block;
    }
    .mono {
      font-family: var(--mono);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .waiting {
      padding: 8px;
      text-align: center;
      color: var(--muted);
      font-size: 12px;
      border: 1px dashed rgba(23,32,42,0.18);
      border-radius: 10px;
      background: rgba(255,255,255,0.52);
      min-height: 0;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .split-2 {
      display: grid;
      grid-template-columns: minmax(0, 1.18fr) minmax(0, 0.82fr);
      gap: 6px;
      min-height: 0;
      overflow: hidden;
    }
    .mini-card, .gpu-card {
      padding: 7px 8px;
      border-radius: 10px;
      background: rgba(255,255,255,0.68);
      border: 1px solid var(--line);
      min-height: 0;
      overflow: hidden;
    }
    .mini-card h3, .gpu-card h3 {
      margin: 0 0 6px;
      font-size: 11px;
      line-height: 1.1;
    }
    .gpu-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(122px, 1fr));
      gap: 6px;
      min-height: 0;
      overflow: hidden;
      align-content: start;
    }
    .gpu-grid.two-col {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .gpu-grid.compact {
      grid-template-columns: repeat(auto-fit, minmax(118px, 1fr));
    }
    .gpu-card {
      display: flex;
      flex-direction: column;
    }
    .gpu-card .histogram {
      margin-top: 2px;
    }
    @media (max-width: 1500px) {
      .wrap {
        grid-template-rows: 68px 78px minmax(0, 1fr);
        gap: 8px;
        padding: 8px;
      }
      .title { font-size: 21px; }
      .subtitle { font-size: 11px; }
      .dashboard {
        grid-template-columns: 0.92fr 1fr 1.16fr 1.44fr;
        grid-template-rows: minmax(0, 1.32fr) minmax(0, 0.68fr);
        gap: 8px;
      }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="topbar">
      <div class="panel brand">
        <div class="eyebrow">KataGo GTP Realtime Monitor</div>
        <h1 class="title">搜索资源监控</h1>
        <p class="subtitle">固定一屏展示，1Hz 刷新；深度/队列是真实直方图，耗时指标按分位数重建为稳定横轴的 PDF 轮廓。</p>
      </div>
      <div class="panel">
        <div id="status-grid" class="status-grid"></div>
      </div>
    </section>

    <section id="metrics" class="metric-grid"></section>

    <section class="dashboard">
      <div class="panel tile-trend">
        <h2>近 60 秒趋势</h2>
        <p class="hint">`visits/s` 与 `nnEval/s`</p>
        <div id="sparklines" class="chart-body spark-wrap"></div>
      </div>
      <div class="panel tile-depth">
        <h2>搜索深度与请求队列</h2>
        <p class="hint">展示实际提交到 GPU 的搜索深度，以及过去 1 秒请求队列长度时间占比</p>
        <div id="depth-and-queue" class="chart-body"></div>
      </div>
      <div class="panel tile-search">
        <h2>搜索线程循环耗时</h2>
        <p class="hint">按分位数重建的 PDF 轮廓，附带 P50 / P95 / P99 / Max 标记</p>
        <div id="search-loop" class="chart-body"></div>
      </div>
      <div class="panel tile-infer">
        <h2>推理阶段耗时</h2>
        <p class="hint">等待提交 / 预处理 / H2D / 推理 / D2H / 后处理，全部按分位数重建为 PDF 轮廓</p>
        <div id="inference-phases" class="chart-body"></div>
      </div>
      <div class="panel tile-queue">
        <h2>活跃推理线程数</h2>
        <p class="hint">过去 1 秒时间占比</p>
        <div id="queue-and-active" class="chart-body"></div>
      </div>
      <div class="panel tile-batch">
        <h2>GPU Batch 分布</h2>
        <p class="hint">全 GPU 汇总；多卡时保持单屏，不逐卡展开</p>
        <div id="gpu-batches" class="chart-body"></div>
      </div>
      <div class="panel tile-streams">
        <h2>每 GPU 的 cudaStream 活跃数</h2>
        <p class="hint">过去 1 秒不同活跃 stream 数的时间占比</p>
        <div id="gpu-streams" class="chart-body"></div>
      </div>
    </section>
  </div>

  <script>
    const fmtInt = (v) => Number(v || 0).toLocaleString('en-US');
    const fmtFloat = (v, digits = 2) => {
      if (v === null || v === undefined || Number.isNaN(Number(v))) return 'N/A';
      return Number(v).toFixed(digits);
    };
    const fmtPercent = (v) => `${fmtFloat(Number(v || 0) * 100, 1)}%`;
    const fmtSeconds = (v) => `${fmtFloat(v, 1)}s`;
    const fmtMs = (v) => `${fmtFloat(v, 3)} ms`;
    const fmtMsAxis = (v) => {
      const num = Number(v || 0);
      if (num >= 1) return num.toFixed(1);
      if (num >= 0.1) return num.toFixed(2);
      if (num >= 0.01) return num.toFixed(3);
      return num.toFixed(4);
    };
    const fmtIso = (unixMs) => {
      if (!unixMs) return '等待数据';
      return new Date(unixMs).toLocaleString('zh-CN', { hour12: false });
    };
    const histogramColumns = (count) => {
      if (count > 24) return 4;
      if (count > 12) return 3;
      if (count > 6) return 2;
      return 1;
    };
    const axisMemory = new Map();
    const histogramYAxisMemory = new Map();
    const pdfAxisMemory = new Map();
    const pdfAxisPresets = {
      'search.total_ms': { minMax: 8.0, step: 0.5, bins: 30, color: '#0f766e' },
      'search.search_ms': { minMax: 0.8, step: 0.05, bins: 26, color: '#1d4ed8' },
      'search.wait_nn_ms': { minMax: 8.0, step: 0.5, bins: 30, color: '#c2410c' },
      'infer.wait_task_submit_ms': { minMax: 0.20, step: 0.02, bins: 24, color: '#0f766e' },
      'infer.preprocess_ms': { minMax: 0.05, step: 0.005, bins: 24, color: '#1d4ed8' },
      'infer.h2d_ms': { minMax: 0.03, step: 0.002, bins: 24, color: '#0f766e' },
      'infer.infer_ms': { minMax: 3.6, step: 0.1, bins: 24, color: '#c2410c' },
      'infer.d2h_ms': { minMax: 0.015, step: 0.001, bins: 24, color: '#1d4ed8' },
      'infer.postprocess_ms': { minMax: 0.006, step: 0.0005, bins: 24, color: '#0f766e' },
      'depth': { minMax: 96, step: 8 },
      'queue': { minMax: 20, step: 4 },
    };
    const histogramYAxisPresets = {
      'depth': { minMax: 40, step: 10, formatter: fmtInt, shrinkRatio: 0.62, shrinkVotes: 8 },
      'queue': { fixedMax: 1.0, formatter: fmtPercent },
    };

    function rememberAxisMax(key, desired, step, minMax) {
      const safeDesired = Math.max(Number(desired || 0), Number(minMax || 0), 1e-9);
      const rounded = step > 0 ? Math.ceil(safeDesired / step) * step : safeDesired;
      const previous = axisMemory.get(key) || 0;
      const next = Math.max(previous, rounded);
      axisMemory.set(key, next);
      return next;
    }

    function rememberPdfAxisMax(key, desired, step, minMax) {
      const safeDesired = Math.max(Number(desired || 0), Number(minMax || 0), 1e-9);
      const snapped = step > 0 ? Math.ceil(safeDesired / step) * step : safeDesired;
      const prev = pdfAxisMemory.get(key);
      if (!prev) {
        pdfAxisMemory.set(key, { current: snapped, shrinkVotes: 0 });
        return snapped;
      }
      if (snapped > prev.current + 1e-12) {
        prev.current = snapped;
        prev.shrinkVotes = 0;
        return prev.current;
      }
      if (snapped < prev.current * 0.72) {
        prev.shrinkVotes += 1;
        if (prev.shrinkVotes >= 4) {
          prev.current = snapped;
          prev.shrinkVotes = 0;
        }
        return prev.current;
      }
      prev.shrinkVotes = 0;
      return prev.current;
    }

    function rememberHistogramYAxisMax(key, desired, preset = {}) {
      if (preset.fixedMax !== undefined && preset.fixedMax !== null) {
        return Number(preset.fixedMax);
      }
      const minMax = Number(preset.minMax || 1);
      const step = Number(preset.step || 0);
      const safeDesired = Math.max(Number(desired || 0), minMax, 1e-9);
      const snapped = step > 0 ? Math.ceil(safeDesired / step) * step : safeDesired;
      const shrinkRatio = Number(preset.shrinkRatio || 0.7);
      const shrinkVotesNeeded = Number(preset.shrinkVotes || 6);
      const prev = histogramYAxisMemory.get(key);
      if (!prev) {
        histogramYAxisMemory.set(key, { current: snapped, shrinkVotes: 0 });
        return snapped;
      }
      if (snapped > prev.current + 1e-12) {
        prev.current = snapped;
        prev.shrinkVotes = 0;
        return prev.current;
      }
      if (snapped < prev.current * shrinkRatio) {
        prev.shrinkVotes += 1;
        if (prev.shrinkVotes >= shrinkVotesNeeded) {
          prev.current = Math.max(snapped, minMax);
          prev.shrinkVotes = 0;
        }
        return prev.current;
      }
      prev.shrinkVotes = 0;
      return prev.current;
    }

    function renderStatusGrid(state) {
      const latest = state.latest;
      const status = latest?.status || {};
      const receiver = state.receiver || {};
      const errorText = status.last_send_error || receiver.last_error || '无';
      const items = [
        ['最新快照', fmtIso(latest?.timestamp_unix_ms)],
        ['Socket', status.socket_path || '未配置'],
        ['发送错误', fmtInt(status.send_error_count)],
        ['运行时长', fmtSeconds(status.session_age_s || 0)],
        ['最近错误', errorText],
        ['接收快照', fmtInt(receiver.received_count)],
      ];
      document.getElementById('status-grid').innerHTML = items.map(([label, value]) => `
        <div class="status-pill">
          <label>${label}</label>
          <strong class="${label === 'Socket' ? 'mono' : ''}">${value}</strong>
        </div>
      `).join('');
    }

    function renderMetricCards(state) {
      const latest = state.latest;
      const totals = latest?.totals || {};
      const win = latest?.window1s || {};
      const cards = [
        ['总 Visits', fmtInt(totals.visits), '累计搜索节点访问数'],
        ['总 nnEval', fmtInt(totals.nn_eval), '累计神经网络样本数'],
        ['搜索线程', fmtInt(totals.search_threads), '当前配置线程数'],
        ['累计搜索时长', fmtSeconds(totals.search_wall_time_s || 0), '总 search 墙钟时间'],
        ['Visits / s', fmtFloat(win.visits_per_s || 0, 1), '过去 1 秒'],
        ['nnEval / s', fmtFloat(win.nn_eval_per_s || 0, 1), '过去 1 秒'],
        ['nnBatch / s', fmtFloat(win.nn_batches_per_s || 0, 1), '过去 1 秒'],
        ['平均 Batch', fmtFloat(win.avg_batch_size || 0, 2), `累计 ${fmtFloat(totals.avg_batch_size || 0, 2)}`],
      ];
      document.getElementById('metrics').innerHTML = cards.map(([label, value, sub]) => `
        <article class="metric-card">
          <label>${label}</label>
          <strong>${value}</strong>
          <div class="sub">${sub}</div>
        </article>
      `).join('');
    }

    function renderHistogram(targetId, buckets, valueFormatter, alt = false, emptyText = '暂无数据') {
      const el = document.getElementById(targetId);
      el.innerHTML = histogramHtml(buckets, valueFormatter, alt, emptyText);
    }

    function denseBuckets(buckets) {
      if (!buckets || buckets.length === 0) {
        return [];
      }
      const sorted = [...buckets]
        .map((item) => ({ bucket: Number(item.bucket || 0), value: Number(item.value || 0) }))
        .sort((a, b) => a.bucket - b.bucket);
      const maxBucket = Math.max(...sorted.map((item) => item.bucket), 0);
      const valueMap = new Map(sorted.map((item) => [item.bucket, item.value]));
      const dense = [];
      for (let bucket = 0; bucket <= maxBucket; bucket += 1) {
        dense.push({ bucket, value: Number(valueMap.get(bucket) || 0) });
      }
      return dense;
    }

    function plotHistogramHtml(key, buckets, formatter, emptyText = '暂无数据', color = '#0f766e', minAxisMax = 16, axisStep = 4) {
      if (!buckets || buckets.length === 0) {
        return `<div class="waiting">${emptyText}</div>`;
      }
      const yPreset = histogramYAxisPresets[key] || { minMax: 1, step: 1, formatter };
      const denseBase = denseBuckets(buckets);
      const maxBucket = denseBase.length > 0 ? denseBase[denseBase.length - 1].bucket : 0;
      const axisMaxBucket = rememberAxisMax(key, maxBucket, axisStep, minAxisMax);
      const valueMap = new Map(denseBase.map((item) => [item.bucket, item.value]));
      const dense = [];
      for (let bucket = 0; bucket <= axisMaxBucket; bucket += 1) {
        dense.push({ bucket, value: Number(valueMap.get(bucket) || 0) });
      }
      if (dense.length === 0) {
        return `<div class="waiting">${emptyText}</div>`;
      }
      const width = 360;
      const height = 112;
      const padLeft = 18;
      const padRight = 8;
      const padTop = 8;
      const padBottom = 22;
      const chartWidth = width - padLeft - padRight;
      const chartHeight = height - padTop - padBottom;
      const observedMaxValue = Math.max(...dense.map((item) => Number(item.value || 0)), 1e-9);
      const yAxisMax = rememberHistogramYAxisMax(key, observedMaxValue, yPreset);
      const step = chartWidth / Math.max(dense.length, 1);
      const barWidth = Math.max(step - 1, 1);
      const tickCount = 6;
      const labelBuckets = Array.from({ length: tickCount + 1 }, (_, idx) => {
        if (idx === tickCount) return axisMaxBucket;
        return Math.round(idx * axisMaxBucket / tickCount);
      });

      const gridLines = [0.25, 0.5, 0.75, 1.0].map((ratio) => {
        const y = padTop + chartHeight - chartHeight * ratio;
        return `<line class="plot-grid-line" x1="${padLeft}" y1="${y.toFixed(1)}" x2="${(padLeft + chartWidth).toFixed(1)}" y2="${y.toFixed(1)}"></line>`;
      }).join('');

      const bars = dense.map((item, idx) => {
        const value = Number(item.value || 0);
        const barHeight = yAxisMax <= 0 ? 0 : (value / yAxisMax) * chartHeight;
        const x = padLeft + idx * step;
        const y = padTop + chartHeight - barHeight;
        return `
          <rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barWidth.toFixed(1)}" height="${Math.max(barHeight, 1).toFixed(1)}" rx="1.5" fill="${color}" fill-opacity="0.88">
            <title>${item.bucket}: ${formatter(value)}</title>
          </rect>
        `;
      }).join('');

      const labels = labelBuckets.map((bucket) => {
        const idx = Math.min(bucket, dense.length - 1);
        const x = padLeft + idx * step + barWidth / 2;
        return `<text class="plot-label" x="${x.toFixed(1)}" y="${(height - 6).toFixed(1)}" text-anchor="middle">${bucket}</text>`;
      }).join('');

      const maxText = formatter(observedMaxValue);
      const axisMaxText = (yPreset.formatter || formatter)(yAxisMax);
      const axisY = padTop + chartHeight;
      return `
        <div class="plot-hist">
          <div class="plot-meta">
            <span>bins <strong>${dense.length}</strong></span>
            <span>y 上限 <strong>${axisMaxText}</strong></span>
          </div>
          <svg class="plot-svg" viewBox="0 0 ${width} ${height}">
            ${gridLines}
            <line class="plot-axis-line" x1="${padLeft}" y1="${axisY.toFixed(1)}" x2="${(padLeft + chartWidth).toFixed(1)}" y2="${axisY.toFixed(1)}"></line>
            ${bars}
            ${labels}
          </svg>
          <div class="plot-footer">
            <span>x 轴固定到 <strong>${axisMaxBucket}</strong></span>
            <span>当前峰值 <strong>${maxText}</strong></span>
          </div>
        </div>
      `;
    }

    function buildQuantileMassSegments(stat) {
      if (!stat?.has_data) return [];
      const points = [{ q: 0.0, x: 0.0 }];
      for (const item of (stat.deciles || [])) {
        points.push({ q: Number(item.p || 0) / 100.0, x: Number(item.value || 0) });
      }
      if (stat.p95 !== null && stat.p95 !== undefined) points.push({ q: 0.95, x: Number(stat.p95 || 0) });
      if (stat.p99 !== null && stat.p99 !== undefined) points.push({ q: 0.99, x: Number(stat.p99 || 0) });
      if (stat.max !== null && stat.max !== undefined) points.push({ q: 1.0, x: Number(stat.max || 0) });
      points.sort((a, b) => a.q - b.q || a.x - b.x);
      const deduped = [];
      for (const point of points) {
        if (deduped.length > 0 && Math.abs(deduped[deduped.length - 1].q - point.q) < 1e-9) {
          deduped[deduped.length - 1].x = Math.max(deduped[deduped.length - 1].x, point.x);
          continue;
        }
        deduped.push({ q: point.q, x: point.x });
      }
      for (let i = 1; i < deduped.length; i += 1) {
        deduped[i].x = Math.max(deduped[i - 1].x, deduped[i].x);
      }
      const segments = [];
      for (let i = 1; i < deduped.length; i += 1) {
        segments.push({
          x0: deduped[i - 1].x,
          x1: deduped[i].x,
          mass: Math.max(0, deduped[i].q - deduped[i - 1].q),
        });
      }
      return segments;
    }

    function quantilePdfBins(stat, axisMax, binCount) {
      const bins = Array.from({ length: binCount }, () => 0);
      const segments = buildQuantileMassSegments(stat);
      if (segments.length === 0) return bins;
      const safeAxisMax = Math.max(axisMax, 1e-9);
      const binWidth = safeAxisMax / binCount;
      for (const segment of segments) {
        const segStart = Math.max(0, Math.min(segment.x0, safeAxisMax));
        const segEnd = Math.max(0, Math.min(segment.x1, safeAxisMax));
        if (segment.mass <= 0) continue;
        if (Math.abs(segEnd - segStart) < 1e-9) {
          const bucket = Math.min(binCount - 1, Math.floor(segStart / binWidth));
          bins[bucket] += segment.mass;
          continue;
        }
        const startIdx = Math.max(0, Math.floor(segStart / binWidth));
        const endIdx = Math.min(binCount - 1, Math.floor(Math.max(segStart, segEnd - 1e-12) / binWidth));
        for (let idx = startIdx; idx <= endIdx; idx += 1) {
          const x0 = idx * binWidth;
          const x1 = x0 + binWidth;
          const overlap = Math.max(0, Math.min(segEnd, x1) - Math.max(segStart, x0));
          if (overlap > 0) {
            bins[idx] += segment.mass * overlap / (segEnd - segStart);
          }
        }
      }
      return bins;
    }

    function buildPlotFromBins(bins, axisMax, formatter, color, markers = []) {
      const width = 330;
      const height = 88;
      const padLeft = 18;
      const padRight = 8;
      const padTop = 6;
      const padBottom = 18;
      const chartWidth = width - padLeft - padRight;
      const chartHeight = height - padTop - padBottom;
      const maxValue = Math.max(...bins, 1e-9);
      const step = chartWidth / Math.max(bins.length, 1);
      const axisY = padTop + chartHeight;
      const gridLines = [0.25, 0.5, 0.75, 1.0].map((ratio) => {
        const y = padTop + chartHeight - chartHeight * ratio;
        return `<line class="plot-grid-line" x1="${padLeft}" y1="${y.toFixed(1)}" x2="${(padLeft + chartWidth).toFixed(1)}" y2="${y.toFixed(1)}"></line>`;
      }).join('');
      const areaPoints = [`${padLeft},${axisY}`];
      for (let idx = 0; idx < bins.length; idx += 1) {
        const value = Number(bins[idx] || 0);
        const y = padTop + chartHeight - (value / maxValue) * chartHeight;
        const x0 = padLeft + idx * step;
        const x1 = padLeft + (idx + 1) * step;
        areaPoints.push(`${x0.toFixed(1)},${y.toFixed(1)}`);
        areaPoints.push(`${x1.toFixed(1)},${y.toFixed(1)}`);
      }
      areaPoints.push(`${padLeft + chartWidth},${axisY}`);
      const markersSvg = markers.map((marker) => {
        if (marker.value === null || marker.value === undefined) return '';
        const raw = Number(marker.value || 0);
        const x = padLeft + Math.min(raw, axisMax) / Math.max(axisMax, 1e-9) * chartWidth;
        return `
          <line class="plot-marker" x1="${x.toFixed(1)}" y1="${padTop}" x2="${x.toFixed(1)}" y2="${axisY.toFixed(1)}" stroke="${marker.color}"></line>
          <text class="plot-marker-text" x="${x.toFixed(1)}" y="${(padTop + 8).toFixed(1)}" text-anchor="middle">${marker.label}</text>
        `;
      }).join('');
      const ticks = [0, 1 / 3, 2 / 3, 1.0].map((ratio) => {
        const value = axisMax * ratio;
        const x = padLeft + chartWidth * ratio;
        return `<text class="plot-label" x="${x.toFixed(1)}" y="${(height - 4).toFixed(1)}" text-anchor="middle">${formatter(value)}</text>`;
      }).join('');
      return `
        <svg class="plot-svg" viewBox="0 0 ${width} ${height}">
          ${gridLines}
          <line class="plot-axis-line" x1="${padLeft}" y1="${axisY.toFixed(1)}" x2="${(padLeft + chartWidth).toFixed(1)}" y2="${axisY.toFixed(1)}"></line>
          <polygon class="plot-fill" points="${areaPoints.join(' ')}" fill="${color}"></polygon>
          <polyline class="plot-stroke" points="${areaPoints.slice(1, -1).join(' ')}" stroke="${color}"></polyline>
          ${markersSvg}
          ${ticks}
        </svg>
      `;
    }

    function pdfCardHtml(key, title, stat, formatter = fmtMs) {
      if (!stat?.has_data) {
        return `<div class="pdf-card"><div class="pdf-head"><div class="pdf-title">${title}</div><div class="pdf-count">count=0</div></div><div class="waiting">过去 1 秒没有样本</div><div></div></div>`;
      }
      const preset = pdfAxisPresets[key] || { minMax: Number(stat.max || 1), bins: 24, color: '#0f766e' };
      const p95 = Number(stat.p95 || 0);
      const p99 = Number(stat.p99 || 0);
      const maxValue = Number(stat.max || 0);
      const desiredAxisMax = Math.max(
        preset.minMax,
        p95 * 1.25,
        p99 * 1.12,
        Math.min(maxValue * 1.04, p99 * 1.45)
      );
      const axisMax = rememberPdfAxisMax(key, desiredAxisMax, preset.step || 0, preset.minMax);
      const bins = quantilePdfBins(stat, axisMax, preset.bins);
      const markers = [
        { label: 'P50', value: stat.deciles?.[4]?.value, color: '#0f766e' },
        { label: 'P95', value: stat.p95, color: '#1d4ed8' },
        { label: 'P99', value: stat.p99, color: '#7c3aed' },
        { label: 'Max', value: stat.max, color: '#c2410c' },
      ];
      return `
        <div class="pdf-card">
          <div class="pdf-head">
            <div class="pdf-title">${title}</div>
            <div class="pdf-count">count=${fmtInt(stat.count)}</div>
          </div>
          ${buildPlotFromBins(bins, axisMax, fmtMsAxis, preset.color, markers)}
          <div class="pdf-stats">
            <div><span>P50</span><strong>${formatter(stat.deciles?.[4]?.value)}</strong></div>
            <div><span>P95</span><strong>${formatter(stat.p95)}</strong></div>
            <div><span>P99</span><strong>${formatter(stat.p99)}</strong></div>
            <div><span>Max</span><strong>${formatter(stat.max)}</strong></div>
          </div>
        </div>
      `;
    }

    function renderPdfBlock(targetId, series, columns) {
      const el = document.getElementById(targetId);
      const anyData = series.some((item) => item.stat?.has_data);
      if (!anyData) {
        el.innerHTML = `<div class="waiting">过去 1 秒没有样本</div>`;
        return;
      }
      el.innerHTML = `<div class="pdf-grid" style="grid-template-columns:repeat(${columns}, minmax(0, 1fr));">${series.map((item) => pdfCardHtml(item.key, item.label, item.stat)).join('')}</div>`;
    }

    function renderDepthAndQueue(win) {
      const depthHtml = `
        <div class="dock-card">
          <h3>搜索深度</h3>
          ${plotHistogramHtml('depth', win.search_depth_histogram || [], (v) => fmtInt(v), '过去 1 秒没有 playout 样本', '#0f766e', pdfAxisPresets.depth.minMax, pdfAxisPresets.depth.step)}
        </div>
      `;
      const queueHtml = `
        <div class="dock-card">
          <h3>请求队列长度</h3>
          ${plotHistogramHtml('queue', win.queue_length_time_share, fmtPercent, '暂无队列时间占比', '#c2410c', pdfAxisPresets.queue.minMax, pdfAxisPresets.queue.step)}
        </div>
      `;
      document.getElementById('depth-and-queue').innerHTML = `<div class="dock-stack">${depthHtml}${queueHtml}</div>`;
    }

    function renderQueueAndActive(win) {
      document.getElementById('queue-and-active').innerHTML = `
        <div class="mini-card" style="height:100%">
          <h3>活跃推理线程数</h3>
          ${histogramHtml(win.inference_thread_active_time_share, fmtPercent, true, '暂无线程活跃占比')}
        </div>
      `;
    }

    function histogramHtml(buckets, formatter, alt = false, emptyText = '暂无数据') {
      if (!buckets || buckets.length === 0) {
        return `<div class="waiting">${emptyText}</div>`;
      }
      const maxValue = Math.max(...buckets.map((item) => Number(item.value || 0)), 1e-9);
      const columns = histogramColumns(buckets.length);
      return `<div class="histogram" style="grid-template-columns:repeat(${columns}, minmax(0, 1fr));">${buckets.map((item) => {
        const width = Math.max(2, Number(item.value || 0) / maxValue * 100);
        return `
          <div class="hist-row">
            <div class="hist-label">${item.bucket}</div>
            <div class="bar-bg"><div class="bar-fill ${alt ? 'alt' : ''}" style="width:${width}%"></div></div>
            <div class="hist-value">${formatter(item.value)}</div>
          </div>
        `;
      }).join('')}</div>`;
    }

    function renderGpuBatches(win) {
      const overall = histogramHtml(win.gpu_batch_time_share, fmtPercent, false, '暂无 batch 分布');
      const perGpu = win.gpu_batch_time_share_by_gpu || [];
      const overallBuckets = [...(win.gpu_batch_time_share || [])];
      const dominant = overallBuckets.sort((a, b) => Number(b.value || 0) - Number(a.value || 0))[0];
      document.getElementById('gpu-batches').innerHTML = `
        <div class="mini-card" style="height:100%">
          <h3>总体 BatchSize 分布</h3>
          ${overall}
          <div class="plot-footer" style="margin-top:8px">
            <span>活跃 GPU <strong>${fmtInt(perGpu.length)}</strong></span>
            <span>主 batch <strong>${dominant ? dominant.bucket : 'N/A'}</strong></span>
          </div>
        </div>
      `;
    }

    function renderGpuStreams(win) {
      const items = win.cuda_stream_active_time_share_by_gpu || [];
      if (items.length === 0) {
        document.getElementById('gpu-streams').innerHTML = '<div class="waiting">当前后端未提供 GPU stream 活跃样本</div>';
        return;
      }
      const gridClass = items.length > 1 ? 'gpu-grid two-col' : 'gpu-grid';
      document.getElementById('gpu-streams').innerHTML = `<div class="${gridClass}">${items.map((item) => `
        <div class="gpu-card">
          <h3>GPU ${item.gpu}</h3>
          ${histogramHtml(item.buckets, fmtPercent, false, '暂无 stream 活跃数据')}
        </div>
      `).join('')}</div>`;
    }

    function buildSparkline(points, stroke) {
      if (!points || points.length === 0) {
        return '<div class="waiting">暂无趋势数据</div>';
      }
      const width = 320;
      const height = 70;
      const max = Math.max(...points, 1e-9);
      const coords = points.map((value, idx) => {
        const x = points.length === 1 ? width / 2 : idx * (width / (points.length - 1));
        const y = height - (Number(value || 0) / max) * (height - 8) - 4;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      }).join(' ');
      return `
        <svg class="spark" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
          <polyline fill="none" stroke="${stroke}" stroke-width="3" points="${coords}"></polyline>
        </svg>
      `;
    }

    function renderSparklines(history) {
      const visits = history.map((item) => Number(item.visits_per_s || 0));
      const nneval = history.map((item) => Number(item.nn_eval_per_s || 0));
      const lastVisits = visits.length > 0 ? visits[visits.length - 1] : 0;
      const lastNnEval = nneval.length > 0 ? nneval[nneval.length - 1] : 0;
      document.getElementById('sparklines').innerHTML = `
        <div class="spark-card">
          <header><h3>Visits / s</h3><strong>${fmtFloat(lastVisits, 1)}</strong></header>
          ${buildSparkline(visits, '#0f766e')}
        </div>
        <div class="spark-card">
          <header><h3>nnEval / s</h3><strong>${fmtFloat(lastNnEval, 1)}</strong></header>
          ${buildSparkline(nneval, '#c2410c')}
        </div>
      `;
    }

    function renderAll(state) {
      renderStatusGrid(state);
      renderMetricCards(state);
      const latest = state.latest;
      if (!latest) {
        document.getElementById('depth-and-queue').innerHTML = '<div class="waiting">等待来自 KataGo 的快照</div>';
        document.getElementById('search-loop').innerHTML = '<div class="waiting">等待来自 KataGo 的快照</div>';
        document.getElementById('inference-phases').innerHTML = '<div class="waiting">等待来自 KataGo 的快照</div>';
        document.getElementById('queue-and-active').innerHTML = '<div class="waiting">等待来自 KataGo 的快照</div>';
        document.getElementById('gpu-batches').innerHTML = '<div class="waiting">等待来自 KataGo 的快照</div>';
        document.getElementById('gpu-streams').innerHTML = '<div class="waiting">等待来自 KataGo 的快照</div>';
        document.getElementById('sparklines').innerHTML = '<div class="waiting">等待来自 KataGo 的快照</div>';
        return;
      }
      const win = latest.window1s || {};
      renderDepthAndQueue(win);
      renderPdfBlock('search-loop', [
        { key: 'search.total_ms', label: '总循环耗时', stat: win.search_loop?.total_ms },
        { key: 'search.search_ms', label: '搜索耗时', stat: win.search_loop?.search_ms },
        { key: 'search.wait_nn_ms', label: '等待推理耗时', stat: win.search_loop?.wait_nn_ms },
      ], 1);
      renderPdfBlock('inference-phases', [
        { key: 'infer.wait_task_submit_ms', label: '等待任务提交', stat: win.inference?.wait_task_submit_ms },
        { key: 'infer.preprocess_ms', label: '预处理', stat: win.inference?.preprocess_ms },
        { key: 'infer.h2d_ms', label: 'H2D', stat: win.inference?.h2d_ms },
        { key: 'infer.infer_ms', label: '推理', stat: win.inference?.infer_ms },
        { key: 'infer.d2h_ms', label: 'D2H', stat: win.inference?.d2h_ms },
        { key: 'infer.postprocess_ms', label: '后处理', stat: win.inference?.postprocess_ms },
      ], 2);
      renderQueueAndActive(win);
      renderGpuBatches(win);
      renderGpuStreams(win);
      renderSparklines(state.history || []);
    }

    async function fetchState() {
      try {
        const resp = await fetch('/api/state', { cache: 'no-store' });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const state = await resp.json();
        renderAll(state);
      } catch (err) {
        renderAll({ latest: null, history: [] });
        document.getElementById('status-grid').innerHTML = `
          <div class="status-pill"><label>状态</label><strong>页面请求失败</strong></div>
          <div class="status-pill"><label>错误</label><strong class="mono">${String(err)}</strong></div>
        `;
      }
    }

    fetchState();
    window.setInterval(fetchState, 1000);
  </script>
</body>
</html>
"""


def to_int(raw: str, default: int) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class MonitorState:
    def __init__(self, history_size: int) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[Dict[str, Any]] = None
        self._history: Deque[Dict[str, Any]] = deque(maxlen=history_size)
        self._received_count = 0
        self._last_error: Optional[str] = None

    def update(self, snapshot: Dict[str, Any]) -> None:
        totals = snapshot.get("totals", {})
        window = snapshot.get("window1s", {})
        summary = {
            "sequence": snapshot.get("sequence"),
            "timestamp_unix_ms": snapshot.get("timestamp_unix_ms"),
            "visits_per_s": window.get("visits_per_s", 0.0),
            "nn_eval_per_s": window.get("nn_eval_per_s", 0.0),
            "nn_batches_per_s": window.get("nn_batches_per_s", 0.0),
            "avg_batch_size": window.get("avg_batch_size", 0.0),
            "search_threads": totals.get("search_threads", 0),
        }
        with self._lock:
            self._latest = snapshot
            self._history.append(summary)
            self._received_count += 1
            self._last_error = None

    def set_error(self, error: str) -> None:
        with self._lock:
            self._last_error = error

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "generated_at": now_iso(),
                "latest": self._latest,
                "history": list(self._history),
                "receiver": {
                    "received_count": self._received_count,
                    "last_error": self._last_error,
                },
            }


def default_value(
    cli_value: Optional[str],
    env_name: str,
    env_defaults: Dict[str, str],
    fallback: str,
) -> str:
    if cli_value not in (None, ""):
        return cli_value
    if os.environ.get(env_name):
        return os.environ[env_name]
    if env_defaults.get(env_name):
        return env_defaults[env_name]
    return fallback


def make_handler(state: MonitorState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/":
                body = HTML_PAGE.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/api/state":
                payload = json.dumps(state.snapshot(), ensure_ascii=False).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def receiver_loop(sock: socket.socket, state: MonitorState, stop_event: threading.Event) -> None:
    sock.settimeout(0.5)
    while not stop_event.is_set():
        try:
            payload = sock.recv(1024 * 1024)
        except socket.timeout:
            continue
        except OSError as exc:
            if stop_event.is_set():
                return
            state.set_error(f"socket error: {exc}")
            continue

        try:
            snapshot = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            state.set_error(f"invalid payload: {exc}")
            continue
        if isinstance(snapshot, dict):
            state.update(snapshot)
        else:
            state.set_error("payload is not a JSON object")


def bind_unix_socket(socket_path: Path) -> socket.socket:
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(str(socket_path))
    return sock


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    env_sh_path = script_dir.parent / "env.sh"
    env_defaults = load_env_sh_defaults(env_sh_path)

    parser = argparse.ArgumentParser(description="Serve a local realtime KataGo performance monitor page.")
    parser.add_argument("--socket-path", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--history-size", type=int, default=60)
    args = parser.parse_args()

    socket_path = default_value(
        args.socket_path,
        "KATAGO_MONITOR_SOCKET_PATH",
        env_defaults,
        "/tmp/katago_perf_monitor.sock",
    )
    host = default_value(
        args.host,
        "KATAGO_MONITOR_HTTP_HOST",
        env_defaults,
        "0.0.0.0",
    )
    port = args.port if args.port is not None else to_int(
        default_value(None, "KATAGO_MONITOR_HTTP_PORT", env_defaults, "8765"),
        8765,
    )

    state = MonitorState(history_size=max(10, args.history_size))
    sock = bind_unix_socket(Path(socket_path))
    stop_event = threading.Event()
    receiver = threading.Thread(target=receiver_loop, args=(sock, state, stop_event), daemon=True)
    receiver.start()

    httpd = ThreadingHTTPServer((host, port), make_handler(state))
    print(f"monitor page listening on http://{host}:{port}")
    print(f"unix datagram socket: {socket_path}")
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        httpd.shutdown()
        httpd.server_close()
        sock.close()
        receiver.join(timeout=1.0)
        try:
            Path(socket_path).unlink()
        except FileNotFoundError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
