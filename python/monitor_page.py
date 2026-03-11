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


DASHBOARD_PAGE = """<!doctype html>
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
    .brand-meta {
      display: flex;
      align-items: center;
      gap: 10px;
      justify-content: space-between;
    }
    .brand-link {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid rgba(15,118,110,0.18);
      background: rgba(255,255,255,0.74);
      color: var(--accent);
      text-decoration: none;
      font-size: 11px;
      font-weight: 700;
      white-space: nowrap;
    }
    .brand-link:hover {
      background: rgba(255,255,255,0.92);
    }
    .status-grid {
      display: grid;
      grid-template-columns: repeat(7, minmax(0, 1fr));
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
    .timeline-shell {
      min-height: 0;
      overflow: hidden;
    }
    .timeline-top {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: start;
      margin-bottom: 8px;
    }
    .timeline-titlebox .hint {
      margin-bottom: 0;
      min-height: auto;
    }
    .timeline-controls {
      display: grid;
      grid-auto-flow: column;
      gap: 8px;
      align-items: center;
      justify-content: end;
      font-size: 10px;
      color: var(--muted);
    }
    .timeline-controls label {
      display: grid;
      gap: 4px;
      font-size: 9px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .timeline-controls select, .timeline-controls button {
      font: inherit;
      color: var(--ink);
      border: 1px solid var(--line);
      background: var(--panel-strong);
      border-radius: 9px;
      padding: 6px 8px;
    }
    .timeline-controls button {
      cursor: pointer;
    }
    .timeline-summary {
      min-width: 172px;
      text-align: right;
      font-family: var(--mono);
      color: var(--muted);
      font-size: 10px;
      white-space: nowrap;
    }
    .timeline-view {
      flex: 1;
      min-height: 0;
      height: 100%;
      border-radius: 12px;
      border: 1px solid var(--line);
      background:
        linear-gradient(180deg, rgba(255,255,255,0.72), rgba(255,255,255,0.58)),
        linear-gradient(90deg, rgba(15,118,110,0.04), rgba(29,78,216,0.04));
      overflow: hidden;
      position: relative;
      cursor: grab;
    }
    .timeline-view.dragging {
      cursor: grabbing;
    }
    .timeline-svg {
      width: 100%;
      height: 100%;
      display: block;
      user-select: none;
    }
    .timeline-axis {
      stroke: rgba(23,32,42,0.18);
      stroke-width: 1;
    }
    .timeline-grid {
      stroke: rgba(23,32,42,0.12);
      stroke-width: 1;
      stroke-dasharray: 3 4;
    }
    .timeline-lane-divider {
      stroke: rgba(23,32,42,0.08);
      stroke-width: 1;
    }
    .timeline-label {
      fill: var(--muted);
      font-size: 10px;
      font-family: var(--mono);
    }
    .timeline-lane-label {
      fill: var(--ink);
      font-size: 11px;
      font-weight: 700;
    }
    .timeline-block {
      stroke: rgba(23,32,42,0.18);
      stroke-width: 1;
    }
    .timeline-block-label {
      fill: rgba(255,255,255,0.94);
      font-size: 9px;
      font-family: var(--mono);
      pointer-events: none;
    }
    .timeline-arrow {
      fill: none;
      stroke: rgba(23,32,42,0.42);
      stroke-width: 1.2;
      stroke-linecap: round;
    }
    .timeline-legend {
      position: absolute;
      right: 10px;
      top: 8px;
      display: flex;
      gap: 8px;
      padding: 4px 6px;
      border-radius: 999px;
      background: rgba(255,255,255,0.74);
      border: 1px solid rgba(23,32,42,0.08);
      font-size: 9px;
      color: var(--muted);
      pointer-events: none;
    }
    .timeline-legend span {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      white-space: nowrap;
    }
    .timeline-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      display: inline-block;
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
    .panel-stack {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 6px;
      min-height: 0;
      height: 100%;
    }
    .panel-note {
      padding: 7px 8px;
      border-radius: 10px;
      background: rgba(255,255,255,0.68);
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 10px;
      line-height: 1.35;
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
        <div class="brand-meta">
          <div>
            <h1 class="title">搜索资源监控</h1>
            <p class="subtitle">固定一屏展示 1Hz 搜索与推理概览。详细调度线程 / CUDA Stream 时间线已拆到独立页面，避免主 dashboard 被压扁。</p>
          </div>
          <a class="brand-link" href="/timeline">打开时间线页面</a>
        </div>
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
        <h2 id="inference-panel-title">推理执行概况</h2>
        <p id="inference-panel-hint" class="hint">按当前后端模式展示推理侧可解释指标；single-scheduler 模式会切换成保守视图。</p>
        <div id="inference-phases" class="chart-body"></div>
      </div>
      <div class="panel tile-queue">
        <h2 id="activity-panel-title">推理资源占用</h2>
        <p id="activity-panel-hint" class="hint">过去 1 秒时间占比</p>
        <div id="queue-and-active" class="chart-body"></div>
      </div>
      <div class="panel tile-batch">
        <h2 id="batch-panel-title">GPU Batch 分布</h2>
        <p id="batch-panel-hint" class="hint">按当前后端语义汇总；多卡时保持单屏，不逐卡展开</p>
        <div id="gpu-batches" class="chart-body"></div>
      </div>
      <div class="panel tile-streams">
        <h2 id="streams-panel-title">每 GPU 的推理并发数</h2>
        <p id="streams-panel-hint" class="hint">过去 1 秒不同运行中推理图数的时间占比</p>
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
    const inferenceModeOf = (latest) => latest?.status?.inference_mode || 'legacy_worker_threads';
    const isSingleSchedulerMode = (latest) => inferenceModeOf(latest) === 'single_scheduler_slots';
    const inferenceModeLabel = (mode) => {
      if (mode === 'single_scheduler_slots') return '单调度 slot';
      return '传统 worker';
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
    const timelineStageColors = {
      preprocess: '#0f766e',
      h2d: '#1d4ed8',
      infer: '#c2410c',
      d2h: '#16a34a',
      postprocess: '#7c3aed',
    };
    const timelineLaneNames = ['scheduler', 'h2d', 'infer', 'd2h'];
    const timelineStageNames = ['preprocess', 'h2d', 'infer', 'd2h', 'postprocess'];
    const timelineUiState = {
      selectedSlotKey: null,
      spanNs: 50e6,
      centerNs: null,
      followLatest: true,
      drag: null,
      chartLeftPx: 0,
      chartWidthPx: 1,
      viewStartNs: 0,
      latestRangeEndNs: 0,
      latestRangeStartNs: 0,
    };
    let latestRealtimeState = null;
    const hasTimelineDom = () => Boolean(document.getElementById('timeline-view'));

    function clamp(value, minValue, maxValue) {
      return Math.min(Math.max(value, minValue), maxValue);
    }

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
      const mode = inferenceModeOf(latest);
      const errorText = status.last_send_error || receiver.last_error || '无';
      const items = [
        ['最新快照', fmtIso(latest?.timestamp_unix_ms)],
        ['推理模式', inferenceModeLabel(mode)],
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

    function updatePanelLabels(latest) {
      const single = isSingleSchedulerMode(latest);
      document.getElementById('inference-panel-title').textContent = single ? '推理工作量概况' : '推理阶段耗时';
      document.getElementById('inference-panel-hint').textContent = single
        ? '单 scheduler / logical slot 模式下仅 infer_ms 为近似有效值；wait_task_submit / preprocess / H2D / D2H / postprocess 仍未重新接线。'
        : '等待提交 / 预处理 / H2D / 推理 / D2H / 后处理，全部按分位数重建为 PDF 轮廓';
      document.getElementById('activity-panel-title').textContent = single ? '占用推理槽位数' : '活跃推理线程数';
      document.getElementById('activity-panel-hint').textContent = single
        ? '过去 1 秒 non-idle slot 数的时间占比'
        : '过去 1 秒时间占比';
      document.getElementById('batch-panel-title').textContent = single ? '按推理工作量加权的 GPU Batch 分布' : 'GPU Batch 分布';
      document.getElementById('batch-panel-hint').textContent = single
        ? '按 infer_ms / 等效工作量近似加权；多卡时保持单屏，不逐卡展开'
        : '全 GPU 汇总；多卡时保持单屏，不逐卡展开';
      document.getElementById('streams-panel-title').textContent = single ? '每 GPU 的运行中推理图数' : '每 GPU 的 cudaStream 活跃数';
      document.getElementById('streams-panel-hint').textContent = single
        ? '过去 1 秒不同 active infer graph 数的时间占比，不含 H2D / D2H stream'
        : '过去 1 秒不同活跃 stream 数的时间占比';
      const timelineHint = document.getElementById('timeline-hint');
      if (timelineHint) {
        timelineHint.textContent = single
          ? '每秒附带最近约 50ms 的 sampled timeline。默认聚焦这 50ms；拖拽可左右平移，滚轮可水平缩放，`回到最新` 会重新跟随尾部。Scheduler 是真实 CPU span；Infer/D2H 接近完成时刻；H2D 目前是 enqueue proxy。'
          : 'timeline 当前主要服务于 TRT single-scheduler 路径；其他后端暂未接线。';
      }
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

    function renderInferencePhases(win, singleScheduler) {
      if (!singleScheduler) {
        renderPdfBlock('inference-phases', [
          { key: 'infer.wait_task_submit_ms', label: '等待任务提交', stat: win.inference?.wait_task_submit_ms },
          { key: 'infer.preprocess_ms', label: '预处理', stat: win.inference?.preprocess_ms },
          { key: 'infer.h2d_ms', label: 'H2D', stat: win.inference?.h2d_ms },
          { key: 'infer.infer_ms', label: '推理', stat: win.inference?.infer_ms },
          { key: 'infer.d2h_ms', label: 'D2H', stat: win.inference?.d2h_ms },
          { key: 'infer.postprocess_ms', label: '后处理', stat: win.inference?.postprocess_ms },
        ], 2);
        return;
      }

      const inferStat = win.inference?.infer_ms;
      const body = inferStat?.has_data
        ? `<div class="pdf-grid" style="grid-template-columns:repeat(1, minmax(0, 1fr));">${pdfCardHtml('infer.infer_ms', '推理工作量(近似)', inferStat)}</div>`
        : '<div class="waiting">过去 1 秒没有推理样本</div>';
      document.getElementById('inference-phases').innerHTML = `
        <div class="panel-stack">
          <div class="panel-note">当前 TRT overlapping 路径下，realtime 监控只保留了 infer_ms 的近似值。它来自 scheduler 的等效工作量 / ETA 记账，不是 CUDA event 实测；其余 phase 目前仍是占位零值，所以这里不再展示。</div>
          ${body}
        </div>
      `;
    }

    function renderQueueAndActive(win, singleScheduler) {
      const title = singleScheduler ? '占用推理槽位数' : '活跃推理线程数';
      const emptyText = singleScheduler ? '暂无 slot 占用时间占比' : '暂无线程活跃占比';
      document.getElementById('queue-and-active').innerHTML = `
        <div class="mini-card" style="height:100%">
          <h3>${title}</h3>
          ${histogramHtml(win.inference_thread_active_time_share, fmtPercent, true, emptyText)}
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

    function renderGpuBatches(win, singleScheduler) {
      const overall = histogramHtml(win.gpu_batch_time_share, fmtPercent, false, '暂无 batch 分布');
      const perGpu = win.gpu_batch_time_share_by_gpu || [];
      const overallBuckets = [...(win.gpu_batch_time_share || [])];
      const dominant = overallBuckets.sort((a, b) => Number(b.value || 0) - Number(a.value || 0))[0];
      document.getElementById('gpu-batches').innerHTML = `
        <div class="mini-card" style="height:100%">
          <h3>${singleScheduler ? '按推理工作量加权的 BatchSize 分布' : '总体 BatchSize 分布'}</h3>
          ${overall}
          <div class="plot-footer" style="margin-top:8px">
            <span>活跃 GPU <strong>${fmtInt(perGpu.length)}</strong></span>
            <span>${singleScheduler ? '主 batch(近似)' : '主 batch'} <strong>${dominant ? dominant.bucket : 'N/A'}</strong></span>
          </div>
        </div>
      `;
    }

    function renderGpuStreams(win, singleScheduler) {
      const items = win.cuda_stream_active_time_share_by_gpu || [];
      if (items.length === 0) {
        document.getElementById('gpu-streams').innerHTML = singleScheduler
          ? '<div class="waiting">当前后端未提供 infer graph 并发样本</div>'
          : '<div class="waiting">当前后端未提供 GPU stream 活跃样本</div>';
        return;
      }
      const gridClass = items.length > 1 ? 'gpu-grid two-col' : 'gpu-grid';
      document.getElementById('gpu-streams').innerHTML = `<div class="${gridClass}">${items.map((item) => `
        <div class="gpu-card">
          <h3>GPU ${item.gpu}</h3>
          ${histogramHtml(item.buckets, fmtPercent, false, singleScheduler ? '暂无 infer graph 并发数据' : '暂无 stream 活跃数据')}
        </div>
      `).join('')}</div>`;
    }

    function timelineSlotKey(slot) {
      return `${slot.gpu}:${slot.slot}`;
    }

    function timelineTickStepNs(spanNs) {
      const candidates = [1e6, 2e6, 5e6, 10e6, 20e6, 50e6, 100e6, 200e6, 500e6];
      for (const candidate of candidates) {
        if (spanNs / candidate <= 9) return candidate;
      }
      return 1000e6;
    }

    function decodeTimelineSpan(rawSpan, slotInfoBySlot, rangeStartNs) {
      if (!Array.isArray(rawSpan)) return rawSpan;
      const slot = Number(rawSpan[1] ?? -1);
      const slotInfo = slotInfoBySlot.get(slot) || { gpu: -1, slot };
      return {
        id: Number(rawSpan[0] ?? 0),
        slot,
        gpu: Number(slotInfo.gpu ?? -1),
        lane: timelineLaneNames[Number(rawSpan[2] ?? -1)] || 'unknown',
        stage: timelineStageNames[Number(rawSpan[3] ?? -1)] || 'unknown',
        batch_uid: Number(rawSpan[4] ?? 0),
        row: Number(rawSpan[5] ?? -1),
        start_ns: rangeStartNs + Number(rawSpan[6] ?? 0),
        end_ns: rangeStartNs + Number(rawSpan[7] ?? 0),
        dep0: Number(rawSpan[8] ?? 0),
        dep1: Number(rawSpan[9] ?? 0),
      };
    }

    function timelineSpanLabel(span) {
      if (span.stage === 'preprocess') return `prep r${span.row}`;
      if (span.stage === 'h2d') return `h2d r${span.row}`;
      if (span.stage === 'infer') return `infer b${span.batch_uid}`;
      if (span.stage === 'd2h') return `d2h b${span.batch_uid}`;
      if (span.stage === 'postprocess') return `post b${span.batch_uid}`;
      return span.stage || 'event';
    }

    function timelineSpanTitle(span, selectedSlot) {
      const slotText = `GPU ${span.gpu} / Slot ${span.slot}`;
      const batchText = span.batch_uid ? `batch=${span.batch_uid}` : 'batch=n/a';
      const rowText = span.row >= 0 ? `row=${span.row}` : 'row=-';
      const laneText = span.lane || 'unknown';
      const selectedText = selectedSlot && Number(span.slot) === Number(selectedSlot.slot) ? 'selected' : 'other-slot';
      return `${slotText}\n${laneText} / ${span.stage}\n${batchText} ${rowText}\n${selectedText}`;
    }

    function updateTimelineSlotSelect(slots) {
      const select = document.getElementById('timeline-slot-select');
      if (!select) return;
      const renderedKeys = slots.map((slot) => timelineSlotKey(slot)).join('|');
      if (select.dataset.renderedKeys === renderedKeys) {
        if (timelineUiState.selectedSlotKey) select.value = timelineUiState.selectedSlotKey;
        return;
      }
      select.dataset.renderedKeys = renderedKeys;
      select.innerHTML = slots.map((slot) => {
        const key = timelineSlotKey(slot);
        return `<option value="${key}">GPU ${slot.gpu} / Slot ${slot.slot}</option>`;
      }).join('');
      if (timelineUiState.selectedSlotKey) select.value = timelineUiState.selectedSlotKey;
    }

    function renderTimeline(latest) {
      if (!hasTimelineDom()) return;
      const view = document.getElementById('timeline-view');
      const summary = document.getElementById('timeline-summary');
      const timeline = latest?.timeline;
      if (!latest || !timeline || !Array.isArray(timeline.slots) || !Array.isArray(timeline.spans) || !isSingleSchedulerMode(latest)) {
        view.innerHTML = '<div class="waiting">当前只对 TRT single-scheduler 路径提供 timeline；等待可视化样本。</div>';
        summary.textContent = '等待数据';
        document.getElementById('timeline-slot-select').innerHTML = '';
        return;
      }

      const slots = timeline.slots || [];
      if (slots.length === 0) {
        view.innerHTML = '<div class="waiting">当前没有可观察的 logical slot</div>';
        summary.textContent = '暂无 slot';
        document.getElementById('timeline-slot-select').innerHTML = '';
        return;
      }

      if (!slots.some((slot) => timelineSlotKey(slot) === timelineUiState.selectedSlotKey)) {
        const defaultSlot = [...slots].sort((a, b) => Number(a.gpu) - Number(b.gpu) || Number(a.slot) - Number(b.slot))[0];
        timelineUiState.selectedSlotKey = timelineSlotKey(defaultSlot);
        timelineUiState.followLatest = true;
      }
      updateTimelineSlotSelect(slots);

      const selectedSlot = slots.find((slot) => timelineSlotKey(slot) === timelineUiState.selectedSlotKey) || slots[0];
      const latestRangeStartNs = Number(timeline.range_start_ns || 0);
      const latestRangeEndNs = Number(timeline.range_end_ns || 0);
      const slotInfoBySlot = new Map(slots.map((slot) => [Number(slot.slot), slot]));
      const decodedSpans = (timeline.spans || []).map((span) => decodeTimelineSpan(span, slotInfoBySlot, latestRangeStartNs));
      const dataSpanNs = Math.max(latestRangeEndNs - latestRangeStartNs, 20e6);
      timelineUiState.latestRangeStartNs = latestRangeStartNs;
      timelineUiState.latestRangeEndNs = latestRangeEndNs;
      timelineUiState.spanNs = clamp(timelineUiState.spanNs, 5e6, dataSpanNs);
      if (timelineUiState.followLatest || timelineUiState.centerNs === null) {
        timelineUiState.centerNs = latestRangeEndNs - timelineUiState.spanNs / 2;
      }

      const maxViewStartNs = Math.max(latestRangeStartNs, latestRangeEndNs - timelineUiState.spanNs);
      const viewStartNs = clamp(timelineUiState.centerNs - timelineUiState.spanNs / 2, latestRangeStartNs, maxViewStartNs);
      const viewEndNs = viewStartNs + timelineUiState.spanNs;
      timelineUiState.viewStartNs = viewStartNs;
      timelineUiState.centerNs = viewStartNs + timelineUiState.spanNs / 2;

      const width = Math.max(view.clientWidth || 960, 960);
      const height = Math.max(view.clientHeight || 168, 168);
      const leftPad = 152;
      const rightPad = 16;
      const topPad = 22;
      const bottomPad = 22;
      const laneGap = 12;
      const lanes = [
        { id: 'scheduler', label: 'Scheduler Thread' },
        { id: 'h2d', label: `GPU ${selectedSlot.gpu} / Slot ${selectedSlot.slot} H2D` },
        { id: 'infer', label: `GPU ${selectedSlot.gpu} / Slot ${selectedSlot.slot} Infer` },
        { id: 'd2h', label: `GPU ${selectedSlot.gpu} / Slot ${selectedSlot.slot} D2H` },
      ];
      const chartWidth = Math.max(width - leftPad - rightPad, 200);
      const laneHeight = Math.max((height - topPad - bottomPad - laneGap * (lanes.length - 1)) / lanes.length, 20);
      timelineUiState.chartLeftPx = leftPad;
      timelineUiState.chartWidthPx = chartWidth;

      const xForNs = (ns) => leftPad + (Number(ns || 0) - viewStartNs) / Math.max(timelineUiState.spanNs, 1) * chartWidth;
      const stageSpans = decodedSpans.filter((span) => Number(span.end_ns || 0) >= viewStartNs && Number(span.start_ns || 0) <= viewEndNs);
      const schedulerSpans = stageSpans.filter((span) => span.lane === 'scheduler');
      const selectedSlotSpans = stageSpans.filter((span) => Number(span.slot) === Number(selectedSlot.slot));

      const laneSpans = {
        scheduler: schedulerSpans,
        h2d: selectedSlotSpans.filter((span) => span.lane === 'h2d'),
        infer: selectedSlotSpans.filter((span) => span.lane === 'infer'),
        d2h: selectedSlotSpans.filter((span) => span.lane === 'd2h'),
      };

      const visibleRects = new Map();
      const laneY = new Map();
      lanes.forEach((lane, idx) => {
        laneY.set(lane.id, topPad + idx * (laneHeight + laneGap));
      });

      const allBlockSvgs = [];
      for (const lane of lanes) {
        const spans = [...(laneSpans[lane.id] || [])].sort((a, b) => Number(a.start_ns || 0) - Number(b.start_ns || 0));
        for (const span of spans) {
          const laneTop = laneY.get(lane.id);
          const rawStartX = xForNs(span.start_ns);
          const rawEndX = xForNs(span.end_ns);
          const x = Math.max(leftPad, Math.min(rawStartX, width - rightPad));
          const endX = Math.max(x + 2, Math.min(Math.max(rawEndX, rawStartX + 2), width - rightPad));
          const rectWidth = Math.max(endX - x, 2);
          const color = timelineStageColors[span.stage] || '#64748b';
          const selected = Number(span.slot) === Number(selectedSlot.slot);
          const opacity = lane.id === 'scheduler' && !selected ? 0.34 : 0.92;
          const y = laneTop + 7;
          const blockHeight = Math.max(laneHeight - 14, 10);
          visibleRects.set(Number(span.id), {
            x,
            y,
            w: rectWidth,
            h: blockHeight,
            cx: x + rectWidth / 2,
            cy: y + blockHeight / 2,
            span,
          });
          const label = rectWidth >= 46 ? timelineSpanLabel(span) : '';
          const slotPrefix = lane.id === 'scheduler' && !selected ? `s${span.slot} ` : '';
          allBlockSvgs.push(`
            <g>
              <rect class="timeline-block" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${rectWidth.toFixed(1)}" height="${blockHeight.toFixed(1)}" rx="4" fill="${color}" fill-opacity="${opacity}">
                <title>${timelineSpanTitle(span, selectedSlot)}</title>
              </rect>
              ${label ? `<text class="timeline-block-label" x="${(x + 6).toFixed(1)}" y="${(y + blockHeight / 2 + 3).toFixed(1)}">${slotPrefix}${label}</text>` : ''}
            </g>
          `);
        }
      }

      const arrowSvgs = [];
      for (const rect of visibleRects.values()) {
        const span = rect.span;
        const deps = [Number(span.dep0 || 0), Number(span.dep1 || 0)].filter((dep) => dep > 0);
        for (const dep of deps) {
          const from = visibleRects.get(dep);
          if (!from) continue;
          const startX = from.x + from.w;
          const startY = from.cy;
          const endX = rect.x;
          const endY = rect.cy;
          const midX = startX + Math.max((endX - startX) * 0.5, 12);
          arrowSvgs.push(`<path class="timeline-arrow" d="M ${startX.toFixed(1)} ${startY.toFixed(1)} C ${midX.toFixed(1)} ${startY.toFixed(1)}, ${(endX - 14).toFixed(1)} ${endY.toFixed(1)}, ${endX.toFixed(1)} ${endY.toFixed(1)}" marker-end="url(#timeline-arrowhead)"></path>`);
        }
      }

      const tickStepNs = timelineTickStepNs(timelineUiState.spanNs);
      const tickStartNs = Math.ceil(viewStartNs / tickStepNs) * tickStepNs;
      const tickSvgs = [];
      for (let tickNs = tickStartNs; tickNs <= viewEndNs + 1; tickNs += tickStepNs) {
        const x = xForNs(tickNs);
        if (x < leftPad || x > width - rightPad) continue;
        const relMs = (tickNs - latestRangeEndNs) / 1e6;
        tickSvgs.push(`
          <line class="timeline-grid" x1="${x.toFixed(1)}" y1="${topPad - 6}" x2="${x.toFixed(1)}" y2="${(height - bottomPad + 4).toFixed(1)}"></line>
          <text class="timeline-label" x="${x.toFixed(1)}" y="${(height - 6).toFixed(1)}" text-anchor="middle">${fmtFloat(relMs, 1)}ms</text>
        `);
      }

      const laneLabelSvgs = lanes.map((lane) => {
        const y = laneY.get(lane.id);
        return `
          <text class="timeline-lane-label" x="10" y="${(y + 18).toFixed(1)}">${lane.label}</text>
          <line class="timeline-lane-divider" x1="0" y1="${(y + laneHeight + 1).toFixed(1)}" x2="${width}" y2="${(y + laneHeight + 1).toFixed(1)}"></line>
        `;
      }).join('');

      const axisLine = `<line class="timeline-axis" x1="${leftPad}" y1="${(height - bottomPad + 1).toFixed(1)}" x2="${(width - rightPad).toFixed(1)}" y2="${(height - bottomPad + 1).toFixed(1)}"></line>`;
      const relStartMs = (viewStartNs - latestRangeEndNs) / 1e6;
      const relEndMs = (viewEndNs - latestRangeEndNs) / 1e6;
      const droppedSpans = Number(timeline.dropped_spans || 0);
      const droppedSuffix = droppedSpans > 0 ? ` | clipped ${fmtInt(droppedSpans)}` : '';
      summary.textContent = `GPU ${selectedSlot.gpu} / Slot ${selectedSlot.slot} | ${fmtFloat(timelineUiState.spanNs / 1e6, 1)} ms | ${fmtFloat(relStartMs, 1)} .. ${fmtFloat(relEndMs, 1)} ms${droppedSuffix}`;

      view.innerHTML = `
        <svg class="timeline-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
          <defs>
            <marker id="timeline-arrowhead" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
              <path d="M 0 0 L 8 4 L 0 8 z" fill="rgba(23,32,42,0.42)"></path>
            </marker>
          </defs>
          ${tickSvgs.join('')}
          ${axisLine}
          ${laneLabelSvgs}
          ${arrowSvgs.join('')}
          ${allBlockSvgs.join('')}
        </svg>
        <div class="timeline-legend">
          <span><i class="timeline-dot" style="background:${timelineStageColors.preprocess}"></i>Preprocess</span>
          <span><i class="timeline-dot" style="background:${timelineStageColors.h2d}"></i>H2D</span>
          <span><i class="timeline-dot" style="background:${timelineStageColors.infer}"></i>Infer</span>
          <span><i class="timeline-dot" style="background:${timelineStageColors.d2h}"></i>D2H</span>
          <span><i class="timeline-dot" style="background:${timelineStageColors.postprocess}"></i>Post</span>
        </div>
      `;
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
      latestRealtimeState = state;
      renderStatusGrid(state);
      renderMetricCards(state);
      const latest = state.latest;
      updatePanelLabels(latest);
      if (!latest) {
        document.getElementById('depth-and-queue').innerHTML = '<div class="waiting">等待来自 KataGo 的快照</div>';
        document.getElementById('search-loop').innerHTML = '<div class="waiting">等待来自 KataGo 的快照</div>';
        document.getElementById('inference-phases').innerHTML = '<div class="waiting">等待来自 KataGo 的快照</div>';
        document.getElementById('queue-and-active').innerHTML = '<div class="waiting">等待来自 KataGo 的快照</div>';
        document.getElementById('gpu-batches').innerHTML = '<div class="waiting">等待来自 KataGo 的快照</div>';
        document.getElementById('gpu-streams').innerHTML = '<div class="waiting">等待来自 KataGo 的快照</div>';
        document.getElementById('sparklines').innerHTML = '<div class="waiting">等待来自 KataGo 的快照</div>';
        renderTimeline(null);
        return;
      }
      const win = latest.window1s || {};
      const singleScheduler = isSingleSchedulerMode(latest);
      renderDepthAndQueue(win);
      renderPdfBlock('search-loop', [
        { key: 'search.total_ms', label: '总循环耗时', stat: win.search_loop?.total_ms },
        { key: 'search.search_ms', label: '搜索耗时', stat: win.search_loop?.search_ms },
        { key: 'search.wait_nn_ms', label: '等待推理耗时', stat: win.search_loop?.wait_nn_ms },
      ], 1);
      renderInferencePhases(win, singleScheduler);
      renderQueueAndActive(win, singleScheduler);
      renderGpuBatches(win, singleScheduler);
      renderGpuStreams(win, singleScheduler);
      renderSparklines(state.history || []);
      renderTimeline(latest);
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

    const timelineSlotSelect = document.getElementById('timeline-slot-select');
    const timelineLatestBtn = document.getElementById('timeline-latest-btn');
    const timelineView = document.getElementById('timeline-view');

    if (timelineSlotSelect) {
      timelineSlotSelect.addEventListener('change', (event) => {
        timelineUiState.selectedSlotKey = event.target.value || null;
        timelineUiState.followLatest = true;
        timelineUiState.centerNs = null;
        renderTimeline(latestRealtimeState?.latest || null);
      });
    }

    if (timelineLatestBtn) {
      timelineLatestBtn.addEventListener('click', () => {
        timelineUiState.followLatest = true;
        timelineUiState.centerNs = null;
        renderTimeline(latestRealtimeState?.latest || null);
      });
    }

    if (timelineView) {
      timelineView.addEventListener('mousedown', (event) => {
        if (event.button !== 0 || timelineUiState.chartWidthPx <= 1 || !latestRealtimeState?.latest?.timeline) return;
        timelineUiState.drag = { startX: event.clientX, centerNs: timelineUiState.centerNs };
        timelineUiState.followLatest = false;
        event.currentTarget.classList.add('dragging');
        event.preventDefault();
      });
    }

    window.addEventListener('mousemove', (event) => {
      if (!timelineUiState.drag || timelineUiState.chartWidthPx <= 1) return;
      const deltaPx = event.clientX - timelineUiState.drag.startX;
      const deltaNs = deltaPx / timelineUiState.chartWidthPx * timelineUiState.spanNs;
      timelineUiState.centerNs = timelineUiState.drag.centerNs - deltaNs;
      renderTimeline(latestRealtimeState?.latest || null);
    });

    window.addEventListener('mouseup', () => {
      timelineUiState.drag = null;
      if (timelineView) timelineView.classList.remove('dragging');
    });

    if (timelineView) {
      timelineView.addEventListener('wheel', (event) => {
        if (timelineUiState.chartWidthPx <= 1 || !latestRealtimeState?.latest?.timeline) return;
        const rect = event.currentTarget.getBoundingClientRect();
        const localX = event.clientX - rect.left;
        if (localX < timelineUiState.chartLeftPx) return;
        event.preventDefault();
        const focusRatio = clamp((localX - timelineUiState.chartLeftPx) / timelineUiState.chartWidthPx, 0, 1);
        const focusNs = timelineUiState.viewStartNs + focusRatio * timelineUiState.spanNs;
        const dataSpanNs = Math.max(timelineUiState.latestRangeEndNs - timelineUiState.latestRangeStartNs, 20e6);
        const zoomFactor = event.deltaY > 0 ? 1.18 : 0.84;
        const newSpanNs = clamp(timelineUiState.spanNs * zoomFactor, 5e6, dataSpanNs);
        const newStartNs = focusNs - focusRatio * newSpanNs;
        timelineUiState.spanNs = newSpanNs;
        timelineUiState.centerNs = newStartNs + newSpanNs / 2;
        timelineUiState.followLatest = false;
        renderTimeline(latestRealtimeState?.latest || null);
      }, { passive: false });
    }

    fetchState();
    window.setInterval(fetchState, 1000);
  </script>
</body>
</html>
"""


TIMELINE_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KataGo Timeline</title>
  <style>
    :root {
      --bg: #f4efe6;
      --panel: rgba(255,255,255,0.82);
      --panel-strong: rgba(255,255,255,0.94);
      --ink: #17202a;
      --muted: #5d6670;
      --line: rgba(23,32,42,0.12);
      --accent: #0f766e;
      --accent-2: #c2410c;
      --accent-3: #1d4ed8;
      --shadow: 0 14px 34px rgba(25, 32, 40, 0.08);
      --radius: 18px;
      --mono: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
      --sans: "IBM Plex Sans", "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; min-width: 0; }
    html, body { height: 100%; }
    body {
      margin: 0;
      font-family: var(--sans);
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.18), transparent 38%),
        radial-gradient(circle at top right, rgba(194,65,12,0.12), transparent 32%),
        linear-gradient(180deg, #f8f4ec 0%, #f2ebe0 100%);
    }
    .timeline-app {
      width: 100vw;
      height: 100vh;
      padding: 14px;
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
      gap: 12px;
    }
    .panel {
      background: var(--panel);
      backdrop-filter: blur(14px);
      border: 1px solid rgba(255,255,255,0.55);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 14px 16px;
      min-height: 0;
      overflow: hidden;
    }
    .topbar {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: start;
    }
    .eyebrow {
      font-size: 11px;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
    }
    h1 {
      margin: 0;
      font-size: 34px;
      line-height: 1.02;
    }
    .subtitle {
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
      max-width: 78rem;
    }
    .nav-link {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 10px 14px;
      border-radius: 999px;
      border: 1px solid rgba(15,118,110,0.18);
      background: var(--panel-strong);
      color: var(--accent);
      text-decoration: none;
      font-size: 13px;
      font-weight: 700;
      white-space: nowrap;
    }
    .statusbar {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: center;
    }
    .status-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
      font-family: var(--mono);
    }
    .status-pill {
      padding: 8px 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.78);
      border: 1px solid rgba(23,32,42,0.08);
      white-space: nowrap;
    }
    .status-pill.paused {
      color: #9a3412;
      border-color: rgba(194,65,12,0.18);
      background: rgba(255,237,213,0.82);
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      justify-content: end;
      gap: 10px;
      align-items: end;
    }
    .controls label {
      display: grid;
      gap: 5px;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }
    .controls select,
    .controls button {
      font: inherit;
      color: var(--ink);
      border: 1px solid var(--line);
      background: var(--panel-strong);
      border-radius: 10px;
      padding: 8px 10px;
      min-height: 38px;
    }
    .controls button {
      cursor: pointer;
      font-weight: 700;
    }
    .controls button.pause-active {
      color: #9a3412;
      border-color: rgba(194,65,12,0.18);
      background: rgba(255,237,213,0.9);
    }
    .timeline-shell {
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
      gap: 12px;
      min-height: 0;
    }
    .timeline-summary {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
      font-family: var(--mono);
    }
    .timeline-summary strong {
      color: var(--ink);
    }
    .timeline-summary .hint {
      font-family: var(--sans);
      font-size: 12px;
      line-height: 1.4;
    }
    .timeline-view {
      min-height: 0;
      height: 100%;
      border-radius: 16px;
      border: 1px solid var(--line);
      background:
        linear-gradient(180deg, rgba(255,255,255,0.78), rgba(255,255,255,0.62)),
        linear-gradient(90deg, rgba(15,118,110,0.04), rgba(29,78,216,0.04));
      overflow: hidden;
      position: relative;
      cursor: grab;
    }
    .timeline-view.dragging {
      cursor: grabbing;
    }
    .timeline-svg {
      width: 100%;
      height: 100%;
      display: block;
      user-select: none;
    }
    .timeline-axis {
      stroke: rgba(23,32,42,0.18);
      stroke-width: 1;
    }
    .timeline-grid {
      stroke: rgba(23,32,42,0.12);
      stroke-width: 1;
      stroke-dasharray: 3 4;
    }
    .timeline-lane-divider {
      stroke: rgba(23,32,42,0.08);
      stroke-width: 1;
    }
    .timeline-label {
      fill: var(--muted);
      font-size: 12px;
      font-family: var(--mono);
    }
    .timeline-lane-label {
      fill: var(--ink);
      font-size: 13px;
      font-weight: 700;
    }
    .timeline-block {
      stroke: rgba(23,32,42,0.18);
      stroke-width: 1;
    }
    .timeline-block-label {
      fill: rgba(255,255,255,0.96);
      font-size: 10px;
      font-family: var(--mono);
      pointer-events: none;
    }
    .timeline-arrow {
      fill: none;
      stroke: rgba(23,32,42,0.42);
      stroke-width: 1.3;
      stroke-linecap: round;
    }
    .timeline-legend {
      position: absolute;
      right: 14px;
      top: 12px;
      display: flex;
      gap: 10px;
      padding: 5px 8px;
      border-radius: 999px;
      background: rgba(255,255,255,0.78);
      border: 1px solid rgba(23,32,42,0.08);
      font-size: 10px;
      color: var(--muted);
      pointer-events: none;
    }
    .timeline-legend span {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      white-space: nowrap;
    }
    .timeline-dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      display: inline-block;
    }
    .waiting {
      padding: 8px;
      text-align: center;
      color: var(--muted);
      font-size: 13px;
      border: 1px dashed rgba(23,32,42,0.18);
      border-radius: 12px;
      background: rgba(255,255,255,0.52);
      min-height: 0;
      height: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
    }
  </style>
</head>
<body>
  <div class="timeline-app">
    <section class="panel topbar">
      <div>
        <div class="eyebrow">KataGo Realtime Timeline</div>
        <h1>调度线程 / CUDA Stream 时间线</h1>
        <p class="subtitle">独立页面只展示 sampled timeline。默认跟随最近窗口；可拖拽平移、滚轮水平缩放、切换 slot，并且支持暂停自动刷新后停在某个快照上慢慢看。</p>
      </div>
      <a class="nav-link" href="/">返回 Dashboard</a>
    </section>

    <section class="panel statusbar">
      <div id="timeline-status-meta" class="status-meta">
        <div class="status-pill">等待快照</div>
      </div>
      <div class="controls">
        <button id="timeline-pause-btn" type="button">暂停刷新</button>
        <button id="timeline-refresh-btn" type="button">手动刷新</button>
        <button id="timeline-latest-btn" type="button">回到最新</button>
      </div>
    </section>

    <section class="panel timeline-shell">
      <div id="timeline-summary" class="timeline-summary">
        <span><strong>等待数据</strong></span>
      </div>
      <div id="timeline-view" class="timeline-view"></div>
    </section>
  </div>

  <script>
    const fmtFloat = (v, digits = 2) => {
      if (v === null || v === undefined || Number.isNaN(Number(v))) return 'N/A';
      return Number(v).toFixed(digits);
    };
    const fmtInt = (v) => Number(v || 0).toLocaleString('en-US');
    const fmtIso = (unixMs) => {
      if (!unixMs) return '等待数据';
      return new Date(unixMs).toLocaleString('zh-CN', { hour12: false });
    };
    const clamp = (value, minValue, maxValue) => Math.min(Math.max(value, minValue), maxValue);
    const inferenceModeOf = (latest) => latest?.status?.inference_mode || 'legacy_worker_threads';
    const isSingleSchedulerMode = (latest) => inferenceModeOf(latest) === 'single_scheduler_slots';
    const timelineStageColors = {
      preprocess: '#0f766e',
      h2d: '#1d4ed8',
      infer: '#c2410c',
      d2h: '#16a34a',
      postprocess: '#7c3aed',
    };
    const timelineLaneNames = ['scheduler', 'h2d', 'infer', 'd2h'];
    const timelineStageNames = ['preprocess', 'h2d', 'infer', 'd2h', 'postprocess'];
    const minTimelineSpanNs = 1e3;
    const timelineUiState = {
      spanNs: 50e6,
      centerNs: null,
      followLatest: true,
      drag: null,
      chartLeftPx: 0,
      chartWidthPx: 1,
      viewStartNs: 0,
      latestRangeEndNs: 0,
      latestRangeStartNs: 0,
      paused: false,
    };
    let latestRealtimeState = null;

    function timelineSlotKey(slot) {
      return `${slot.gpu}:${slot.slot}`;
    }

    function timelineTickStepNs(spanNs) {
      const candidates = [1e2, 2e2, 5e2, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8, 2e8, 5e8];
      for (const candidate of candidates) {
        if (spanNs / candidate <= 9) return candidate;
      }
      return 1000e6;
    }

    function fmtTimelineNs(ns) {
      const absNs = Math.abs(Number(ns || 0));
      if (absNs >= 1e6) return `${fmtFloat(ns / 1e6, absNs >= 10e6 ? 1 : 3)}ms`;
      if (absNs >= 1e3) return `${fmtFloat(ns / 1e3, absNs >= 10e3 ? 1 : 3)}us`;
      return `${fmtFloat(ns, 0)}ns`;
    }

    function timelineScaleForSpan(spanNs, tickStepNs) {
      const absSpanNs = Math.abs(Number(spanNs || 0));
      let unit = 'ns';
      let divisor = 1;
      if (absSpanNs >= 1e6) {
        unit = 'ms';
        divisor = 1e6;
      } else if (absSpanNs >= 1e3) {
        unit = 'us';
        divisor = 1e3;
      }
      const scaledStep = Math.abs(Number(tickStepNs || spanNs || 0)) / divisor;
      let decimals = 0;
      if (unit !== 'ns') {
        if (scaledStep < 0.01) decimals = 3;
        else if (scaledStep < 0.1) decimals = 2;
        else if (scaledStep < 1) decimals = 1;
      }
      return { unit, divisor, decimals };
    }

    function fmtTimelineNsWithScale(ns, scale) {
      return `${fmtFloat(Number(ns || 0) / scale.divisor, scale.decimals)}${scale.unit}`;
    }

    function decodeTimelineSpan(rawSpan, slotInfoBySlot, rangeStartNs) {
      if (!Array.isArray(rawSpan)) return rawSpan;
      const slot = Number(rawSpan[1] ?? -1);
      const slotInfo = slotInfoBySlot.get(slot) || { gpu: -1, slot };
      return {
        id: Number(rawSpan[0] ?? 0),
        slot,
        gpu: Number(slotInfo.gpu ?? -1),
        lane: timelineLaneNames[Number(rawSpan[2] ?? -1)] || 'unknown',
        stage: timelineStageNames[Number(rawSpan[3] ?? -1)] || 'unknown',
        batch_uid: Number(rawSpan[4] ?? 0),
        row: Number(rawSpan[5] ?? -1),
        start_ns: rangeStartNs + Number(rawSpan[6] ?? 0),
        end_ns: rangeStartNs + Number(rawSpan[7] ?? 0),
        dep0: Number(rawSpan[8] ?? 0),
        dep1: Number(rawSpan[9] ?? 0),
      };
    }

    function timelineSpanLabel(span) {
      if (span.stage === 'preprocess') return `prep r${span.row}`;
      if (span.stage === 'h2d') return `h2d r${span.row}`;
      if (span.stage === 'infer') return `infer b${span.batch_uid}`;
      if (span.stage === 'd2h') return `d2h b${span.batch_uid}`;
      if (span.stage === 'postprocess') return `post b${span.batch_uid}`;
      return span.stage || 'event';
    }

    function timelineSpanTitle(span, selectedSlot) {
      const slotText = `GPU ${span.gpu} / Slot ${span.slot}`;
      const batchText = span.batch_uid ? `batch=${span.batch_uid}` : 'batch=n/a';
      const rowText = span.row >= 0 ? `row=${span.row}` : 'row=-';
      const laneText = span.lane || 'unknown';
      const selectedText = selectedSlot && Number(span.slot) === Number(selectedSlot.slot) ? 'selected' : 'other-slot';
      return `${slotText}\\n${laneText} / ${span.stage}\\n${batchText} ${rowText}\\n${selectedText}`;
    }

    function updateStatusBar(state) {
      const latest = state?.latest || null;
      const receiver = state?.receiver || {};
      const pills = [];
      pills.push(`<div class="status-pill ${timelineUiState.paused ? 'paused' : ''}">${timelineUiState.paused ? '自动刷新已暂停' : '自动刷新中'}</div>`);
      pills.push(`<div class="status-pill">最新快照 ${fmtIso(latest?.timestamp_unix_ms)}</div>`);
      pills.push(`<div class="status-pill">sequence ${fmtInt(latest?.sequence || 0)}</div>`);
      pills.push(`<div class="status-pill">模式 ${latest ? inferenceModeOf(latest) : '等待数据'}</div>`);
      pills.push(`<div class="status-pill">接收快照 ${fmtInt(receiver.received_count || 0)}</div>`);
      document.getElementById('timeline-status-meta').innerHTML = pills.join('');
      const pauseBtn = document.getElementById('timeline-pause-btn');
      pauseBtn.textContent = timelineUiState.paused ? '恢复刷新' : '暂停刷新';
      pauseBtn.classList.toggle('pause-active', timelineUiState.paused);
    }

    function renderTimeline(latest) {
      const view = document.getElementById('timeline-view');
      const summary = document.getElementById('timeline-summary');
      const timeline = latest?.timeline;
      if (!latest || !timeline || !Array.isArray(timeline.slots) || !Array.isArray(timeline.spans) || !isSingleSchedulerMode(latest)) {
        view.innerHTML = '<div class="waiting">当前只对 TRT single-scheduler 路径提供 timeline；等待可视化样本。</div>';
        summary.innerHTML = '<span><strong>等待数据</strong></span>';
        return;
      }

      const slots = timeline.slots || [];
      if (slots.length === 0) {
        view.innerHTML = '<div class="waiting">当前没有可观察的 logical slot</div>';
        summary.innerHTML = '<span><strong>暂无 slot</strong></span>';
        return;
      }

      const orderedSlots = [...slots].sort((a, b) => Number(a.gpu) - Number(b.gpu) || Number(a.slot) - Number(b.slot));
      const latestRangeStartNs = Number(timeline.range_start_ns || 0);
      const latestRangeEndNs = Number(timeline.range_end_ns || 0);
      const slotInfoBySlot = new Map(slots.map((slot) => [Number(slot.slot), slot]));
      const decodedSpans = (timeline.spans || []).map((span) => decodeTimelineSpan(span, slotInfoBySlot, latestRangeStartNs));
      const dataSpanNs = Math.max(latestRangeEndNs - latestRangeStartNs, 20e6);
      timelineUiState.latestRangeStartNs = latestRangeStartNs;
      timelineUiState.latestRangeEndNs = latestRangeEndNs;
      timelineUiState.spanNs = clamp(timelineUiState.spanNs, minTimelineSpanNs, dataSpanNs);
      if (timelineUiState.followLatest || timelineUiState.centerNs === null) {
        timelineUiState.centerNs = latestRangeEndNs - timelineUiState.spanNs / 2;
      }

      const maxViewStartNs = Math.max(latestRangeStartNs, latestRangeEndNs - timelineUiState.spanNs);
      const viewStartNs = clamp(timelineUiState.centerNs - timelineUiState.spanNs / 2, latestRangeStartNs, maxViewStartNs);
      const viewEndNs = viewStartNs + timelineUiState.spanNs;
      timelineUiState.viewStartNs = viewStartNs;
      timelineUiState.centerNs = viewStartNs + timelineUiState.spanNs / 2;

      const width = Math.max(view.clientWidth || 1480, 1480);
      const height = Math.max(view.clientHeight || 720, 720);
      const leftPad = 194;
      const rightPad = 22;
      const topPad = 28;
      const bottomPad = 28;
      const laneGap = 12;
      const lanes = [{ id: 'scheduler', label: 'Scheduler Thread', slot: null, laneName: 'scheduler' }];
      for (const slot of orderedSlots) {
        lanes.push({ id: `slot-${slot.slot}-h2d`, label: `GPU ${slot.gpu} / Slot ${slot.slot} H2D`, slot: Number(slot.slot), laneName: 'h2d' });
        lanes.push({ id: `slot-${slot.slot}-infer`, label: `GPU ${slot.gpu} / Slot ${slot.slot} Infer`, slot: Number(slot.slot), laneName: 'infer' });
        lanes.push({ id: `slot-${slot.slot}-d2h`, label: `GPU ${slot.gpu} / Slot ${slot.slot} D2H`, slot: Number(slot.slot), laneName: 'd2h' });
      }
      const chartWidth = Math.max(width - leftPad - rightPad, 300);
      const laneHeight = Math.max((height - topPad - bottomPad - laneGap * (lanes.length - 1)) / lanes.length, 34);
      timelineUiState.chartLeftPx = leftPad;
      timelineUiState.chartWidthPx = chartWidth;

      const xForNs = (ns) => leftPad + (Number(ns || 0) - viewStartNs) / Math.max(timelineUiState.spanNs, 1) * chartWidth;
      const stageSpans = decodedSpans.filter((span) => Number(span.end_ns || 0) >= viewStartNs && Number(span.start_ns || 0) <= viewEndNs);
      const laneSpans = new Map();
      laneSpans.set('scheduler', stageSpans.filter((span) => span.lane === 'scheduler'));
      for (const lane of lanes) {
        if (lane.laneName === 'scheduler') continue;
        laneSpans.set(lane.id, stageSpans.filter((span) => Number(span.slot) === lane.slot && span.lane === lane.laneName));
      }

      const visibleRects = new Map();
      const laneY = new Map();
      lanes.forEach((lane, idx) => {
        laneY.set(lane.id, topPad + idx * (laneHeight + laneGap));
      });

      const allBlockSvgs = [];
      for (const lane of lanes) {
        const spans = [...(laneSpans.get(lane.id) || [])].sort((a, b) => Number(a.start_ns || 0) - Number(b.start_ns || 0));
        for (const span of spans) {
          const laneTop = laneY.get(lane.id);
          const rawStartX = xForNs(span.start_ns);
          const rawEndX = xForNs(span.end_ns);
          const x = Math.max(leftPad, Math.min(rawStartX, width - rightPad));
          const endX = Math.max(x + 3, Math.min(Math.max(rawEndX, rawStartX + 3), width - rightPad));
          const rectWidth = Math.max(endX - x, 3);
          const color = timelineStageColors[span.stage] || '#64748b';
          const opacity = lane.laneName === 'scheduler' ? 0.56 : 0.94;
          const y = laneTop + 12;
          const blockHeight = Math.max(laneHeight - 24, 24);
          visibleRects.set(Number(span.id), {
            x,
            y,
            w: rectWidth,
            h: blockHeight,
            cx: x + rectWidth / 2,
            cy: y + blockHeight / 2,
            span,
          });
          const label = rectWidth >= 58 ? timelineSpanLabel(span) : '';
          const slotPrefix = lane.laneName === 'scheduler' ? `s${span.slot} ` : '';
          allBlockSvgs.push(`
            <g>
              <rect class="timeline-block" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${rectWidth.toFixed(1)}" height="${blockHeight.toFixed(1)}" rx="6" fill="${color}" fill-opacity="${opacity}">
                <title>${timelineSpanTitle(span, null)}</title>
              </rect>
              ${label ? `<text class="timeline-block-label" x="${(x + 8).toFixed(1)}" y="${(y + blockHeight / 2 + 4).toFixed(1)}">${slotPrefix}${label}</text>` : ''}
            </g>
          `);
        }
      }

      const arrowSvgs = [];
      for (const rect of visibleRects.values()) {
        const span = rect.span;
        const deps = [Number(span.dep0 || 0), Number(span.dep1 || 0)].filter((dep) => dep > 0);
        for (const dep of deps) {
          const from = visibleRects.get(dep);
          if (!from) continue;
          const startX = from.x + from.w;
          const startY = from.cy;
          const endX = rect.x;
          const endY = rect.cy;
          const midX = startX + Math.max((endX - startX) * 0.5, 14);
          arrowSvgs.push(`<path class="timeline-arrow" d="M ${startX.toFixed(1)} ${startY.toFixed(1)} C ${midX.toFixed(1)} ${startY.toFixed(1)}, ${(endX - 18).toFixed(1)} ${endY.toFixed(1)}, ${endX.toFixed(1)} ${endY.toFixed(1)}" marker-end="url(#timeline-arrowhead)"></path>`);
        }
      }

      const tickStepNs = timelineTickStepNs(timelineUiState.spanNs);
      const axisScale = timelineScaleForSpan(timelineUiState.spanNs, tickStepNs);
      const tickStartNs = Math.ceil(viewStartNs / tickStepNs) * tickStepNs;
      const tickSvgs = [];
      for (let tickNs = tickStartNs; tickNs <= viewEndNs + 1; tickNs += tickStepNs) {
        const x = xForNs(tickNs);
        if (x < leftPad || x > width - rightPad) continue;
        const relNs = tickNs - viewStartNs;
        tickSvgs.push(`
          <line class="timeline-grid" x1="${x.toFixed(1)}" y1="${topPad - 10}" x2="${x.toFixed(1)}" y2="${(height - bottomPad + 6).toFixed(1)}"></line>
          <text class="timeline-label" x="${x.toFixed(1)}" y="${(height - 8).toFixed(1)}" text-anchor="middle">${fmtTimelineNsWithScale(relNs, axisScale)}</text>
        `);
      }

      const laneLabelSvgs = lanes.map((lane) => {
        const y = laneY.get(lane.id);
        return `
          <text class="timeline-lane-label" x="14" y="${(y + 24).toFixed(1)}">${lane.label}</text>
          <line class="timeline-lane-divider" x1="0" y1="${(y + laneHeight + 2).toFixed(1)}" x2="${width}" y2="${(y + laneHeight + 2).toFixed(1)}"></line>
        `;
      }).join('');

      const axisLine = `<line class="timeline-axis" x1="${leftPad}" y1="${(height - bottomPad + 1).toFixed(1)}" x2="${(width - rightPad).toFixed(1)}" y2="${(height - bottomPad + 1).toFixed(1)}"></line>`;
      const sampleOffsetStartNs = viewStartNs - latestRangeStartNs;
      const sampleOffsetEndNs = viewEndNs - latestRangeStartNs;
      const droppedSpans = Number(timeline.dropped_spans || 0);
      const droppedSuffix = droppedSpans > 0 ? ` | clipped ${fmtInt(droppedSpans)}` : '';
      summary.innerHTML = `
        <span><strong>${fmtInt(orderedSlots.length)} 个 slot 全部展开</strong></span>
        <span>窗口 ${fmtTimelineNsWithScale(timelineUiState.spanNs, axisScale)}</span>
        <span>窗口内 0 .. ${fmtTimelineNsWithScale(timelineUiState.spanNs, axisScale)}</span>
        <span>样本内 ${fmtTimelineNsWithScale(sampleOffsetStartNs, axisScale)} .. ${fmtTimelineNsWithScale(sampleOffsetEndNs, axisScale)}</span>
        <span>样本范围 ${fmtTimelineNsWithScale(latestRangeEndNs - latestRangeStartNs, axisScale)}</span>
        <span>${timelineUiState.paused ? '已暂停自动刷新' : '自动跟随最新样本中'}</span>
        <span>${droppedSuffix || '未截断 span'}</span>
        <span class="hint">Scheduler/Pre/Post 来自 CPU 时钟；H2D/Infer/D2H 时长来自 cudaEvent timing，位置按依赖关系回填。</span>
      `;

      view.innerHTML = `
        <svg class="timeline-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
          <defs>
            <marker id="timeline-arrowhead" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
              <path d="M 0 0 L 8 4 L 0 8 z" fill="rgba(23,32,42,0.42)"></path>
            </marker>
          </defs>
          ${tickSvgs.join('')}
          ${axisLine}
          ${laneLabelSvgs}
          ${arrowSvgs.join('')}
          ${allBlockSvgs.join('')}
        </svg>
        <div class="timeline-legend">
          <span><i class="timeline-dot" style="background:${timelineStageColors.preprocess}"></i>Preprocess</span>
          <span><i class="timeline-dot" style="background:${timelineStageColors.h2d}"></i>H2D</span>
          <span><i class="timeline-dot" style="background:${timelineStageColors.infer}"></i>Infer</span>
          <span><i class="timeline-dot" style="background:${timelineStageColors.d2h}"></i>D2H</span>
          <span><i class="timeline-dot" style="background:${timelineStageColors.postprocess}"></i>Post</span>
        </div>
      `;
    }

    async function fetchState(force = false) {
      if (timelineUiState.paused && !force) {
        updateStatusBar(latestRealtimeState);
        return;
      }
      try {
        const resp = await fetch('/api/state', { cache: 'no-store' });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const state = await resp.json();
        latestRealtimeState = state;
        updateStatusBar(state);
        renderTimeline(state?.latest || null);
      } catch (err) {
        updateStatusBar(latestRealtimeState);
        document.getElementById('timeline-view').innerHTML = `<div class="waiting">页面请求失败: ${String(err)}</div>`;
      }
    }

    const timelinePauseBtn = document.getElementById('timeline-pause-btn');
    const timelineRefreshBtn = document.getElementById('timeline-refresh-btn');
    const timelineLatestBtn = document.getElementById('timeline-latest-btn');
    const timelineView = document.getElementById('timeline-view');

    timelinePauseBtn.addEventListener('click', () => {
      timelineUiState.paused = !timelineUiState.paused;
      updateStatusBar(latestRealtimeState);
    });

    timelineRefreshBtn.addEventListener('click', () => {
      fetchState(true);
    });

    timelineLatestBtn.addEventListener('click', () => {
      timelineUiState.followLatest = true;
      timelineUiState.centerNs = null;
      renderTimeline(latestRealtimeState?.latest || null);
    });

    timelineView.addEventListener('mousedown', (event) => {
      if (event.button !== 0 || timelineUiState.chartWidthPx <= 1 || !latestRealtimeState?.latest?.timeline) return;
      timelineUiState.drag = { startX: event.clientX, centerNs: timelineUiState.centerNs };
      timelineUiState.followLatest = false;
      event.currentTarget.classList.add('dragging');
      event.preventDefault();
    });

    window.addEventListener('mousemove', (event) => {
      if (!timelineUiState.drag || timelineUiState.chartWidthPx <= 1) return;
      const deltaPx = event.clientX - timelineUiState.drag.startX;
      const deltaNs = deltaPx / timelineUiState.chartWidthPx * timelineUiState.spanNs;
      timelineUiState.centerNs = timelineUiState.drag.centerNs - deltaNs;
      renderTimeline(latestRealtimeState?.latest || null);
    });

    window.addEventListener('mouseup', () => {
      timelineUiState.drag = null;
      timelineView.classList.remove('dragging');
    });

    timelineView.addEventListener('wheel', (event) => {
      if (timelineUiState.chartWidthPx <= 1 || !latestRealtimeState?.latest?.timeline) return;
      const rect = event.currentTarget.getBoundingClientRect();
      const localX = event.clientX - rect.left;
      if (localX < timelineUiState.chartLeftPx) return;
      event.preventDefault();
      const focusRatio = clamp((localX - timelineUiState.chartLeftPx) / timelineUiState.chartWidthPx, 0, 1);
      const focusNs = timelineUiState.viewStartNs + focusRatio * timelineUiState.spanNs;
      const dataSpanNs = Math.max(timelineUiState.latestRangeEndNs - timelineUiState.latestRangeStartNs, 20e6);
      const zoomFactor = event.deltaY > 0 ? 1.12 : 0.89;
      const newSpanNs = clamp(timelineUiState.spanNs * zoomFactor, minTimelineSpanNs, dataSpanNs);
      const newStartNs = focusNs - focusRatio * newSpanNs;
      timelineUiState.spanNs = newSpanNs;
      timelineUiState.centerNs = newStartNs + newSpanNs / 2;
      timelineUiState.followLatest = false;
      renderTimeline(latestRealtimeState?.latest || null);
    }, { passive: false });

    updateStatusBar(null);
    fetchState(true);
    window.setInterval(() => fetchState(false), 1000);
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
                body = DASHBOARD_PAGE.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/timeline":
                body = TIMELINE_PAGE.encode("utf-8")
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
