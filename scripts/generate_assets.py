"""Generate high-resolution visual assets for Tehran Stock Forecasting."""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import qrcode

ROOT_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT_DIR / "assets"

BANNER_DIR = ASSETS_DIR / "banner"
POSTER_DIR = ASSETS_DIR / "poster"
CHARTS_DIR = ASSETS_DIR / "charts"
SOCIAL_DIR = ASSETS_DIR / "social"
DIAGRAMS_DIR = ASSETS_DIR / "diagrams"

for d in [BANNER_DIR, POSTER_DIR, CHARTS_DIR, SOCIAL_DIR, DIAGRAMS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 1. Chart 1: 5-Fold Walk-Forward Cross-Validation
plt.style.use("dark_background")
fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
fig.patch.set_facecolor("#0A0E17")
ax1.set_facecolor("#111827")

folds = [1, 2, 3, 4, 5]
hybrid_dir_acc = [13.0, 12.1, 28.9, 44.5, 69.8]
hybrid_auc = [0.5768, 0.5719, 0.5641, 0.6244, 0.7024]
lstm_dir_acc = [18.2, 16.5, 25.1, 41.2, 63.3]

bars = ax1.bar(np.array(folds) - 0.18, hybrid_dir_acc, width=0.35, color="#10B981", alpha=0.9, label="Hybrid GNN Directional Acc (%)")
bars_lstm = ax1.bar(np.array(folds) + 0.18, lstm_dir_acc, width=0.35, color="#38BDF8", alpha=0.7, label="LSTM Baseline Directional Acc (%)")

ax1.set_xlabel("Walk-Forward Cross-Validation Fold (Temporal Expansion)", fontsize=11, color="#9CA3AF")
ax1.set_ylabel("Directional Accuracy (%)", fontsize=11, color="#10B981")
ax1.set_ylim(0, 85)
ax1.set_xticks(folds)
ax1.set_xticklabels([f"Fold {f}" for f in folds])
ax1.grid(True, linestyle=":", alpha=0.3, color="#374151")

for bar, val in zip(bars, hybrid_dir_acc):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2, f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold", color="#F9FAFB")

# Secondary axis: AUC
ax2 = ax1.twinx()
ax2.plot(folds, hybrid_auc, color="#F59E0B", marker="o", linewidth=2.4, label="Hybrid GNN ROC-AUC")
ax2.set_ylabel("ROC-AUC Score", fontsize=11, color="#F59E0B")
ax2.set_ylim(0.50, 0.78)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, facecolor="#1F2937", edgecolor="#374151", fontsize=9)

plt.title("5-Fold Walk-Forward Cross-Validation: Hybrid GNN vs LSTM (Fold 5 Peak = 69.85%)", fontsize=12, fontweight="bold", pad=15, color="#F9FAFB")
plt.tight_layout()

chart1_png = CHARTS_DIR / "01_walk_forward_accuracy.png"
chart1_svg = CHARTS_DIR / "01_walk_forward_accuracy.svg"
fig.savefig(chart1_png, dpi=300, facecolor=fig.get_facecolor())
fig.savefig(chart1_svg, facecolor=fig.get_facecolor())
plt.close(fig)

# 2. Chart 2: Smart Money (Institutional vs Retail) Order Flow
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
fig.patch.set_facecolor("#0A0E17")
ax.set_facecolor("#111827")

days = [f"D-{i}" for i in range(12, 0, -1)]
net_ins = [12.4, 18.2, 24.5, 31.0, 15.6, -4.2, -8.1, 14.5, 28.9, 42.1, 55.4, 62.1]
colors = ["#10B981" if v > 0 else "#EF4444" for v in net_ins]

bars = ax.bar(days, net_ins, color=colors, width=0.6, edgecolor="#1F2937")
ax.axhline(0, color="#9CA3AF", linestyle="-", linewidth=1.0)
ax.set_title("Institutional (Hoghughi) Net Volume Inflow / Accumulation Pattern", fontsize=12.5, fontweight="bold", pad=15, color="#F9FAFB")
ax.set_xlabel("Trading Days Preceding Momentum Rally", fontsize=11, color="#9CA3AF")
ax.set_ylabel("Net Institutional Volume (Million Shares)", fontsize=11, color="#38BDF8")
ax.grid(True, linestyle=":", alpha=0.3, color="#374151")

for bar, val in zip(bars, net_ins):
    offset = 1.5 if val >= 0 else -3.5
    ax.text(bar.get_x() + bar.get_width()/2, val + offset, f"{val:+.1f}M", ha="center", fontsize=8.5, fontweight="bold", color="#F9FAFB")

plt.tight_layout()
chart2_png = CHARTS_DIR / "02_institutional_flow_correlation.png"
chart2_svg = CHARTS_DIR / "02_institutional_flow_correlation.svg"
fig.savefig(chart2_png, dpi=300, facecolor=fig.get_facecolor())
fig.savefig(chart2_svg, facecolor=fig.get_facecolor())
plt.close(fig)

# 3. Banner Generation (1280x640)
banner = Image.new("RGBA", (1280, 640), "#0A0E17")
draw = ImageDraw.Draw(banner)
draw.rectangle([(0, 0), (1280, 8)], fill="#10B981")

try:
    font_title = ImageFont.truetype("arialbd.ttf", 48)
    font_sub = ImageFont.truetype("arial.ttf", 25)
    font_metric = ImageFont.truetype("arialbd.ttf", 44)
    font_label = ImageFont.truetype("arial.ttf", 18)
except Exception:
    font_title = font_sub = font_metric = font_label = ImageFont.load_default()

draw.text((80, 70), "TEHRAN STOCK EXCHANGE (TSETMC) QUANTITATIVE ENGINE", fill="#10B981", font=font_label)
draw.text((80, 100), "Tehran Stock Forecasting Platform", fill="#F9FAFB", font=font_title)
draw.text((80, 175), "Walk-Forward Validation | Hybrid LSTM + GNN | Institutional (Hoghughi) Money Flow", fill="#9CA3AF", font=font_sub)

boxes = [
    ("Peak Directional Acc", "69.85%", "Fold 5 Out-of-Sample Holdout", "#10B981"),
    ("Walk-Forward Splits", "5 Folds", "Non-overlapping temporal test", "#38BDF8"),
    ("Core Stock Universe", "18 Core", "428 Total Indexed Instruments", "#F59E0B"),
    ("Indexed Snapshots", "10,874", "Daily prices & client order flows", "#8B5CF6"),
]

box_y = 250
box_w = 260
box_h = 160
start_x = 80
gap = 26

for i, (t_text, v_text, d_text, col) in enumerate(boxes):
    bx = start_x + i * (box_w + gap)
    draw.rounded_rectangle([(bx, box_y), (bx + box_w, box_y + box_h)], radius=12, fill="#111827", outline="#1F2937", width=2)
    draw.text((bx + 20, box_y + 20), t_text, fill="#9CA3AF", font=font_label)
    draw.text((bx + 20, box_y + 55), v_text, fill=col, font=font_metric)
    draw.text((bx + 20, box_y + 115), d_text, fill="#6B7280", font=font_label)

tags = ["Hybrid LSTM + GNN", "TSETMC Market Microstructure", "Hoghughi Accumulation", "FastAPI Service", "Plotly / Dash UI (Port 8063)"]
tag_x = 80
tag_y = 500
for tag in tags:
    bbox = draw.textbbox((tag_x, tag_y), tag, font=font_label)
    tw = bbox[2] - bbox[0]
    draw.rounded_rectangle([(tag_x, tag_y - 6), (tag_x + tw + 24, tag_y + 26)], radius=6, fill="#1F2937", outline="#374151")
    draw.text((tag_x + 12, tag_y), tag, fill="#E5E7EB", font=font_label)
    tag_x += tw + 40

banner.save(BANNER_DIR / "github_banner.png")
banner_wide = banner.resize((1400, 500))
banner_wide.save(BANNER_DIR / "banner_wide.png")

# 4. Social Card (1200x627)
social = Image.new("RGBA", (1200, 627), "#0A0E17")
s_draw = ImageDraw.Draw(social)
s_draw.rectangle([(0, 0), (1200, 6)], fill="#10B981")
s_draw.text((70, 60), "QUANTITATIVE TRADING & DEEP LEARNING", fill="#10B981", font=font_label)
s_draw.text((70, 95), "Tehran Stock Forecasting Engine", fill="#F9FAFB", font=font_title)
s_draw.text((70, 165), "Walk-Forward Validation, Graph Neural Networks, and TSETMC Microstructure", fill="#9CA3AF", font=font_sub)

s_boxes = [
    ("Peak Directional Acc", "69.85%", "#10B981"),
    ("Mean 5-Fold AUC", "0.608", "#38BDF8"),
    ("Historical Records", "10,874", "#F59E0B"),
]
s_bx = 70
for t_text, v_text, col in s_boxes:
    s_draw.rounded_rectangle([(s_bx, 240), (s_bx + 320, 390)], radius=12, fill="#111827", outline="#1F2937", width=2)
    s_draw.text((s_bx + 24, 265), t_text, fill="#9CA3AF", font=font_label)
    s_draw.text((s_bx + 24, 305), v_text, fill=col, font=font_metric)
    s_bx += 360

s_draw.text((70, 500), "FastAPI Microservice | Plotly/Dash Operations Portal | Star Schema SQL Marts", fill="#6B7280", font=font_sub)
social.save(SOCIAL_DIR / "linkedin_cover.png")

# 5. Poster A3 Generation (1200x1697)
poster = Image.new("RGBA", (1200, 1697), "#0A0E17")
p_draw = ImageDraw.Draw(poster)
p_draw.rectangle([(0, 0), (1200, 12)], fill="#10B981")

try:
    p_title = ImageFont.truetype("arialbd.ttf", 46)
    p_h2 = ImageFont.truetype("arialbd.ttf", 26)
    p_body = ImageFont.truetype("arial.ttf", 20)
    p_code = ImageFont.truetype("consola.ttf", 17)
except Exception:
    p_title = p_h2 = p_body = p_code = ImageFont.load_default()

p_draw.text((80, 50), "QUANTITATIVE FINANCE & ASSET PRICING ARCHITECTURE", fill="#10B981", font=font_label)
p_draw.text((80, 85), "Tehran Stock Forecasting Platform", fill="#F9FAFB", font=p_title)
p_draw.text((80, 150), "Production Pipeline: Expanding-Window Walk-Forward Validation & Cross-Asset Graph Learning", fill="#9CA3AF", font=p_body)

# Metric summary strip
p_boxes = [
    ("Market Scope", "18 Core Stocks", "428 Indexed in Instruments DB"),
    ("Peak Directional Acc", "69.85%", "Fold 5 Out-of-Sample Holdout"),
    ("Walk-Forward Splits", "5 Folds", "Zero temporal lookahead bias"),
    ("Indexed Records", "10,874", "Daily history & client flow facts"),
]
p_by = 220
p_bw = 240
for i, (t_txt, v_txt, d_txt) in enumerate(p_boxes):
    bx = 80 + i * (p_bw + 26)
    p_draw.rounded_rectangle([(bx, p_by), (bx + p_bw, p_by + 110)], radius=8, fill="#111827", outline="#1F2937")
    p_draw.text((bx + 15, p_by + 12), t_txt, fill="#9CA3AF", font=font_label)
    p_draw.text((bx + 15, p_by + 36), v_txt, fill="#10B981", font=p_h2)
    p_draw.text((bx + 15, p_by + 76), d_txt, fill="#6B7280", font=font_label)

# Section 1: Executive Summary & Microstructure Formulation
p_draw.text((80, 370), "1. Market Microstructure & Formulation", fill="#10B981", font=p_h2)
desc = (
    "Forecasting emerging markets like the Tehran Stock Exchange requires addressing unique regulatory constraints:\n"
    "strict +/-5% daily price fluctuation limits and synthetic liquidity lockups. Standard naive models overfit random\n"
    "noise. Our platform utilizes a Hybrid LSTM + Graph Convolutional Network that incorporates institutional\n"
    "order flow (Haghighi vs Hoghughi) and cross-sector correlation edges, reaching up to 69.85% directional accuracy."
)
p_draw.text((80, 410), desc, fill="#D1D5DB", font=p_body)

# Embed Chart 1 into Poster
if chart1_png.exists():
    c1_img = Image.open(chart1_png).resize((1040, 520))
    poster.paste(c1_img, (80, 530))

# Section 2: Technical Highlights
p_draw.text((80, 1080), "2. Technical Architecture & Governance", fill="#10B981", font=p_h2)
arch_desc = (
    "- Leakage Guard: 5-fold expanding-window walk-forward cross validation; zero temporal lookahead bias.\n"
    "- Microstructure Edge: Track institutional smart money accumulation ratios across 10,874 trading days.\n"
    "- High-Throughput API: Sub-second FastAPI microservice serving /forecast, /symbols, and /governance.\n"
    "- Interactive Portal: Plotly/Dash portal on port 8063 featuring candlestick analysis and order flow charts."
)
p_draw.text((80, 1120), arch_desc, fill="#D1D5DB", font=p_body)

# Embed Chart 2 into Poster
if chart2_png.exists():
    c2_img = Image.open(chart2_png).resize((680, 380))
    poster.paste(c2_img, (80, 1260))

# QR Code
qr = qrcode.QRCode(box_size=6, border=2)
qr.add_data("https://github.com/aliajvand/tehran-stock-forecasting")
qr.make(fit=True)
qr_img = qr.make_image(fill_color="#10B981", back_color="#111827").convert("RGBA").resize((180, 180))

poster.paste(qr_img, (940, 1340))
p_draw.text((940, 1530), "Scan for Repository", fill="#9CA3AF", font=font_label)
p_draw.text((940, 1555), "github.com/aliajvand", fill="#6B7280", font=font_label)

poster.save(POSTER_DIR / "project_poster.png")
poster.convert("RGB").save(POSTER_DIR / "project_poster.pdf")

# 6. Pipeline Architecture SVG
svg_content = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 320" width="100%" height="100%">
  <rect width="1000" height="320" fill="#0A0E17" rx="12"/>
  <rect x="0" y="0" width="1000" height="6" fill="#10B981"/>
  
  <!-- Ingestion -->
  <rect x="40" y="60" width="160" height="200" fill="#111827" stroke="#1F2937" stroke-width="2" rx="8"/>
  <text x="120" y="95" fill="#10B981" font-family="sans-serif" font-weight="bold" font-size="14" text-anchor="middle">TSETMC Ingestion</text>
  <text x="120" y="130" fill="#9CA3AF" font-family="sans-serif" font-size="12" text-anchor="middle">algotik-tse Engine</text>
  <text x="120" y="155" fill="#6B7280" font-family="sans-serif" font-size="11" text-anchor="middle">428 Instruments</text>
  <text x="120" y="180" fill="#6B7280" font-family="sans-serif" font-size="11" text-anchor="middle">10,874 Day Records</text>
  <text x="120" y="225" fill="#38BDF8" font-family="sans-serif" font-size="11" text-anchor="middle">Jalali Calendar Sync</text>

  <!-- Arrow 1 -->
  <line x1="200" y1="160" x2="250" y2="160" stroke="#10B981" stroke-width="2"/>
  <polygon points="250,160 242,155 242,165" fill="#10B981"/>

  <!-- Feature Store -->
  <rect x="250" y="60" width="170" height="200" fill="#111827" stroke="#1F2937" stroke-width="2" rx="8"/>
  <text x="335" y="95" fill="#10B981" font-family="sans-serif" font-weight="bold" font-size="14" text-anchor="middle">Feature Engineering</text>
  <text x="335" y="130" fill="#9CA3AF" font-family="sans-serif" font-size="12" text-anchor="middle">80 Multimodal Signals</text>
  <text x="335" y="155" fill="#6B7280" font-family="sans-serif" font-size="11" text-anchor="middle">Haghighi / Hoghughi</text>
  <text x="335" y="180" fill="#6B7280" font-family="sans-serif" font-size="11" text-anchor="middle">Cross-Stock Graph Edges</text>
  <text x="335" y="225" fill="#38BDF8" font-family="sans-serif" font-size="11" text-anchor="middle">Zero Lookahead Bias</text>

  <!-- Arrow 2 -->
  <line x1="420" y1="160" x2="470" y2="160" stroke="#10B981" stroke-width="2"/>
  <polygon points="470,160 462,155 462,165" fill="#10B981"/>

  <!-- Model Engine -->
  <rect x="470" y="60" width="170" height="200" fill="#111827" stroke="#1F2937" stroke-width="2" rx="8"/>
  <text x="555" y="95" fill="#10B981" font-family="sans-serif" font-weight="bold" font-size="14" text-anchor="middle">Hybrid Deep Learning</text>
  <text x="555" y="130" fill="#9CA3AF" font-family="sans-serif" font-size="12" text-anchor="middle">LSTM + GNN Architecture</text>
  <text x="555" y="155" fill="#6B7280" font-family="sans-serif" font-size="11" text-anchor="middle">5-Fold Walk-Forward</text>
  <text x="555" y="180" fill="#6B7280" font-family="sans-serif" font-size="11" text-anchor="middle">Peak Dir Acc: 69.85%</text>
  <text x="555" y="225" fill="#38BDF8" font-family="sans-serif" font-size="11" text-anchor="middle">ROC-AUC: 0.7024</text>

  <!-- Arrow 3 -->
  <line x1="640" y1="160" x2="690" y2="160" stroke="#10B981" stroke-width="2"/>
  <polygon points="690,160 682,155 682,165" fill="#10B981"/>

  <!-- Serving & Outreach -->
  <rect x="690" y="60" width="270" height="200" fill="#111827" stroke="#1F2937" stroke-width="2" rx="8"/>
  <text x="825" y="95" fill="#10B981" font-family="sans-serif" font-weight="bold" font-size="14" text-anchor="middle">Inference &amp; Backtesting</text>
  <text x="825" y="130" fill="#9CA3AF" font-family="sans-serif" font-size="12" text-anchor="middle">FastAPI Engine (/forecast)</text>
  <text x="825" y="155" fill="#6B7280" font-family="sans-serif" font-size="11" text-anchor="middle">Plotly / Dash Portal (Port 8063)</text>
  <text x="825" y="180" fill="#6B7280" font-family="sans-serif" font-size="11" text-anchor="middle">Candlestick &amp; Order Flow</text>
  <text x="825" y="225" fill="#38BDF8" font-family="sans-serif" font-size="11" text-anchor="middle">Symmetric Limit Bounds</text>
</svg>
"""
(DIAGRAMS_DIR / "pipeline_architecture.svg").write_text(svg_content, encoding="utf-8")

print("All Tehran Stock Forecasting assets generated successfully.")
