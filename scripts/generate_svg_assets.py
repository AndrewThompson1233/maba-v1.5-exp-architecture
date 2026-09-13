import os
import re

assets_dir = "/workspaces/123123/maba-v1.5-exp-architecture/assets"
os.makedirs(assets_dir, exist_ok=True)

# Extract original icons from downloaded step file
source_step_file = "/home/codespace/.gemini/antigravity-cli/brain/b35eb27c-8ae6-4d09-adcb-1af5e113f746/.system_generated/steps/4419/content.md"
with open(source_step_file, "r") as f:
    text = f.read()

maba_icon_href = re.search(r'<image id=\"maba_icon\" [^>]*href=\"([^\"]+)\"/>', text).group(1)
qwen_icon_href = re.search(r'<image id=\"qwen_icon\" [^>]*href=\"([^\"]+)\"/>', text).group(1)
flash_icon_href = re.search(r'<image id=\"flash_icon\" [^>]*href=\"([^\"]+)\"/>', text).group(1)
cpm_icon_href = re.search(r'<image id=\"cpm_icon\" [^>]*href=\"([^\"]+)\"/>', text).group(1)

# 1. LOGO SVG
logo_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" width="128" height="128">
  <defs>
    <linearGradient id="logoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1e40af"/>
      <stop offset="50%" stop-color="#2563eb"/>
      <stop offset="100%" stop-color="#3b82f6"/>
    </linearGradient>
    <linearGradient id="badgeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#06b6d4"/>
      <stop offset="100%" stop-color="#3b82f6"/>
    </linearGradient>
  </defs>
  <rect width="128" height="128" rx="28" fill="url(#logoGrad)"/>
  <path d="M 64,24 L 102,94 L 26,94 Z" fill="none" stroke="#ffffff" stroke-width="7" stroke-linejoin="round"/>
  <circle cx="64" cy="71" r="7.5" fill="#ffffff"/>
  <path d="M 38,82 Q 64,50 90,82" fill="none" stroke="#93c5fd" stroke-width="4.5" stroke-linecap="round"/>
  <path d="M 46,88 Q 64,62 82,88" fill="none" stroke="#60a5fa" stroke-width="3" stroke-linecap="round"/>
  <rect x="76" y="14" width="38" height="18" rx="9" fill="url(#badgeGrad)"/>
  <text x="95" y="27" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="10.5" font-weight="800" fill="#ffffff" text-anchor="middle">v1.5</text>
</svg>
"""

with open(os.path.join(assets_dir, "logo.svg"), "w") as f:
    f.write(logo_svg.strip() + "\n")
print("1. logo.svg generated")

# GRADIENTS & ICONS FOR 5 MODELS
defs_svg = f"""  <defs>
    <linearGradient id="maba15Grad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#1d4ed8"/>
      <stop offset="100%" stop-color="#60a5fa"/>
    </linearGradient>
    <linearGradient id="mabaGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#4f46e5"/>
      <stop offset="100%" stop-color="#a5b4fc"/>
    </linearGradient>
    <linearGradient id="qwenGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#8b5cf6"/>
      <stop offset="100%" stop-color="#ddd6fe"/>
    </linearGradient>
    <linearGradient id="flashGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#06b6d4"/>
      <stop offset="100%" stop-color="#a5f3fc"/>
    </linearGradient>
    <linearGradient id="cpmGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#e11d48"/>
      <stop offset="100%" stop-color="#fda4af"/>
    </linearGradient>
    <image id="maba_icon" width="18" height="18" href="{maba_icon_href}"/>
    <image id="qwen_icon" width="18" height="18" href="{qwen_icon_href}"/>
    <image id="flash_icon" width="18" height="18" href="{flash_icon_href}"/>
    <image id="cpm_icon" width="18" height="18" href="{cpm_icon_href}"/>
  </defs>"""

# HELPER TO RENDER A 5-BAR PANEL
def render_panel(x, y, title, subtitle, y_ticks, bars, baseline_y=205.0):
    lines = [f'  <g transform="translate({x}, {y})">']
    lines.append(f'    <text x="0" y="16" font-family="-apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, sans-serif" font-size="14" font-weight="700" fill="#111827">{title}</text>')
    lines.append(f'    <text x="0" y="32" font-family="-apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, sans-serif" font-size="11" font-weight="400" fill="#6b7280">{subtitle}</text>')
    
    # y-ticks
    for tick_y, tick_label in y_ticks:
        lines.append(f'    <text x="26" y="{tick_y}" text-anchor="end" font-family="sans-serif" font-size="10" fill="#9ca3af">{tick_label}</text>')
    
    # baseline
    lines.append(f'    <line x1="32" y1="{baseline_y}" x2="220" y2="{baseline_y}" stroke="#e5e7eb" stroke-width="1"/>')
    
    # 5 bars
    # bar = (cx, top_y, val_str, label, grad_id, icon_id, text_color, is_bold)
    for cx, top_y, val_str, label, grad_id, icon_id, text_color, is_bold in bars:
        w = 18.0
        x0 = cx - w/2.0
        x1 = cx + w/2.0
        h = baseline_y - top_y
        
        if h > 4.0:
            lines.append(f'    <path d="M {x0:.1f},{baseline_y:.1f} L {x0:.1f},{top_y + 4.0:.1f} Q {x0:.1f},{top_y:.1f} {x0 + 4.0:.1f},{top_y:.1f} L {x1 - 4.0:.1f},{top_y:.1f} Q {x1:.1f},{top_y:.1f} {x1:.1f},{top_y + 4.0:.1f} L {x1:.1f},{baseline_y:.1f} Z" fill="url(#{grad_id})"/>')
            if icon_id:
                icon_y = top_y + 11.0
                if icon_y < baseline_y - 8.0:
                    lines.append(f'    <g transform="translate({cx:.1f}, {icon_y:.1f})"><circle cx="0" cy="0" r="8.5" fill="#ffffff" stroke="#cbd5e1" stroke-width="0.8"/><use href="#{icon_id}" x="-7.5" y="-7.5" width="15" height="15"/></g>')
        
        # Value text
        val_y = top_y - 6.0 if h > 4.0 else baseline_y - 6.0
        fweight = "700" if is_bold else "600"
        lines.append(f'    <text x="{cx:.1f}" y="{val_y:.1f}" text-anchor="middle" font-family="sans-serif" font-size="10.5" font-weight="{fweight}" fill="{text_color}">{val_str}</text>')
        
        # Bottom label
        lbl_weight = "700" if is_bold else "500"
        lines.append(f'    <text x="{cx:.1f}" y="221.0" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="{lbl_weight}" fill="{text_color}">{label}</text>')
        
    lines.append('  </g>')
    return "\n".join(lines)

# 2. ARCHITECTURE COMPARISON SVG
# 5 bars setup: cx = [46.0, 81.0, 116.0, 151.0, 186.0]
# Bar metadata: (grad_id, icon_id, label, text_color, is_bold)
models_meta = [
    ("maba15Grad", "maba_icon", "v1.5", "#1d4ed8", True),
    ("mabaGrad", "maba_icon", "v1.1", "#4f46e5", True),
    ("qwenGrad", "qwen_icon", "Qwen", "#8b5cf6", False),
    ("flashGrad", "flash_icon", "Flash", "#0891b2", False),
    ("cpmGrad", "cpm_icon", "CPM5", "#e11d48", False),
]

arch_panels = []

# Panel 1: Core Computation Ratio (0% to 100%)
# 0% -> 205, 100% -> 60. y = 205 - pct * 1.45
p1_y_ticks = [(78.5, "100%"), (114.7, "75%"), (151.0, "50%"), (187.2, "25%"), (208.5, "0%")]
p1_bars = [
    (46.0, 205.0 - 95.21 * 1.45, "95.2%", models_meta[0][2], models_meta[0][0], models_meta[0][1], models_meta[0][3], True),
    (81.0, 205.0 - 95.21 * 1.45, "95.2%", models_meta[1][2], models_meta[1][0], models_meta[1][1], models_meta[1][3], True),
    (116.0, 205.0 - 75.00 * 1.45, "75.0%", models_meta[2][2], models_meta[2][0], models_meta[2][1], models_meta[2][3], False),
    (151.0, 205.0 - 74.99 * 1.45, "75.0%", models_meta[3][2], models_meta[3][0], models_meta[3][1], models_meta[3][3], False),
    (186.0, 205.0 - 79.20 * 1.45, "79.2%", models_meta[4][2], models_meta[4][0], models_meta[4][1], models_meta[4][3], False),
]
arch_panels.append(render_panel(35, 25, "Core Computation Ratio", "% Weights in Core Math Layers", p1_y_ticks, p1_bars))

# Panel 2: Recurrence Engine Share (0% to 100%)
p2_y_ticks = [(78.5, "100%"), (114.7, "75%"), (151.0, "50%"), (187.2, "25%"), (208.5, "0%")]
p2_bars = [
    (46.0, 205.0 - 75.0 * 1.45, "75.0%", models_meta[0][2], models_meta[0][0], models_meta[0][1], models_meta[0][3], True),
    (81.0, 205.0 - 75.0 * 1.45, "75.0%", models_meta[1][2], models_meta[1][0], models_meta[1][1], models_meta[1][3], True),
    (116.0, 205.0 - 75.0 * 1.45, "75.0%", models_meta[2][2], models_meta[2][0], models_meta[2][1], models_meta[2][3], False),
    (151.0, 205.0 - 75.0 * 1.45, "75.0%", models_meta[3][2], models_meta[3][0], models_meta[3][1], models_meta[3][3], False),
    (186.0, 205.0, "0.0%", models_meta[4][2], models_meta[4][0], None, models_meta[4][3], False),
]
arch_panels.append(render_panel(285, 25, "Recurrence Engine Share", "% Constant-State Linear Layers", p2_y_ticks, p2_bars))

# Panel 3: Speculative Head (MTP) (0 to 3 tokens)
# 0 -> 205, 3 -> 60. 1 token = 48.33
p3_y_ticks = [(60.0, "3"), (108.3, "2"), (156.6, "1"), (208.5, "0")]
p3_bars = [
    (46.0, 205.0 - 2.0 * 48.33, "k=2", models_meta[0][2], models_meta[0][0], models_meta[0][1], models_meta[0][3], True),
    (81.0, 205.0 - 2.0 * 48.33, "k=2", models_meta[1][2], models_meta[1][0], models_meta[1][1], models_meta[1][3], True),
    (116.0, 205.0 - 2.0 * 48.33, "k=2", models_meta[2][2], models_meta[2][0], models_meta[2][1], models_meta[2][3], False),
    (151.0, 205.0 - 2.0 * 48.33, "k=2", models_meta[3][2], models_meta[3][0], models_meta[3][1], models_meta[3][3], False),
    (186.0, 205.0, "None", models_meta[4][2], models_meta[4][0], None, models_meta[4][3], False),
]
arch_panels.append(render_panel(535, 25, "Speculative Head (MTP)", "Native Prediction Horizon (Tokens)", p3_y_ticks, p3_bars))

# Panel 4: KV Projection Compression Ratio (0% to 100%)
p4_y_ticks = [(78.5, "100%"), (114.7, "75%"), (151.0, "50%"), (187.2, "25%"), (208.5, "0%")]
p4_bars = [
    (46.0, 205.0 - 80.0 * 1.45, "80%", models_meta[0][2], models_meta[0][0], models_meta[0][1], models_meta[0][3], True),
    (81.0, 205.0 - 50.0 * 1.45, "50%", models_meta[1][2], models_meta[1][0], models_meta[1][1], models_meta[1][3], True),
    (116.0, 205.0 - 50.0 * 1.45, "50%", models_meta[2][2], models_meta[2][0], models_meta[2][1], models_meta[2][3], False),
    (151.0, 205.0 - 75.0 * 1.45, "75%", models_meta[3][2], models_meta[3][0], models_meta[3][1], models_meta[3][3], False),
    (186.0, 205.0 - 50.0 * 1.45, "50%", models_meta[4][2], models_meta[4][0], models_meta[4][1], models_meta[4][3], False),
]
arch_panels.append(render_panel(785, 25, "KV Latent Compression", "% Key-Value Dimension Reduction", p4_y_ticks, p4_bars))

# Panel 5: KV-Cache (4k Context) (0 to 50 MB)
# 0 -> 205, 50 -> 60. 1 MB = 2.9 px
p5_y_ticks = [(60.0, "50"), (89.0, "40"), (118.0, "30"), (147.0, "20"), (176.0, "10"), (208.5, "0")]
p5_bars = [
    (46.0, 205.0 - 2.50 * 2.9, "2.5M", models_meta[0][2], models_meta[0][0], None, models_meta[0][3], True),
    (81.0, 205.0 - 10.00 * 2.9, "10.0M", models_meta[1][2], models_meta[1][0], models_meta[1][1], models_meta[1][3], True),
    (116.0, 205.0 - 10.00 * 2.9, "10.0M", models_meta[2][2], models_meta[2][0], models_meta[2][1], models_meta[2][3], False),
    (151.0, 205.0 - 2.50 * 2.9, "2.5M", models_meta[3][2], models_meta[3][0], None, models_meta[3][3], False),
    (186.0, 205.0 - 42.00 * 2.9, "42.0M", models_meta[4][2], models_meta[4][0], models_meta[4][1], models_meta[4][3], False),
]
arch_panels.append(render_panel(35, 305, "KV-Cache (4k Context)", "Runtime VRAM (MB, Lower is Better)", p5_y_ticks, p5_bars))

# Panel 6: KV-Cache Reduction (%) (0% to 100%)
p6_y_ticks = [(78.5, "100%"), (114.7, "75%"), (151.0, "50%"), (187.2, "25%"), (208.5, "0%")]
p6_bars = [
    (46.0, 205.0 - 94.0 * 1.45, "-94.0%", models_meta[0][2], models_meta[0][0], models_meta[0][1], models_meta[0][3], True),
    (81.0, 205.0 - 76.2 * 1.45, "-76.2%", models_meta[1][2], models_meta[1][0], models_meta[1][1], models_meta[1][3], True),
    (116.0, 205.0 - 76.2 * 1.45, "-76.2%", models_meta[2][2], models_meta[2][0], models_meta[2][1], models_meta[2][3], False),
    (151.0, 205.0 - 94.0 * 1.45, "-94.0%", models_meta[3][2], models_meta[3][0], models_meta[3][1], models_meta[3][3], False),
    (186.0, 205.0, "0.0%", models_meta[4][2], models_meta[4][0], None, models_meta[4][3], False),
]
arch_panels.append(render_panel(285, 305, "KV-Cache Reduction (%)", "VRAM Saved vs Pure Attention", p6_y_ticks, p6_bars))

# Panel 7: Active Cache @ 1M Context (Logarithmic Scale)
# 1MB = 205, 10MB = 170, 100MB = 135, 1GB = 100, 10GB = 65
p7_y_ticks = [(65.0, "10GB"), (100.0, "1GB"), (135.0, "100M"), (170.0, "10M"), (205.0, "1M")]
# log10 values: 2.5MB -> 0.40 * 35 = 14px -> y=191. 640MB -> 2.80 * 35 = 98px -> y=107. 2560MB -> 3.41 * 35 = 119px -> y=86. 10752MB -> 4.03 * 35 = 141px -> y=64.
p7_bars = [
    (46.0, 205.0 - 14.0, "2.5M", models_meta[0][2], models_meta[0][0], None, models_meta[0][3], True),
    (81.0, 205.0 - 119.0, "2.56G", models_meta[1][2], models_meta[1][0], models_meta[1][1], models_meta[1][3], True),
    (116.0, 205.0 - 119.0, "2.56G", models_meta[2][2], models_meta[2][0], models_meta[2][1], models_meta[2][3], False),
    (151.0, 205.0 - 98.0, "640M", models_meta[3][2], models_meta[3][0], models_meta[3][1], models_meta[3][3], False),
    (186.0, 205.0 - 141.0, "10.8G", models_meta[4][2], models_meta[4][0], models_meta[4][1], models_meta[4][3], False),
]
arch_panels.append(render_panel(535, 305, "Active Memory @ 1M", "Top-32 Blocks vs Dense (Log)", p7_y_ticks, p7_bars))

# Panel 8: Exact Parameters (M)
p8_y_ticks = [(60.0, "105M"), (96.2, "100M"), (132.5, "95M"), (168.7, "90M"), (208.5, "0M")]
# 0 -> 205, 105 -> 60. 1M = 1.38 px
p8_bars = [
    (46.0, 205.0 - 101.28 * 1.38, "101.3M", models_meta[0][2], models_meta[0][0], models_meta[0][1], models_meta[0][3], True),
    (81.0, 205.0 - 101.18 * 1.38, "101.2M", models_meta[1][2], models_meta[1][0], models_meta[1][1], models_meta[1][3], True),
    (116.0, 205.0 - 101.15 * 1.38, "101.2M", models_meta[2][2], models_meta[2][0], models_meta[2][1], models_meta[2][3], False),
    (151.0, 205.0 - 101.13 * 1.38, "101.1M", models_meta[3][2], models_meta[3][0], models_meta[3][1], models_meta[3][3], False),
    (186.0, 205.0 - 100.40 * 1.38, "100.4M", models_meta[4][2], models_meta[4][0], models_meta[4][1], models_meta[4][3], False),
]
arch_panels.append(render_panel(785, 305, "Exact Parameters (M)", "Total Model Parameters", p8_y_ticks, p8_bars))

# Bottom Legend
legend_svg = """  <g transform="translate(35, 600)">
    <g transform="translate(0.0, 6.0)"><circle cx="0" cy="0" r="9.5" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/><use href="#maba_icon" x="-8.5" y="-8.5" width="17" height="17"/></g>
    <text x="15" y="10" font-family="sans-serif" font-size="11.5" font-weight="700" fill="#1d4ed8">Maba v1.5-exp (DGDA+SA)</text>
    
    <g transform="translate(225.0, 6.0)"><circle cx="0" cy="0" r="9.5" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/><use href="#maba_icon" x="-8.5" y="-8.5" width="17" height="17"/></g>
    <text x="240" y="10" font-family="sans-serif" font-size="11.5" font-weight="700" fill="#4f46e5">Maba v1.1 (GDN-2+GQA)</text>
    
    <g transform="translate(440.0, 6.0)"><circle cx="0" cy="0" r="9.5" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/><use href="#qwen_icon" x="-8.5" y="-8.5" width="17" height="17"/></g>
    <text x="455" y="10" font-family="sans-serif" font-size="11.5" font-weight="500" fill="#4b5563">Qwen 3.8 (GDN+GQA)</text>
    
    <g transform="translate(635.0, 6.0)"><circle cx="0" cy="0" r="9.5" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/><use href="#flash_icon" x="-8.5" y="-8.5" width="17" height="17"/></g>
    <text x="650" y="10" font-family="sans-serif" font-size="11.5" font-weight="500" fill="#4b5563">Qwen 3.8 Flash Next (QSA)</text>
    
    <g transform="translate(855.0, 6.0)"><circle cx="0" cy="0" r="9.5" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/><use href="#cpm_icon" x="-8.5" y="-8.5" width="17" height="17"/></g>
    <text x="870" y="10" font-family="sans-serif" font-size="11.5" font-weight="500" fill="#4b5563">MiniCPM5 (Pure GQA)</text>
  </g>"""

arch_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1040 640" width="100%" height="640">
{defs_svg}
  <rect width="1040" height="640" fill="#ffffff"/>
{"\n".join(arch_panels)}
{legend_svg}
</svg>
"""

with open(os.path.join(assets_dir, "architecture_comparison.svg"), "w") as f:
    f.write(arch_svg.strip() + "\n")
print("2. architecture_comparison.svg generated")

# 3. SCALING COMPARISON SVG
scaling_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1040 640" width="100%" height="640">
  <defs>
    <linearGradient id="mabaGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#2563eb"/>
      <stop offset="100%" stop-color="#93c5fd"/>
    </linearGradient>
    <linearGradient id="qwenGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#8b5cf6"/>
      <stop offset="100%" stop-color="#ddd6fe"/>
    </linearGradient>
    <linearGradient id="museGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#0284c7"/>
      <stop offset="100%" stop-color="#bae6fd"/>
    </linearGradient>
    <linearGradient id="gemmaGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#ea580c"/>
      <stop offset="100%" stop-color="#fed7aa"/>
    </linearGradient>
    <linearGradient id="otherGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#64748b"/>
      <stop offset="100%" stop-color="#cbd5e1"/>
    </linearGradient>
  </defs>

  <rect width="1040" height="640" fill="#ffffff"/>

  <!-- Top Metric Cards (Row 1, y=25) -->
  <!-- Card 1: Core Parameter Ratio -->
  <g transform="translate(35, 25)">
    <rect width="220" height="215" rx="8" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1"/>
    <text x="16" y="24" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="13" font-weight="700" fill="#111827">Core Parameter Ratio (%)</text>
    <text x="16" y="40" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="10.5" fill="#64748b">Computation Weights vs Vocab</text>
    
    <!-- Maba-1B -->
    <text x="16" y="70" font-family="sans-serif" font-size="11" font-weight="700" fill="#2563eb">Maba v1.5-1B</text>
    <rect x="16" y="76" width="188" height="12" rx="3" fill="#e2e8f0"/>
    <rect x="16" y="76" width="183.8" height="12" rx="3" fill="url(#mabaGrad)"/>
    <text x="185" y="69" font-family="sans-serif" font-size="11" font-weight="700" fill="#2563eb" text-anchor="end">97.8%</text>

    <!-- Qwen3.5-0.8B -->
    <text x="16" y="104" font-family="sans-serif" font-size="11" fill="#475569">Qwen3.5-0.8B</text>
    <rect x="16" y="110" width="188" height="12" rx="3" fill="#e2e8f0"/>
    <rect x="16" y="110" width="128.2" height="12" rx="3" fill="url(#qwenGrad)"/>
    <text x="185" y="103" font-family="sans-serif" font-size="11" font-weight="600" fill="#475569" text-anchor="end">68.2%</text>

    <!-- Maba-30B -->
    <text x="16" y="138" font-family="sans-serif" font-size="11" font-weight="700" fill="#2563eb">Maba v1.5-30B</text>
    <rect x="16" y="144" width="188" height="12" rx="3" fill="#e2e8f0"/>
    <rect x="16" y="144" width="187.3" height="12" rx="3" fill="url(#mabaGrad)"/>
    <text x="185" y="137" font-family="sans-serif" font-size="11" font-weight="700" fill="#2563eb" text-anchor="end">99.6%</text>

    <!-- Gemma4-31B -->
    <text x="16" y="172" font-family="sans-serif" font-size="11" fill="#475569">Gemma4-31B</text>
    <rect x="16" y="178" width="188" height="12" rx="3" fill="#e2e8f0"/>
    <rect x="16" y="178" width="166.1" height="12" rx="3" fill="url(#gemmaGrad)"/>
    <text x="185" y="171" font-family="sans-serif" font-size="11" font-weight="600" fill="#475569" text-anchor="end">88.4%</text>
  </g>

  <!-- Card 2: Vocab Parameter Tax -->
  <g transform="translate(285, 25)">
    <rect width="220" height="215" rx="8" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1"/>
    <text x="16" y="24" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="13" font-weight="700" fill="#111827">Vocab Parameter Tax (%)</text>
    <text x="16" y="40" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="10.5" fill="#64748b">Overhead from Large Vocabularies</text>
    
    <!-- Maba-1B -->
    <text x="16" y="70" font-family="sans-serif" font-size="11" font-weight="700" fill="#16a34a">Maba v1.5-1B (Factorized)</text>
    <rect x="16" y="76" width="188" height="12" rx="3" fill="#e2e8f0"/>
    <rect x="16" y="76" width="10.3" height="12" rx="3" fill="#16a34a"/>
    <text x="185" y="69" font-family="sans-serif" font-size="11" font-weight="700" fill="#16a34a" text-anchor="end">1.74%</text>

    <!-- Qwen3.5-0.8B -->
    <text x="16" y="104" font-family="sans-serif" font-size="11" fill="#dc2626">Qwen3.5-0.8B (Direct 248k)</text>
    <rect x="16" y="110" width="188" height="12" rx="3" fill="#e2e8f0"/>
    <rect x="16" y="110" width="188.0" height="12" rx="3" fill="#ef4444"/>
    <text x="185" y="103" font-family="sans-serif" font-size="11" font-weight="700" fill="#dc2626" text-anchor="end">31.8%</text>

    <!-- Maba-30B -->
    <text x="16" y="138" font-family="sans-serif" font-size="11" font-weight="700" fill="#16a34a">Maba v1.5-30B (Factorized)</text>
    <rect x="16" y="144" width="188" height="12" rx="3" fill="#e2e8f0"/>
    <rect x="16" y="144" width="4.0" height="12" rx="3" fill="#16a34a"/>
    <text x="185" y="137" font-family="sans-serif" font-size="11" font-weight="700" fill="#16a34a" text-anchor="end">0.20%</text>

    <!-- Gemma4-31B -->
    <text x="16" y="172" font-family="sans-serif" font-size="11" fill="#dc2626">Gemma4-31B (Direct 256k)</text>
    <rect x="16" y="178" width="188" height="12" rx="3" fill="#e2e8f0"/>
    <rect x="16" y="178" width="68.9" height="12" rx="3" fill="#ef4444"/>
    <text x="185" y="171" font-family="sans-serif" font-size="11" font-weight="600" fill="#dc2626" text-anchor="end">11.6%</text>
  </g>

  <!-- Card 3: KV Cache Footprint (131k) -->
  <g transform="translate(535, 25)">
    <rect width="220" height="215" rx="8" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1"/>
    <text x="16" y="24" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="13" font-weight="700" fill="#111827">KV Cache Footprint (131k)</text>
    <text x="16" y="40" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="10.5" fill="#64748b">FP16 VRAM (Lower is Better)</text>
    
    <!-- Maba-7B -->
    <text x="16" y="70" font-family="sans-serif" font-size="11" font-weight="700" fill="#2563eb">Maba v1.5-7B</text>
    <rect x="16" y="76" width="188" height="12" rx="3" fill="#e2e8f0"/>
    <rect x="16" y="76" width="11.2" height="12" rx="3" fill="url(#mabaGrad)"/>
    <text x="185" y="69" font-family="sans-serif" font-size="11" font-weight="700" fill="#2563eb" text-anchor="end">1.15 GB</text>

    <!-- Qwen3.5-9B -->
    <text x="16" y="104" font-family="sans-serif" font-size="11" fill="#475569">Qwen3.5-9B</text>
    <rect x="16" y="110" width="188" height="12" rx="3" fill="#e2e8f0"/>
    <rect x="16" y="110" width="40.0" height="12" rx="3" fill="url(#qwenGrad)"/>
    <text x="185" y="103" font-family="sans-serif" font-size="11" font-weight="600" fill="#475569" text-anchor="end">4.10 GB</text>

    <!-- Maba-30B -->
    <text x="16" y="138" font-family="sans-serif" font-size="11" font-weight="700" fill="#2563eb">Maba v1.5-30B</text>
    <rect x="16" y="144" width="188" height="12" rx="3" fill="#e2e8f0"/>
    <rect x="16" y="144" width="17.1" height="12" rx="3" fill="url(#mabaGrad)"/>
    <text x="185" y="137" font-family="sans-serif" font-size="11" font-weight="700" fill="#2563eb" text-anchor="end">1.75 GB</text>

    <!-- Gemma4-31B -->
    <text x="16" y="172" font-family="sans-serif" font-size="11" fill="#475569">Gemma4-31B</text>
    <rect x="16" y="178" width="188" height="12" rx="3" fill="#e2e8f0"/>
    <rect x="16" y="178" width="188.0" height="12" rx="3" fill="url(#gemmaGrad)"/>
    <text x="185" y="171" font-family="sans-serif" font-size="11" font-weight="600" fill="#475569" text-anchor="end">28.31 GB</text>
  </g>

  <!-- Card 4: Architecture Layout -->
  <g transform="translate(785, 25)">
    <rect width="220" height="215" rx="8" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1"/>
    <text x="16" y="24" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="13" font-weight="700" fill="#111827">Maba Macro-Stack</text>
    <text x="16" y="40" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-size="10.5" fill="#64748b">Hybrid Sub-Quadratic Engine</text>
    
    <!-- Recurrence Engine -->
    <rect x="16" y="62" width="188" height="42" rx="6" fill="#eff6ff" stroke="#bfdbfe" stroke-width="1"/>
    <text x="26" y="80" font-family="sans-serif" font-size="11.5" font-weight="700" fill="#1d4ed8">75% DGDA Recurrence</text>
    <text x="26" y="95" font-family="sans-serif" font-size="10" fill="#3b82f6">Decoupled Key/Value Gates + O(1)</text>

    <!-- Sparse Attention -->
    <rect x="16" y="112" width="188" height="42" rx="6" fill="#f5f3ff" stroke="#ddd6fe" stroke-width="1"/>
    <text x="26" y="130" font-family="sans-serif" font-size="11.5" font-weight="700" fill="#6d28d9">25% MABA-SA Attention</text>
    <text x="26" y="145" font-family="sans-serif" font-size="10" fill="#8b5cf6">MLA + Top-32 Sparse + HCA</text>

    <!-- Drafter -->
    <rect x="16" y="162" width="188" height="34" rx="6" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1"/>
    <text x="26" y="183" font-family="sans-serif" font-size="11" font-weight="700" fill="#047857">Speculative Drafter: MTP (k=2)</text>
  </g>

  <!-- Bottom Comparison Table (y=265) -->
  <g transform="translate(35, 265)">
    <rect width="970" height="345" rx="8" fill="#f8fafc" stroke="#e2e8f0" stroke-width="1"/>
    
    <!-- Table Header -->
    <rect x="0" y="0" width="970" height="38" rx="8" fill="#f1f5f9"/>
    <text x="20" y="24" font-family="sans-serif" font-size="11.5" font-weight="700" fill="#334155">Model Specification</text>
    <text x="210" y="24" font-family="sans-serif" font-size="11.5" font-weight="700" fill="#334155">Total / Core Weights</text>
    <text x="380" y="24" font-family="sans-serif" font-size="11.5" font-weight="700" fill="#334155">Macro Topology</text>
    <text x="540" y="24" font-family="sans-serif" font-size="11.5" font-weight="700" fill="#334155">Vocab Tax</text>
    <text x="680" y="24" font-family="sans-serif" font-size="11.5" font-weight="700" fill="#334155">KV Cache (131k)</text>
    <text x="830" y="24" font-family="sans-serif" font-size="11.5" font-weight="700" fill="#334155">Active Memory (1M)</text>
    <line x1="0" y1="38" x2="970" y2="38" stroke="#cbd5e1" stroke-width="1"/>

    <!-- Row 1: Maba-30B -->
    <text x="20" y="68" font-family="sans-serif" font-size="11.5" font-weight="700" fill="#1d4ed8">Maba v1.5-30B (2026)</text>
    <text x="210" y="68" font-family="sans-serif" font-size="11.5" fill="#0f172a">29.08B / 28.97B (99.6%)</text>
    <text x="380" y="68" font-family="sans-serif" font-size="11.5" fill="#0f172a">3:1 (39 DGDA + 13 SA)</text>
    <text x="540" y="68" font-family="sans-serif" font-size="11.5" font-weight="600" fill="#16a34a">0.20% (Rank 768)</text>
    <text x="680" y="68" font-family="sans-serif" font-size="11.5" font-weight="700" fill="#1d4ed8">1.75 GB</text>
    <text x="830" y="68" font-family="sans-serif" font-size="11.5" font-weight="700" fill="#16a34a">3.5 MB (Top-64)</text>
    <line x1="20" y1="81" x2="950" y2="81" stroke="#e2e8f0" stroke-width="1"/>

    <!-- Row 2: Muse-Glimmer-30B -->
    <text x="20" y="103" font-family="sans-serif" font-size="11.5" font-weight="600" fill="#0f172a">Muse-Glimmer-30B (Meta 2026)</text>
    <text x="210" y="103" font-family="sans-serif" font-size="11.5" fill="#475569">29.60B / 26.90B (90.9%)</text>
    <text x="380" y="103" font-family="sans-serif" font-size="11.5" fill="#475569">3:1 (39 Local 2k + 13 Global)</text>
    <text x="540" y="103" font-family="sans-serif" font-size="11.5" fill="#dc2626">9.12% (Untied 202k)</text>
    <text x="680" y="103" font-family="sans-serif" font-size="11.5" fill="#475569">4.50 GB</text>
    <text x="830" y="103" font-family="sans-serif" font-size="11.5" fill="#dc2626">34.40 GB (Dense)</text>
    <line x1="20" y1="116" x2="950" y2="116" stroke="#e2e8f0" stroke-width="1"/>

    <!-- Row 3: Gemma4-31B -->
    <text x="20" y="138" font-family="sans-serif" font-size="11.5" font-weight="600" fill="#0f172a">Gemma4-31B (Google 2026)</text>
    <text x="210" y="138" font-family="sans-serif" font-size="11.5" fill="#475569">30.70B / 27.12B (88.4%)</text>
    <text x="380" y="138" font-family="sans-serif" font-size="11.5" fill="#475569">100% Full Quadratic Attention</text>
    <text x="540" y="138" font-family="sans-serif" font-size="11.5" fill="#dc2626">11.65% (Direct 256k)</text>
    <text x="680" y="138" font-family="sans-serif" font-size="11.5" fill="#475569">28.31 GB</text>
    <text x="830" y="138" font-family="sans-serif" font-size="11.5" fill="#dc2626">226.50 GB (Dense)</text>
    <line x1="20" y1="151" x2="950" y2="151" stroke="#e2e8f0" stroke-width="1"/>

    <!-- Row 4: Maba-7B -->
    <text x="20" y="173" font-family="sans-serif" font-size="11.5" font-weight="700" fill="#1d4ed8">Maba v1.5-7B (2026)</text>
    <text x="210" y="173" font-family="sans-serif" font-size="11.5" fill="#0f172a">7.14B / 7.08B (99.2%)</text>
    <text x="380" y="173" font-family="sans-serif" font-size="11.5" fill="#0f172a">3:1 (27 DGDA + 9 SA)</text>
    <text x="540" y="173" font-family="sans-serif" font-size="11.5" font-weight="600" fill="#16a34a">0.52% (Rank 512)</text>
    <text x="680" y="173" font-family="sans-serif" font-size="11.5" font-weight="700" fill="#1d4ed8">1.15 GB</text>
    <text x="830" y="173" font-family="sans-serif" font-size="11.5" font-weight="700" fill="#16a34a">2.5 MB (Top-48)</text>
    <line x1="20" y1="186" x2="950" y2="186" stroke="#e2e8f0" stroke-width="1"/>

    <!-- Row 5: Qwen3.5-9B -->
    <text x="20" y="208" font-family="sans-serif" font-size="11.5" font-weight="600" fill="#0f172a">Qwen3.5-9B (Alibaba 2026)</text>
    <text x="210" y="208" font-family="sans-serif" font-size="11.5" fill="#475569">8.80B / 7.78B (88.4%)</text>
    <text x="380" y="208" font-family="sans-serif" font-size="11.5" fill="#475569">3:1 (24 GDN + 8 GQA)</text>
    <text x="540" y="208" font-family="sans-serif" font-size="11.5" fill="#dc2626">11.56% (Direct 248k)</text>
    <text x="680" y="208" font-family="sans-serif" font-size="11.5" fill="#475569">4.10 GB</text>
    <text x="830" y="208" font-family="sans-serif" font-size="11.5" fill="#dc2626">32.80 GB (Dense)</text>
    <line x1="20" y1="221" x2="950" y2="221" stroke="#e2e8f0" stroke-width="1"/>

    <!-- Row 6: Maba-1B -->
    <text x="20" y="243" font-family="sans-serif" font-size="11.5" font-weight="700" fill="#1d4ed8">Maba v1.5-1B (2026)</text>
    <text x="210" y="243" font-family="sans-serif" font-size="11.5" fill="#0f172a">1.01B / 0.98B (97.8%)</text>
    <text x="380" y="243" font-family="sans-serif" font-size="11.5" fill="#0f172a">3:1 (15 DGDA + 5 SA)</text>
    <text x="540" y="243" font-family="sans-serif" font-size="11.5" font-weight="600" fill="#16a34a">1.74% (Rank 256)</text>
    <text x="680" y="243" font-family="sans-serif" font-size="11.5" font-weight="700" fill="#1d4ed8">320.0 MB</text>
    <text x="830" y="243" font-family="sans-serif" font-size="11.5" font-weight="700" fill="#16a34a">1.3 MB (Top-32)</text>
    <line x1="20" y1="256" x2="950" y2="256" stroke="#e2e8f0" stroke-width="1"/>

    <!-- Row 7: Qwen3.5-0.8B -->
    <text x="20" y="278" font-family="sans-serif" font-size="11.5" font-weight="600" fill="#0f172a">Qwen3.5-0.8B (Alibaba 2026)</text>
    <text x="210" y="278" font-family="sans-serif" font-size="11.5" fill="#475569">0.80B / 0.55B (68.2%)</text>
    <text x="380" y="278" font-family="sans-serif" font-size="11.5" fill="#475569">3:1 (18 GDN + 6 GQA)</text>
    <text x="540" y="278" font-family="sans-serif" font-size="11.5" fill="#dc2626">31.79% (Direct 248k)</text>
    <text x="680" y="278" font-family="sans-serif" font-size="11.5" fill="#475569">768.0 MB</text>
    <text x="830" y="278" font-family="sans-serif" font-size="11.5" fill="#dc2626">6.14 GB (Dense)</text>
  </g>
</svg>
"""

with open(os.path.join(assets_dir, "scaling_comparison.svg"), "w") as f:
    f.write(scaling_svg.strip() + "\n")
print("4. scaling_comparison.svg generated")
