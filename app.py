from flask import Flask, render_template_string, request, jsonify
import pickle, json, numpy as np, os

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as f:
    d = pickle.load(f)
    model, encoders, feature_cols = d['model'], d['encoders'], d['features']

with open(os.path.join(BASE_DIR, 'stats.json'), 'r') as f:
    stats = json.load(f)


BASE_STYLE = """
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=Fraunces:ital,wght@0,300;0,600;1,300&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{
  --bg:#07070e;--surface:#0f0f1a;--surface2:#171726;--border:#252538;
  --accent:#6c5ce7;--accent2:#e17055;--accent3:#00cec9;--accent4:#fdcb6e;
  --accent5:#fd79a8;--text:#dcdcf0;--muted:#5a5a7a;
  --resolved:#00cec9;--unresolved:#e17055;
}
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:'DM Mono',monospace;background:var(--bg);color:var(--text);min-height:100vh;}
body::before{content:'';position:fixed;inset:0;
  background:radial-gradient(ellipse 70% 50% at 15% 5%,rgba(108,92,231,.07) 0%,transparent 55%),
             radial-gradient(ellipse 50% 40% at 85% 95%,rgba(0,206,201,.05) 0%,transparent 50%);
  pointer-events:none;z-index:0;}
body::after{content:'';position:fixed;inset:0;
  background-image:linear-gradient(rgba(108,92,231,.03) 1px,transparent 1px),
                   linear-gradient(90deg,rgba(108,92,231,.03) 1px,transparent 1px);
  background-size:48px 48px;pointer-events:none;z-index:0;}

.wrap{max-width:1320px;margin:0 auto;padding:0 28px;position:relative;z-index:1;}

nav{border-bottom:1px solid var(--border);padding:20px 0;margin-bottom:36px;}
.nav-inner{display:flex;align-items:center;justify-content:space-between;}
.brand-name{font-family:'Syne',sans-serif;font-size:22px;font-weight:800;letter-spacing:-1px;}
.brand-name span{color:var(--accent);}
.brand-sub{font-size:9px;color:var(--muted);letter-spacing:3px;text-transform:uppercase;margin-top:2px;}
.nav-links{display:flex;gap:4px;}
.nav-link{padding:8px 16px;border-radius:8px;font-size:11px;font-family:'Syne',sans-serif;font-weight:600;
  letter-spacing:1px;text-transform:uppercase;text-decoration:none;color:var(--muted);transition:all .2s;}
.nav-link:hover{color:var(--text);background:var(--surface2);}
.nav-link.active{color:var(--accent);background:rgba(108,92,231,.12);border:1px solid rgba(108,92,231,.25);}
.nav-badge{display:flex;align-items:center;gap:7px;background:rgba(0,206,201,.08);
  border:1px solid rgba(0,206,201,.25);border-radius:100px;padding:7px 16px;font-size:10px;color:var(--accent3);}
.dot{width:7px;height:7px;border-radius:50%;background:var(--accent3);animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.4;transform:scale(.7);}}

.page-header{margin-bottom:32px;}
.page-title{font-family:'Fraunces',serif;font-size:32px;font-weight:600;margin-bottom:6px;}
.page-desc{font-size:12px;color:var(--muted);letter-spacing:.5px;line-height:1.6;}

.card{background:var(--surface);border:1px solid var(--border);border-radius:18px;padding:28px;}
.card-title{font-family:'Syne',sans-serif;font-size:12px;font-weight:700;letter-spacing:1.5px;
  text-transform:uppercase;color:var(--muted);margin-bottom:20px;display:flex;align-items:center;gap:8px;}
.card-title::before{content:'';width:3px;height:14px;background:var(--accent);border-radius:2px;flex-shrink:0;}

.stats-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:28px;}
.stat-card{background:var(--surface);border:1px solid var(--border);border-radius:18px;padding:24px;
  position:relative;overflow:hidden;transition:transform .2s,border-color .2s;}
.stat-card:hover{transform:translateY(-2px);border-color:var(--ca,var(--accent));}
.stat-top-bar{position:absolute;top:0;left:0;right:0;height:2px;background:var(--ca,var(--accent));}
.stat-label{font-size:9px;color:var(--muted);letter-spacing:2.5px;text-transform:uppercase;margin-bottom:14px;}
.stat-value{font-family:'Syne',sans-serif;font-size:38px;font-weight:800;color:var(--ca,var(--accent));line-height:1;}
.stat-sub{font-size:10px;color:var(--muted);margin-top:8px;}

.chart-nav-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:28px;}
.chart-nav-card{background:var(--surface);border:1px solid var(--border);border-radius:18px;padding:28px;
  text-decoration:none;color:var(--text);display:flex;flex-direction:column;gap:12px;
  transition:all .25s;position:relative;overflow:hidden;}
.chart-nav-card:hover{border-color:var(--cc,var(--accent));transform:translateY(-3px);
  box-shadow:0 12px 40px rgba(0,0,0,.4);}
.chart-nav-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:var(--cc,var(--accent));}
.cnc-icon{font-size:30px;margin-bottom:2px;}
.cnc-title{font-family:'Syne',sans-serif;font-size:14px;font-weight:700;}
.cnc-desc{font-size:11px;color:var(--muted);line-height:1.6;}
.cnc-arrow{font-size:18px;color:var(--cc,var(--accent));margin-top:auto;transition:transform .2s;}
.chart-nav-card:hover .cnc-arrow{transform:translateX(4px);}

.back-btn{display:inline-flex;align-items:center;gap:8px;padding:10px 20px;
  background:var(--surface2);border:1px solid var(--border);border-radius:10px;
  text-decoration:none;color:var(--muted);font-size:11px;letter-spacing:1px;
  transition:all .2s;margin-bottom:28px;}
.back-btn:hover{color:var(--text);border-color:var(--accent);}

.chart-wrap{position:relative;height:360px;width:100%;}
.chart-wrap.tall{height:460px;}
.chart-wrap.short{height:260px;}

.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px;}
.grid-1{margin-bottom:20px;}

/* Confusion matrix */
.cm-wrap{display:grid;grid-template-columns:32px 1fr 1fr;grid-template-rows:32px 1fr 1fr;gap:10px;height:280px;}
.cm-h{font-size:9px;color:var(--muted);text-align:center;display:flex;align-items:center;justify-content:center;letter-spacing:1px;}
.cm-rh{font-size:9px;color:var(--muted);display:flex;align-items:center;justify-content:center;
  writing-mode:vertical-lr;transform:rotate(180deg);letter-spacing:1px;}
.cm-cell{border-radius:14px;display:flex;flex-direction:column;align-items:center;justify-content:center;border:1px solid transparent;}
.cm-cell.good{background:rgba(0,206,201,.12);border-color:rgba(0,206,201,.3);}
.cm-cell.bad{background:rgba(225,112,85,.1);border-color:rgba(225,112,85,.25);}
.cm-n{font-family:'Syne',sans-serif;font-size:36px;font-weight:800;}
.cm-d{font-size:9px;color:var(--muted);letter-spacing:1px;margin-top:4px;}

/* Metric pills */
.metric-row{display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap;}
.metric-pill{background:var(--surface2);border:1px solid var(--border);border-radius:12px;
  padding:14px 20px;display:flex;flex-direction:column;gap:4px;flex:1;min-width:120px;}
.mp-val{font-family:'Syne',sans-serif;font-size:24px;font-weight:800;color:var(--accent3);}
.mp-label{font-size:9px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;}

/* Feature importance */
.feat-list{display:flex;flex-direction:column;gap:10px;}
.feat-row{display:flex;align-items:center;gap:14px;}
.feat-rank{font-family:'Syne',sans-serif;font-size:13px;font-weight:800;color:var(--accent);min-width:26px;}
.feat-name{font-size:11px;min-width:160px;}
.feat-track{flex:1;height:6px;background:var(--border);border-radius:3px;overflow:hidden;}
.feat-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--accent),var(--accent3));}
.feat-pct{font-size:11px;color:var(--accent);min-width:40px;text-align:right;}

/* Predict form */
.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px;}
.form-group{display:flex;flex-direction:column;gap:8px;}
label{font-size:9px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;}
select,input[type=number]{background:var(--surface2);border:1px solid var(--border);border-radius:10px;
  padding:12px 14px;color:var(--text);font-family:'DM Mono',monospace;font-size:12px;width:100%;outline:none;
  transition:border-color .2s;}
select:focus,input:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(108,92,231,.1);}
.predict-btn{width:100%;padding:16px;background:var(--accent);border:none;border-radius:12px;
  color:#fff;font-family:'Syne',sans-serif;font-size:13px;font-weight:700;letter-spacing:2px;
  text-transform:uppercase;cursor:pointer;transition:all .2s;}
.predict-btn:hover{background:#8075ff;transform:translateY(-1px);box-shadow:0 8px 24px rgba(108,92,231,.4);}
.predict-layout{display:grid;grid-template-columns:1fr 400px;gap:24px;align-items:start;}
.result-wrap{background:var(--surface);border:1px solid var(--border);border-radius:18px;padding:28px;position:sticky;top:24px;}
.result-ph{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:280px;
  text-align:center;color:var(--muted);gap:14px;}
.result-ph-icon{font-size:44px;opacity:.25;}
.result-ph-txt{font-size:11px;line-height:1.9;}
.result-body{display:none;}
.result-box{text-align:center;padding:28px 20px;border-radius:14px;margin-bottom:20px;}
.result-box.resolved{background:rgba(0,206,201,.08);border:1px solid rgba(0,206,201,.3);}
.result-box.unresolved{background:rgba(225,112,85,.08);border:1px solid rgba(225,112,85,.3);}
.rl{font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--muted);margin-bottom:10px;}
.rv{font-family:'Syne',sans-serif;font-size:30px;font-weight:800;margin-bottom:6px;}
.rv.resolved{color:var(--resolved);}
.rv.unresolved{color:var(--unresolved);}
.rc{font-size:12px;color:var(--muted);}
.conf-wrap{margin:18px 0;}
.conf-lbl{display:flex;justify-content:space-between;font-size:10px;color:var(--muted);margin-bottom:7px;}
.conf-track{height:8px;background:var(--border);border-radius:4px;overflow:hidden;}
.conf-fill{height:100%;border-radius:4px;transition:width .8s cubic-bezier(.22,1,.36,1);}
.meta-list{display:flex;flex-direction:column;}
.meta-row{display:flex;justify-content:space-between;padding:9px 0;border-bottom:1px solid var(--border);}
.meta-row:last-child{border-bottom:none;}
.mk{font-size:10px;color:var(--muted);letter-spacing:1px;text-transform:uppercase;}
.mv{font-size:11px;color:var(--text);max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}

/* Insight note box */
.insight{background:rgba(108,92,231,.07);border:1px solid rgba(108,92,231,.2);border-radius:12px;
  padding:14px 18px;font-size:11px;color:var(--muted);line-height:1.7;margin-bottom:20px;}
.insight strong{color:var(--accent);}

@media(max-width:1000px){
  .stats-grid,.chart-nav-grid{grid-template-columns:1fr 1fr;}
  .grid-2{grid-template-columns:1fr;}
  .predict-layout{grid-template-columns:1fr;}
  .result-wrap{position:static;}
}
</style>"""

def nav(active):
    links = ''.join(
        f'<a href="{h}" class="nav-link{" active" if h==active else ""}">{l}</a>'
        for h,l in [('/', 'Dashboard'), ('/predict', 'Predict')]
    )
    return f"""<nav><div class="wrap nav-inner">
      <div>
        <div class="brand-name">Complaint<span>IQ</span></div>
        <div class="brand-sub">Resolution Intelligence Platform</div>
      </div>
      <div class="nav-links">{links}</div>
      <div class="nav-badge"><div class="dot"></div>Model Live · {stats['accuracy']}% Accuracy</div>
    </div></nav>"""

CHART_CFG = """
Chart.defaults.color='#5a5a7a';
Chart.defaults.font.family="'DM Mono',monospace";
Chart.defaults.borderColor='#252538';
const AX={
  x:{ticks:{color:'#5a5a7a'},grid:{color:'rgba(255,255,255,.04)'}},
  y:{ticks:{color:'#5a5a7a'},grid:{color:'rgba(255,255,255,.04)'}}
};
const LEG={labels:{color:'#5a5a7a',font:{family:"'DM Mono',monospace",size:11}}};
"""


@app.route('/')
def dashboard():
    s = stats
    res_pct   = round(s['resolved_count'] / s['total_complaints'] * 100, 1)
    unres_pct = round(100 - res_pct, 1)

    pages = [
        ('/charts/product-resolution', 'var(--accent)',  'rgba(108,92,231,.06)',  '📦',
         'Product Resolution Analysis',
         'Resolved vs Unresolved counts per product. Shows which products have the most unresolved complaints.'),
        ('/charts/monthly-trend',      'var(--accent3)', 'rgba(0,206,201,.06)',   '📈',
         'Monthly Complaint Trend',
         'Complaint volume across 12 months plus quarterly totals. Identifies seasonal spikes.'),
        ('/charts/submission-channel', 'var(--accent4)', 'rgba(253,203,110,.06)', '📡',
         'Submission Channel Breakdown',
         'How customers file complaints — web, phone, mail etc. One clear horizontal bar, no duplicates.'),
        ('/charts/top-companies',      'var(--accent2)', 'rgba(225,112,85,.06)',  '🏢',
         'Company Deep-Dive',
         'Three distinct views: complaint volume, timely response rate, and consumer dispute rate per company.'),
        ('/charts/state-distribution', 'var(--accent5)', 'rgba(253,121,168,.06)', '🗺️',
         'Geographic Distribution',
         'Complaint volume by US state. One focused bar chart — no redundant donut or radar.'),
        ('/charts/model-performance',  'var(--accent3)', 'rgba(0,206,201,.06)',   '🤖',
         'Model Performance Report',
         'Confusion matrix, Precision/Recall/F1 grouped comparison, and feature importance ranking.'),
    ]

    cards = ''
    for href,cc,bg,icon,title,desc in pages:
        cards += f"""<a href="{href}" class="chart-nav-card" style="--cc:{cc};--cc-bg:{bg}">
          <div class="cnc-icon">{icon}</div>
          <div class="cnc-title">{title}</div>
          <div class="cnc-desc">{desc}</div>
          <div class="cnc-arrow">→</div></a>"""

    return f"""<!DOCTYPE html><html lang="en"><head>
    <meta charset="UTF-8"><title>Dashboard — ComplaintIQ</title>{BASE_STYLE}</head><body>
    {nav('/')}
    <div class="wrap">
      <div class="page-header">
        <div class="page-title">Dashboard</div>
        <div class="page-desc">Overview of {s['total_complaints']:,} consumer complaints. Click any card below to explore that analysis.</div>
      </div>
      <div class="stats-grid">
        <div class="stat-card" style="--ca:var(--accent)"><div class="stat-top-bar"></div>
          <div class="stat-label">Total Complaints</div><div class="stat-value">{s['total_complaints']:,}</div>
          <div class="stat-sub">Records analysed</div></div>
        <div class="stat-card" style="--ca:var(--resolved)"><div class="stat-top-bar"></div>
          <div class="stat-label">Resolved</div><div class="stat-value">{s['resolved_count']:,}</div>
          <div class="stat-sub">{res_pct}% of all complaints</div></div>
        <div class="stat-card" style="--ca:var(--unresolved)"><div class="stat-top-bar"></div>
          <div class="stat-label">Unresolved</div><div class="stat-value">{s['unresolved_count']:,}</div>
          <div class="stat-sub">{unres_pct}% need attention</div></div>
        <div class="stat-card" style="--ca:var(--accent4)"><div class="stat-top-bar"></div>
          <div class="stat-label">Model Accuracy</div><div class="stat-value">{s['accuracy']}%</div>
          <div class="stat-sub">Random Forest · 200 trees</div></div>
      </div>
      <div style="font-family:'Syne',sans-serif;font-size:10px;color:var(--muted);letter-spacing:2px;
        text-transform:uppercase;margin-bottom:14px;">ANALYSIS PAGES</div>
      <div class="chart-nav-grid">{cards}</div>
    </div></body></html>"""

# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — PRODUCT RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/charts/product-resolution')
def chart_product():
    s = stats
    labels   = json.dumps(s['prod_labels'])
    resolved = json.dumps(s['prod_resolved'])
    unresolved = json.dumps(s['prod_unresolved'])
    rates = []
    for i in range(len(s['prod_labels'])):
        total = s['prod_resolved'][i] + s['prod_unresolved'][i]
        rates.append(round(s['prod_resolved'][i] / total * 100, 1) if total else 0)

    return f"""<!DOCTYPE html><html lang="en"><head>
    <meta charset="UTF-8"><title>Product Resolution — ComplaintIQ</title>{BASE_STYLE}</head><body>
    {nav('')}<div class="wrap">
    <a href="/" class="back-btn">← Back to Dashboard</a>
    <div class="page-header">
      <div class="page-title">Product Resolution Analysis</div>
      <div class="page-desc">Two charts telling different stories: how many complaints exist per product, and what percentage actually get resolved.</div>
    </div>
    <div class="insight"><strong>How to read this:</strong> The stacked bar shows raw counts (how big the problem is).
      The rate bar shows quality (what % are resolved). A product can have few complaints but low resolution — that's also a problem.</div>
    <div class="grid-1"><div class="card">
      <div class="card-title">Resolved vs Unresolved Count by Product (Stacked Bar)</div>
      <div class="chart-wrap tall"><canvas id="c1"></canvas></div>
    </div></div>
    <div class="grid-1"><div class="card">
      <div class="card-title">Resolution Rate % per Product — colour coded: green ≥80%, yellow ≥60%, red &lt;60%</div>
      <div class="chart-wrap"><canvas id="c2"></canvas></div>
    </div></div>
    <script>{CHART_CFG}
    new Chart(document.getElementById('c1'),{{type:'bar',
      data:{{labels:{labels},datasets:[
        {{label:'Resolved',data:{resolved},backgroundColor:'rgba(0,206,201,.75)',borderRadius:4}},
        {{label:'Unresolved',data:{unresolved},backgroundColor:'rgba(225,112,85,.75)',borderRadius:4}}
      ]}},
      options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{...LEG}}}},
        scales:{{x:{{...AX.x,stacked:true}},y:{{...AX.y,stacked:true}}}}}}
    }});
    const RATES={json.dumps(rates)};
    new Chart(document.getElementById('c2'),{{type:'bar',
      data:{{labels:{labels},datasets:[{{label:'Resolution Rate (%)',data:RATES,
        backgroundColor:RATES.map(r=>r>=80?'rgba(0,206,201,.75)':r>=60?'rgba(253,203,110,.75)':'rgba(225,112,85,.75)'),
        borderRadius:6}}]}},
      options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},
        tooltip:{{callbacks:{{label:ctx=>ctx.raw+'%'}}}}}},
        scales:{{x:{{...AX.x}},y:{{...AX.y,max:100,ticks:{{callback:v=>v+'%'}}}}}}}}
    }});
    </script></div></body></html>"""

# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — MONTHLY TREND
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/charts/monthly-trend')
def chart_monthly():
    s = stats
    monthly = s['monthly_trend_data']
    peak_idx = monthly.index(max(monthly))
    peak_month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][peak_idx]
    quarters = [
        sum(monthly[0:3]), sum(monthly[3:6]),
        sum(monthly[6:9]), sum(monthly[9:12])
    ]

    return f"""<!DOCTYPE html><html lang="en"><head>
    <meta charset="UTF-8"><title>Monthly Trend — ComplaintIQ</title>{BASE_STYLE}</head><body>
    {nav('')}<div class="wrap">
    <a href="/" class="back-btn">← Back to Dashboard</a>
    <div class="page-header">
      <div class="page-title">Monthly Complaint Trend</div>
      <div class="page-desc">Peak month: <strong style="color:var(--accent4)">{peak_month}</strong> with {max(monthly):,} complaints.
        Two charts: month-by-month detail and quarterly rollup — different levels of granularity, not the same view.</div>
    </div>
    <div class="grid-1"><div class="card">
      <div class="card-title">Complaint Volume by Month (Area Line) — peak month highlighted</div>
      <div class="chart-wrap tall"><canvas id="c1"></canvas></div>
    </div></div>
    <div class="grid-1"><div class="card">
      <div class="card-title">Quarterly Total — Q1 through Q4</div>
      <div class="chart-wrap short"><canvas id="c2"></canvas></div>
    </div></div>
    <script>{CHART_CFG}
    const M={json.dumps(monthly)};
    const ML=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    const peak=Math.max(...M);
    new Chart(document.getElementById('c1'),{{type:'line',
      data:{{labels:ML,datasets:[{{label:'Complaints',data:M,
        borderColor:'#6c5ce7',backgroundColor:'rgba(108,92,231,.12)',
        fill:true,tension:0.45,borderWidth:2.5,
        pointBackgroundColor:M.map(v=>v===peak?'#fdcb6e':'#6c5ce7'),
        pointRadius:M.map(v=>v===peak?8:4),pointHoverRadius:8}}]}},
      options:{{responsive:true,maintainAspectRatio:false,
        plugins:{{legend:{{display:false}},
          tooltip:{{callbacks:{{title:ctx=>ML[ctx[0].dataIndex],label:ctx=>ctx.raw+' complaints'}}}}}},
        scales:{{x:{{...AX.x}},y:{{...AX.y}}}}}}
    }});
    new Chart(document.getElementById('c2'),{{type:'bar',
      data:{{labels:['Q1 Jan-Mar','Q2 Apr-Jun','Q3 Jul-Sep','Q4 Oct-Dec'],
        datasets:[{{label:'Total',data:{json.dumps(quarters)},
          backgroundColor:['rgba(108,92,231,.7)','rgba(0,206,201,.7)','rgba(253,203,110,.7)','rgba(225,112,85,.7)'],
          borderRadius:10}}]}},
      options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},
        scales:{{x:{{...AX.x}},y:{{...AX.y}}}}}}
    }});
    </script></div></body></html>"""

# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 — SUBMISSION CHANNEL
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/charts/submission-channel')
def chart_channel():
    s = stats
    return f"""<!DOCTYPE html><html lang="en"><head>
    <meta charset="UTF-8"><title>Submission Channel — ComplaintIQ</title>{BASE_STYLE}</head><body>
    {nav('')}<div class="wrap">
    <a href="/" class="back-btn">← Back to Dashboard</a>
    <div class="page-header">
      <div class="page-title">Submission Channel Breakdown</div>
      <div class="page-desc">How customers choose to file complaints. One horizontal bar — the clearest way to compare channels.</div>
    </div>
    <div class="insight"><strong>Why this matters:</strong> If 80% of complaints come via Web,
      the company's online complaint portal needs to be the most reliable. This chart directly informs where to invest in infrastructure.</div>
    <div class="grid-1"><div class="card">
      <div class="card-title">Complaints by Submission Channel</div>
      <div class="chart-wrap"><canvas id="c1"></canvas></div>
    </div></div>
    <script>{CHART_CFG}
    new Chart(document.getElementById('c1'),{{type:'bar',
      data:{{labels:{json.dumps(s['sub_labels'])},
        datasets:[{{label:'Complaints',data:{json.dumps(s['sub_data'])},
          backgroundColor:['#6c5ce7','#00cec9','#fdcb6e','#e17055','#fd79a8','#a29bfe'].slice(0,{len(s['sub_labels'])}),
          borderRadius:8}}]}},
      options:{{indexAxis:'y',responsive:true,maintainAspectRatio:false,
        plugins:{{legend:{{display:false}}}},
        scales:{{
          x:{{...AX.x,ticks:{{...AX.x.ticks}}}},
          y:{{ticks:{{color:'#8a8aaa',font:{{size:12}}}},grid:{{display:false}}}}
        }}}}
    }});
    </script></div></body></html>"""

# ─────────────────────────────────────────────────────────────────────────────
# CHART 4 — COMPANY DEEP-DIVE (3 genuinely different views)
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/charts/top-companies')
def chart_companies():
    s = stats
    COLORS = ['#6c5ce7','#00cec9','#fdcb6e','#e17055','#fd79a8','#a29bfe','#55efc4','#fab1a0','#74b9ff','#81ecec']
    colors_js = json.dumps(COLORS[:len(s['company_labels'])])

    return f"""<!DOCTYPE html><html lang="en"><head>
    <meta charset="UTF-8"><title>Company Deep-Dive — ComplaintIQ</title>{BASE_STYLE}</head><body>
    {nav('')}<div class="wrap">
    <a href="/" class="back-btn">← Back to Dashboard</a>
    <div class="page-header">
      <div class="page-title">Company Deep-Dive</div>
      <div class="page-desc">Three genuinely different questions about the same companies —
        how many complaints, how often they respond on time, and how often consumers push back.</div>
    </div>
    <div class="insight">
      <strong>Chart 1</strong> — Volume: which company gets the most complaints.<br>
      <strong>Chart 2</strong> — Timely Response Rate: does the company respond within the required timeframe? Higher = better.<br>
      <strong>Chart 3</strong> — Consumer Dispute Rate: after the company responds, how often does the consumer still disagree? Higher = worse.
    </div>
    <div class="grid-1"><div class="card">
      <div class="card-title">Complaint Volume — Top 10 Companies</div>
      <div class="chart-wrap"><canvas id="c1"></canvas></div>
    </div></div>
    <div class="grid-2">
      <div class="card">
        <div class="card-title">Timely Response Rate % — higher is better</div>
        <div class="chart-wrap"><canvas id="c2"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title">Consumer Dispute Rate % — lower is better</div>
        <div class="chart-wrap"><canvas id="c3"></canvas></div>
      </div>
    </div>
    <script>{CHART_CFG}
    const CL={json.dumps(s['company_labels'])};
    const CD={json.dumps(s['company_data'])};
    const TL={json.dumps(s['timely_labels'])};
    const TD={json.dumps(s['timely_data'])};
    const DL={json.dumps(s['disputed_labels'])};
    const DD={json.dumps(s['disputed_data'])};
    const COLORS={colors_js};

    new Chart(document.getElementById('c1'),{{type:'bar',
      data:{{labels:CL,datasets:[{{label:'Complaints',data:CD,
        backgroundColor:CL.map((_,i)=>i===0?'rgba(253,203,110,.85)':COLORS[i]+'bb'),borderRadius:6}}]}},
      options:{{indexAxis:'y',responsive:true,maintainAspectRatio:false,
        plugins:{{legend:{{display:false}}}},
        scales:{{x:{{...AX.x}},y:{{ticks:{{color:'#8a8aaa',font:{{size:11}}}},grid:{{display:false}}}}}}}}
    }});

    new Chart(document.getElementById('c2'),{{type:'bar',
      data:{{labels:TL,datasets:[{{label:'Timely Response %',data:TD,
        backgroundColor:TD.map(v=>v>=90?'rgba(0,206,201,.8)':v>=75?'rgba(253,203,110,.8)':'rgba(225,112,85,.8)'),
        borderRadius:6}}]}},
      options:{{indexAxis:'y',responsive:true,maintainAspectRatio:false,
        plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:ctx=>ctx.raw+'%'}}}}}},
        scales:{{x:{{...AX.x,max:100,ticks:{{callback:v=>v+'%'}}}},
          y:{{ticks:{{color:'#8a8aaa',font:{{size:10}}}},grid:{{display:false}}}}}}}}
    }});

    new Chart(document.getElementById('c3'),{{type:'bar',
      data:{{labels:DL,datasets:[{{label:'Dispute Rate %',data:DD,
        backgroundColor:DD.map(v=>v<=10?'rgba(0,206,201,.8)':v<=25?'rgba(253,203,110,.8)':'rgba(225,112,85,.8)'),
        borderRadius:6}}]}},
      options:{{indexAxis:'y',responsive:true,maintainAspectRatio:false,
        plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:ctx=>ctx.raw+'%'}}}}}},
        scales:{{x:{{...AX.x,ticks:{{callback:v=>v+'%'}}}},
          y:{{ticks:{{color:'#8a8aaa',font:{{size:10}}}},grid:{{display:false}}}}}}}}
    }});
    </script></div></body></html>"""

# ─────────────────────────────────────────────────────────────────────────────
# CHART 5 — STATE DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/charts/state-distribution')
def chart_states():
    s = stats
    return f"""<!DOCTYPE html><html lang="en"><head>
    <meta charset="UTF-8"><title>State Distribution — ComplaintIQ</title>{BASE_STYLE}</head><body>
    {nav('')}<div class="wrap">
    <a href="/" class="back-btn">← Back to Dashboard</a>
    <div class="page-header">
      <div class="page-title">Geographic Distribution</div>
      <div class="page-desc">Which US states generate the most complaints. One focused bar chart — clear, no redundant views.</div>
    </div>
    <div class="insight"><strong>Why this matters:</strong> High complaint states may correlate with population size,
      but also with consumer awareness or company presence. States with unexpectedly high counts
      warrant further investigation.</div>
    <div class="grid-1"><div class="card">
      <div class="card-title">Top 10 States by Complaint Volume</div>
      <div class="chart-wrap tall"><canvas id="c1"></canvas></div>
    </div></div>
    <script>{CHART_CFG}
    const COLORS=['#6c5ce7','#00cec9','#fdcb6e','#e17055','#fd79a8','#a29bfe','#55efc4','#fab1a0','#74b9ff','#81ecec'];
    new Chart(document.getElementById('c1'),{{type:'bar',
      data:{{labels:{json.dumps(s['state_labels'])},
        datasets:[{{label:'Complaints',data:{json.dumps(s['state_data'])},
          backgroundColor:{json.dumps(s['state_labels'])}.map((_,i)=>COLORS[i]+'cc'),borderRadius:8}}]}},
      options:{{responsive:true,maintainAspectRatio:false,
        plugins:{{legend:{{display:false}}}},
        scales:{{x:{{...AX.x}},y:{{...AX.y}}}}}}
    }});
    </script></div></body></html>"""

# ─────────────────────────────────────────────────────────────────────────────
# CHART 6 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/charts/model-performance')
def chart_model():
    s = stats
    cm = s['confusion_matrix']
    cr = s['classification_report']
    fi = s['feature_importance']
    fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    max_fi = fi_sorted[0][1] if fi_sorted else 1

    feat_rows = ''.join(f"""
      <div class="feat-row">
        <div class="feat-rank">#{i+1}</div>
        <div class="feat-name">{feat.replace('_',' ')}</div>
        <div class="feat-track"><div class="feat-fill" style="width:{round(imp/max_fi*100,1)}%"></div></div>
        <div class="feat-pct">{round(imp*100,1)}%</div>
      </div>""" for i,(feat,imp) in enumerate(fi_sorted))

    res   = cr.get('Resolved', {})
    unres = cr.get('Unresolved', {})
    macro = cr.get('macro avg', {})

    return f"""<!DOCTYPE html><html lang="en"><head>
    <meta charset="UTF-8"><title>Model Performance — ComplaintIQ</title>{BASE_STYLE}</head><body>
    {nav('')}<div class="wrap">
    <a href="/" class="back-btn">← Back to Dashboard</a>
    <div class="page-header">
      <div class="page-title">Model Performance Report</div>
      <div class="page-desc">Evaluation of the Random Forest classifier. Three distinct views: error breakdown, metric comparison, and feature importance.</div>
    </div>
    <div class="metric-row">
      <div class="metric-pill"><div class="mp-val">{s['accuracy']}%</div><div class="mp-label">Accuracy</div></div>
      <div class="metric-pill"><div class="mp-val">{round(res.get('precision',0),3)}</div><div class="mp-label">Precision (Resolved)</div></div>
      <div class="metric-pill"><div class="mp-val">{round(res.get('recall',0),3)}</div><div class="mp-label">Recall (Resolved)</div></div>
      <div class="metric-pill"><div class="mp-val">{round(res.get('f1-score',0),3)}</div><div class="mp-label">F1 (Resolved)</div></div>
      <div class="metric-pill"><div class="mp-val">{round(macro.get('f1-score',0),3)}</div><div class="mp-label">Macro F1</div></div>
    </div>
    <div class="grid-2">
      <div class="card">
        <div class="card-title">Confusion Matrix — what types of errors the model makes</div>
        <div class="cm-wrap">
          <div></div><div class="cm-h">PRED RESOLVED</div><div class="cm-h">PRED UNRESOLVED</div>
          <div class="cm-rh">ACT. RESOLVED</div>
          <div class="cm-cell good"><div class="cm-n" style="color:var(--resolved)">{cm[0][0]}</div><div class="cm-d">TRUE POS ✓</div></div>
          <div class="cm-cell bad"><div class="cm-n" style="color:var(--unresolved)">{cm[0][1]}</div><div class="cm-d">FALSE NEG ✗</div></div>
          <div class="cm-rh">ACT. UNRESOLVED</div>
          <div class="cm-cell bad"><div class="cm-n" style="color:var(--unresolved)">{cm[1][0]}</div><div class="cm-d">FALSE POS ✗</div></div>
          <div class="cm-cell good"><div class="cm-n" style="color:var(--resolved)">{cm[1][1]}</div><div class="cm-d">TRUE NEG ✓</div></div>
        </div>
      </div>
      <div class="card">
        <div class="card-title">Precision / Recall / F1 — Resolved vs Unresolved vs Macro Avg</div>
        <div class="chart-wrap"><canvas id="c1"></canvas></div>
      </div>
    </div>
    <div class="grid-2">
      <div class="card">
        <div class="card-title">Feature Importance — ranked list</div>
        <div class="feat-list">{feat_rows}</div>
      </div>
      <div class="card">
        <div class="card-title">Feature Importance — bar chart</div>
        <div class="chart-wrap"><canvas id="c2"></canvas></div>
      </div>
    </div>
    </div>
    <script>{CHART_CFG}
    new Chart(document.getElementById('c1'),{{type:'bar',
      data:{{labels:['Precision','Recall','F1-Score'],datasets:[
        {{label:'Resolved',data:[{round(res.get('precision',0),3)},{round(res.get('recall',0),3)},{round(res.get('f1-score',0),3)}],backgroundColor:'rgba(0,206,201,.7)',borderRadius:5}},
        {{label:'Unresolved',data:[{round(unres.get('precision',0),3)},{round(unres.get('recall',0),3)},{round(unres.get('f1-score',0),3)}],backgroundColor:'rgba(225,112,85,.7)',borderRadius:5}},
        {{label:'Macro Avg',data:[{round(macro.get('precision',0),3)},{round(macro.get('recall',0),3)},{round(macro.get('f1-score',0),3)}],backgroundColor:'rgba(253,203,110,.7)',borderRadius:5}}
      ]}},
      options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:LEG}},
        scales:{{x:{{...AX.x}},y:{{...AX.y,max:1,ticks:{{callback:v=>v.toFixed(2)}}}}}}}}
    }});
    new Chart(document.getElementById('c2'),{{type:'bar',
      data:{{labels:{json.dumps([f[0].replace("_"," ") for f in fi_sorted])},
        datasets:[{{label:'Importance %',data:{json.dumps([round(f[1]*100,2) for f in fi_sorted])},
          backgroundColor:'rgba(108,92,231,.7)',borderRadius:5}}]}},
      options:{{indexAxis:'y',responsive:true,maintainAspectRatio:false,
        plugins:{{legend:{{display:false}}}},
        scales:{{x:{{...AX.x,ticks:{{callback:v=>v+'%'}}}},
          y:{{ticks:{{color:'#8a8aaa',font:{{size:10}}}},grid:{{display:false}}}}}}}}
    }});
    </script></body></html>"""

# ─────────────────────────────────────────────────────────────────────────────
# PREDICT PAGE
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/predict')
def predict_page():
    s = stats
    def opts(key):
        return ''.join(f'<option value="{v}">{v}</option>' for v in s.get(key,[]))

    return f"""<!DOCTYPE html><html lang="en"><head>
    <meta charset="UTF-8"><title>Predict — ComplaintIQ</title>{BASE_STYLE}</head><body>
    {nav('/predict')}<div class="wrap">
    <div class="page-header">
      <div class="page-title">Prediction Engine</div>
      <div class="page-desc">Enter complaint details to predict whether it will be resolved or remain unresolved.</div>
    </div>
    <div class="predict-layout">
      <div class="card">
        <div class="card-title">Complaint Details</div>
        <div class="form-grid">
          <div class="form-group"><label>Product Category</label><select id="p_cat">{opts('product_categories')}</select></div>
          <div class="form-group"><label>Issue Type</label><select id="i_type">{opts('issue_types')}</select></div>
          <div class="form-group"><label>Company</label><select id="comp">{opts('companies')}</select></div>
          <div class="form-group"><label>State</label><select id="st">{opts('states')}</select></div>
          <div class="form-group"><label>Submitted Via</label><select id="sv">{opts('submitted_vias')}</select></div>
          <div class="form-group"><label>Tags</label><select id="tg">{opts('tags')}</select></div>
          <div class="form-group"><label>Complaint Month (1-12)</label><input type="number" id="mth" value="6" min="1" max="12"></div>
          <div class="form-group"><label>Day of Week (0=Mon, 6=Sun)</label><input type="number" id="dow" value="1" min="0" max="6"></div>
        </div>
        <button class="predict-btn" onclick="predict()">⚡ Analyze &amp; Predict</button>
      </div>
      <div class="result-wrap">
        <div class="result-ph" id="rph">
          <div class="result-ph-icon">⚙️</div>
          <div class="result-ph-txt">Fill in the complaint details<br>and click Analyze to get<br>a resolution prediction</div>
        </div>
        <div class="result-body" id="rbody">
          <div class="result-box" id="rbox">
            <div class="rl">PREDICTED OUTCOME</div>
            <div class="rv" id="rv">—</div>
            <div class="rc" id="rc">—</div>
          </div>
          <div class="conf-wrap">
            <div class="conf-lbl"><span>Resolution Probability</span><span id="cpct">—</span></div>
            <div class="conf-track"><div class="conf-fill" id="cbar" style="width:0%"></div></div>
          </div>
          <div class="meta-list" id="rmeta"></div>
        </div>
      </div>
    </div></div>
    <script>
    async function predict(){{
      const p={{
        product_category:document.getElementById('p_cat').value,
        issue_type:document.getElementById('i_type').value,
        company:document.getElementById('comp').value,
        state:document.getElementById('st').value,
        submitted_via:document.getElementById('sv').value,
        tags:document.getElementById('tg').value,
        complaint_month:parseInt(document.getElementById('mth').value),
        complaint_dow:parseInt(document.getElementById('dow').value)
      }};
      try{{
        const r=await fetch('/api/predict',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(p)}});
        const d=await r.json(); showResult(d,p);
      }}catch(e){{alert('Error: '+e);}}
    }}
    function showResult(d,p){{
      document.getElementById('rph').style.display='none';
      document.getElementById('rbody').style.display='block';
      const ok=d.prediction==='Resolved';
      const conf=Math.round(d.confidence*100);
      const box=document.getElementById('rbox');
      box.className='result-box '+(ok?'resolved':'unresolved');
      const rv=document.getElementById('rv');
      rv.textContent=d.prediction; rv.className='rv '+(ok?'resolved':'unresolved');
      document.getElementById('rc').textContent=conf+'% confidence';
      document.getElementById('cpct').textContent=conf+'%';
      const bar=document.getElementById('cbar');
      bar.style.width=conf+'%'; bar.style.background=ok?'var(--resolved)':'var(--unresolved)';
      document.getElementById('rmeta').innerHTML=`
        <div class="meta-row"><span class="mk">Product</span><span class="mv">${{p.product_category}}</span></div>
        <div class="meta-row"><span class="mk">Issue</span><span class="mv" title="${{p.issue_type}}">${{p.issue_type}}</span></div>
        <div class="meta-row"><span class="mk">Company</span><span class="mv" title="${{p.company}}">${{p.company}}</span></div>
        <div class="meta-row"><span class="mk">State</span><span class="mv">${{p.state}}</span></div>
        <div class="meta-row"><span class="mk">Channel</span><span class="mv">${{p.submitted_via}}</span></div>`;
    }}
    </script></body></html>"""

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION API
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    def enc(key, val):
        e = encoders.get(key)
        return int(e.transform([val])[0]) if e and val in e.classes_ else 0

    features = [
        enc('Product_Category', data.get('product_category','')),
        enc('Issue_Type',        data.get('issue_type','')),
        int(data.get('complaint_month', 6)),
        int(data.get('complaint_dow', 1)),
        enc('Company',           data.get('company','')),
        enc('State',             data.get('state','')),
        enc('Submitted_via',     data.get('submitted_via','')),
        enc('Tags',              data.get('tags','')),
    ]
    X = np.array(features).reshape(1, -1)
    idx = model.predict(X)[0]
    conf = float(model.predict_proba(X)[0][idx])
    label = encoders['Resolved_Status'].inverse_transform([idx])[0]
    return jsonify({'prediction': label, 'confidence': conf})

if __name__ == '__main__':
    app.run(debug=True, port=5050)