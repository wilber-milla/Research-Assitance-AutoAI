"""
Systematic Literature Review: CNNs in Chromosome Identification, Segmentation & Classification (2020-2026)
Fetches from Semantic Scholar, arXiv, and PubMed.
Generates a structured CSV and a rich HTML report.
"""

import urllib.request
import urllib.parse
import json
import xml.etree.ElementTree as ET
import csv
import re
import time
import ssl
import html as html_mod
from datetime import datetime

# ──────────────────────────────────────────────
# 1. PARAMETER EXTRACTION FROM ABSTRACT TEXT
# ──────────────────────────────────────────────

MODEL_PATTERNS = [
    (r'u-?net\b', 'U-Net'),
    (r'attention\s*u-?net', 'Attention U-Net'),
    (r'resnet[-\s]?(\d+)', 'ResNet'),
    (r'res[-\s]?net', 'ResNet'),
    (r'vgg[-\s]?(\d+)', 'VGG'),
    (r'vgg\b', 'VGG'),
    (r'inception', 'Inception'),
    (r'google\s*net', 'GoogLeNet'),
    (r'alex\s*net', 'AlexNet'),
    (r'dense\s*net', 'DenseNet'),
    (r'efficient\s*net', 'EfficientNet'),
    (r'mobile\s*net', 'MobileNet'),
    (r'squeeze\s*net', 'SqueezeNet'),
    (r'yolo\s*v?\d*', 'YOLO'),
    (r'mask\s*r-?cnn', 'Mask R-CNN'),
    (r'faster\s*r-?cnn', 'Faster R-CNN'),
    (r'r-?cnn', 'R-CNN'),
    (r'fcn\b', 'FCN'),
    (r'seg\s*net', 'SegNet'),
    (r'deep\s*lab', 'DeepLab'),
    (r'transformer', 'Transformer/ViT'),
    (r'vi[ts]\b', 'Vision Transformer'),
    (r'gan\b', 'GAN'),
    (r'auto\s*encoder', 'Autoencoder'),
    (r'lstm', 'LSTM'),
    (r'capsule', 'CapsNet'),
    (r'xception', 'Xception'),
    (r'nas\s*net', 'NASNet'),
    (r'dcnn', 'DCNN'),
    (r'deep\s+convolutional', 'DCNN'),
    (r'convolutional neural network', 'CNN'),
    (r'\bcnn\b', 'CNN'),
]

def extract_parameters(text):
    default = {
        "Model": "Not specified",
        "Num_Images": "Not specified",
        "Epochs": "Not specified",
        "Batch_Size": "Not specified",
        "Accuracy": "Not specified",
        "Precision": "Not specified",
        "Recall": "Not specified",
        "F1_Score": "Not specified",
        "IoU": "Not specified",
        "AUC": "Not specified",
        "Other_Metrics": "Not specified",
    }
    if not text:
        return default

    t = text.lower()

    # ── Models ──
    found_models = []
    for pat, name in MODEL_PATTERNS:
        m = re.search(pat, t)
        if m:
            label = name
            if name == 'ResNet' and m.lastindex and m.group(m.lastindex):
                label = f"ResNet-{m.group(m.lastindex)}"
            if name == 'VGG' and m.lastindex and m.group(m.lastindex):
                label = f"VGG-{m.group(m.lastindex)}"
            if label not in found_models:
                found_models.append(label)
    # Deduplicate: if we have a specific CNN variant, drop generic 'CNN'
    if len(found_models) > 1 and 'CNN' in found_models:
        found_models.remove('CNN')
    default["Model"] = ", ".join(found_models) if found_models else "Not specified"

    # ── Dataset size ──
    img_pat = r'([\d,\.]+)\s*(?:k\s*)?(training\s+)?(images|samples|chromosomes|metaphase|karyotype|cells|patches|data\s*points|instances)'
    m = re.search(img_pat, t)
    if m:
        num = m.group(1).replace(',', '')
        try:
            val = float(num)
            if 'k' in t[m.start():m.end()+2]:
                val = int(val * 1000)
            default["Num_Images"] = f"{int(val) if val == int(val) else val} {m.group(3)}"
        except ValueError:
            default["Num_Images"] = f"{m.group(1)} {m.group(3)}"

    # ── Epochs ──
    m = re.search(r'(\d+)\s*epochs?', t)
    if m:
        default["Epochs"] = m.group(1)

    # ── Batch size ──
    m = re.search(r'batch[\s\-_]*size\s*(?:of\s*|=\s*|:\s*)?(\d+)', t)
    if m:
        default["Batch_Size"] = m.group(1)

    # ── Metrics ──
    def find_metric(pattern):
        m = re.search(pattern, t)
        if m:
            return m.group(m.lastindex)
        return None

    # Accuracy
    v = find_metric(r'(?:accuracy|acc\.?)\s*(?:of|is|was|=|:)?\s*([\d\.]+)\s*%') or \
        find_metric(r'([\d\.]+)\s*%\s*(?:accuracy|acc)')
    if v: default["Accuracy"] = f"{v}%"

    # Precision
    v = find_metric(r'precision\s*(?:of|is|was|=|:)?\s*([\d\.]+)\s*%') or \
        find_metric(r'([\d\.]+)\s*%\s*precision')
    if v: default["Precision"] = f"{v}%"

    # Recall / Sensitivity
    v = find_metric(r'(?:recall|sensitivity)\s*(?:of|is|was|=|:)?\s*([\d\.]+)\s*%') or \
        find_metric(r'([\d\.]+)\s*%\s*(?:recall|sensitivity)')
    if v: default["Recall"] = f"{v}%"

    # F1
    v = find_metric(r'f1[\s\-_]*score?\s*(?:of|is|was|=|:)?\s*([\d\.]+)\s*%') or \
        find_metric(r'([\d\.]+)\s*%\s*f1')
    if v: default["F1_Score"] = f"{v}%"

    # IoU
    v = find_metric(r'(?:iou|intersection[\s\-]over[\s\-]union)\s*(?:of|is|was|=|:)?\s*([\d\.]+)\s*%')
    if v: default["IoU"] = f"{v}%"

    # AUC
    v = find_metric(r'(?:auc|area under)\s*(?:of|is|was|=|:)?\s*(0?\.\d+|[\d\.]+\s*%)')
    if v: default["AUC"] = v

    return default


# ──────────────────────────────────────────────
# 2. DATA FETCHERS
# ──────────────────────────────────────────────

def fetch_semantic_scholar():
    print("  [S2] Querying Semantic Scholar...")
    queries = [
        "convolutional neural network chromosome classification",
        "deep learning chromosome segmentation karyotyping",
        "CNN chromosome identification automated karyotype",
    ]
    all_results = []
    seen_titles = set()

    for qi, q in enumerate(queries):
        encoded = urllib.parse.quote(q)
        url = (
            f"https://api.semanticscholar.org/graph/v1/paper/search"
            f"?query={encoded}&year=2020-2026"
            f"&fields=title,authors,year,abstract,url,citationCount,venue"
            f"&limit=100"
        )
        req = urllib.request.Request(url, headers={'User-Agent': 'ResearchAssistant/1.0'})
        success = False
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                    for item in data.get('data', []):
                        title = (item.get('title') or '').strip()
                        if not title or title.lower() in seen_titles:
                            continue
                        seen_titles.add(title.lower())

                        abstract = item.get('abstract', '') or ''
                        params = extract_parameters(abstract)
                        authors_list = item.get('authors', [])
                        authors_str = ", ".join(a.get('name', '') for a in authors_list[:5])
                        if len(authors_list) > 5:
                            authors_str += " et al."

                        all_results.append({
                            "Title": title,
                            "Authors": authors_str,
                            "Year": item.get('year', ''),
                            "Venue": item.get('venue', '') or '',
                            "Citations": item.get('citationCount', 0) or 0,
                            "Database": "Semantic Scholar",
                            "Abstract": abstract,
                            **params,
                            "URL": item.get('url', '') or '',
                        })
                success = True
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = 5 * (attempt + 1)
                    print(f"    Rate limited (429). Waiting {wait}s before retry {attempt+2}/4...")
                    time.sleep(wait)
                else:
                    print(f"    [!] Semantic Scholar HTTP error: {e}")
                    break
            except Exception as e:
                print(f"    [!] Semantic Scholar query error: {e}")
                break
        if not success:
            print(f"    [!] Skipping S2 query {qi+1} after retries.")
        time.sleep(8)  # longer pause between queries to avoid rate limiting

    print(f"    [OK] Semantic Scholar: {len(all_results)} papers")
    return all_results


def fetch_arxiv():
    print("  [arXiv] Querying arXiv...")
    # Use URL-safe query strings (encode quotes as %22, spaces as +)
    queries = [
        'all:%22convolutional+neural+network%22+AND+all:%22chromosome%22',
        'all:%22deep+learning%22+AND+all:%22karyotype%22+AND+all:%22classification%22',
        'all:%22chromosome+segmentation%22+AND+all:%22neural+network%22',
    ]
    all_results = []
    seen_titles = set()
    ns = {'a': 'http://www.w3.org/2005/Atom'}

    for q in queries:
        url = f'http://export.arxiv.org/api/query?search_query={q}&start=0&max_results=80&sortBy=relevance'
        # Use unverified SSL context to work around local certificate issues
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        try:
            with urllib.request.urlopen(url, timeout=30, context=ssl_ctx) as resp:
                root = ET.fromstring(resp.read())
                for entry in root.findall('a:entry', ns):
                    title_el = entry.find('a:title', ns)
                    if title_el is None or not title_el.text:
                        continue
                    title = ' '.join(title_el.text.split()).strip()
                    if title.lower() in seen_titles:
                        continue
                    seen_titles.add(title.lower())

                    summary_el = entry.find('a:summary', ns)
                    summary = ' '.join((summary_el.text or '').split())
                    pub = entry.find('a:published', ns)
                    year = (pub.text or '')[:4]
                    id_el = entry.find('a:id', ns)
                    url_link = (id_el.text or '').strip()

                    try:
                        if int(year) < 2020:
                            continue
                    except ValueError:
                        continue

                    authors = []
                    for author_el in entry.findall('a:author', ns):
                        name_el = author_el.find('a:name', ns)
                        if name_el is not None and name_el.text:
                            authors.append(name_el.text)
                    authors_str = ", ".join(authors[:5])
                    if len(authors) > 5:
                        authors_str += " et al."

                    params = extract_parameters(summary)
                    all_results.append({
                        "Title": title,
                        "Authors": authors_str,
                        "Year": year,
                        "Venue": "arXiv Preprint",
                        "Citations": "",
                        "Database": "arXiv",
                        "Abstract": summary,
                        **params,
                        "URL": url_link,
                    })
        except Exception as e:
            print(f"    [!] arXiv query error: {e}")
        time.sleep(2)

    print(f"    [OK] arXiv: {len(all_results)} papers")
    return all_results


def fetch_pubmed():
    print("  [PubMed] Querying PubMed...")
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    queries = [
        '"convolutional neural network" AND "chromosome" AND ("classification" OR "segmentation" OR "identification" OR "karyotype")',
        '"deep learning" AND "chromosome" AND "karyotyping"',
    ]
    all_results = []
    seen_titles = set()

    for q in queries:
        # Step 1: esearch
        search_url = (
            f"{base}/esearch.fcgi?db=pubmed&retmode=json&retmax=100"
            f"&mindate=2020/01/01&maxdate=2026/12/31&datetype=pdat"
            f"&term={urllib.parse.quote(q)}"
        )
        try:
            with urllib.request.urlopen(search_url, timeout=30) as resp:
                sdata = json.loads(resp.read().decode())
                ids = sdata.get('esearchresult', {}).get('idlist', [])
        except Exception as e:
            print(f"    [!] PubMed search error: {e}")
            continue

        if not ids:
            continue

        # Step 2: efetch (XML)
        time.sleep(0.5)
        fetch_url = (
            f"{base}/efetch.fcgi?db=pubmed&retmode=xml&id={','.join(ids)}"
        )
        try:
            with urllib.request.urlopen(fetch_url, timeout=60) as resp:
                xml_data = resp.read()
                root = ET.fromstring(xml_data)
        except Exception as e:
            print(f"    [!] PubMed fetch error: {e}")
            continue

        for article in root.findall('.//PubmedArticle'):
            try:
                medline = article.find('MedlineCitation')
                art = medline.find('Article')
                title_el = art.find('ArticleTitle')
                title = ''.join(title_el.itertext()).strip() if title_el is not None else ''
                if not title or title.lower() in seen_titles:
                    continue
                seen_titles.add(title.lower())

                # Abstract
                abs_el = art.find('Abstract')
                abstract = ''
                if abs_el is not None:
                    parts = []
                    for at in abs_el.findall('AbstractText'):
                        parts.append(''.join(at.itertext()))
                    abstract = ' '.join(parts)

                # Year
                pd = art.find('.//PubDate')
                year = ''
                if pd is not None:
                    y_el = pd.find('Year')
                    if y_el is not None:
                        year = y_el.text
                    else:
                        ml = pd.find('MedlineDate')
                        if ml is not None and ml.text:
                            m = re.search(r'(20\d{2})', ml.text)
                            if m: year = m.group(1)

                # Authors
                author_list = art.find('AuthorList')
                authors = []
                if author_list is not None:
                    for au in author_list.findall('Author'):
                        ln = au.find('LastName')
                        fn = au.find('ForeName')
                        if ln is not None and ln.text:
                            name = ln.text
                            if fn is not None and fn.text:
                                name = f"{fn.text} {ln.text}"
                            authors.append(name)
                authors_str = ", ".join(authors[:5])
                if len(authors) > 5:
                    authors_str += " et al."

                # Journal
                journal_el = art.find('.//Journal/Title')
                venue = journal_el.text if journal_el is not None else ''

                # DOI / URL
                pmid_el = medline.find('PMID')
                pmid = pmid_el.text if pmid_el is not None else ''
                url_link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ''

                params = extract_parameters(abstract)
                all_results.append({
                    "Title": title,
                    "Authors": authors_str,
                    "Year": year,
                    "Venue": venue or '',
                    "Citations": "",
                    "Database": "PubMed",
                    "Abstract": abstract,
                    **params,
                    "URL": url_link,
                })
            except Exception:
                continue

        time.sleep(1)

    print(f"    [OK] PubMed: {len(all_results)} papers")
    return all_results


# ──────────────────────────────────────────────
# 3. DEDUPLICATION
# ──────────────────────────────────────────────

def deduplicate(papers):
    seen = set()
    unique = []
    for p in papers:
        key = re.sub(r'[^a-z0-9]', '', p['Title'].lower())[:80]
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


# ──────────────────────────────────────────────
# 4. CSV EXPORT
# ──────────────────────────────────────────────

CSV_FIELDS = [
    "Title", "Authors", "Year", "Venue", "Citations", "Database",
    "Model", "Num_Images", "Epochs", "Batch_Size",
    "Accuracy", "Precision", "Recall", "F1_Score", "IoU", "AUC", "Other_Metrics",
    "URL",
]

def save_csv(papers, filename="chromosome_cnn_papers.csv"):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        w.writeheader()
        for p in papers:
            w.writerow(p)
    print(f"  [CSV] Saved: {filename} ({len(papers)} papers)")


# ──────────────────────────────────────────────
# 5. HTML REPORT GENERATION
# ──────────────────────────────────────────────

def esc(text):
    return html_mod.escape(str(text)) if text else ''

def generate_html(papers, filename="chromosome_cnn_report.html"):
    now = datetime.now().strftime("%B %d, %Y — %H:%M")
    total = len(papers)

    # Stats
    by_year = {}
    by_db = {}
    by_model = {}
    specified_acc = []
    for p in papers:
        y = str(p.get('Year', 'N/A'))
        by_year[y] = by_year.get(y, 0) + 1
        db = p.get('Database', 'Unknown')
        by_db[db] = by_db.get(db, 0) + 1
        models = [m.strip() for m in p.get('Model', '').split(',') if m.strip() and m.strip() != 'Not specified']
        for m in models:
            by_model[m] = by_model.get(m, 0) + 1
        acc = p.get('Accuracy', '')
        if acc and acc != 'Not specified':
            try:
                specified_acc.append(float(acc.replace('%', '')))
            except ValueError:
                pass

    years_sorted = sorted(by_year.keys())
    year_labels = json.dumps(years_sorted)
    year_values = json.dumps([by_year[y] for y in years_sorted])

    model_sorted = sorted(by_model.items(), key=lambda x: -x[1])[:15]
    model_labels = json.dumps([m[0] for m in model_sorted])
    model_values = json.dumps([m[1] for m in model_sorted])

    db_labels = json.dumps(list(by_db.keys()))
    db_values = json.dumps(list(by_db.values()))

    # Build table rows
    rows_html = ""
    for i, p in enumerate(papers):
        title_safe = esc(p.get('Title', ''))
        url = esc(p.get('URL', ''))
        link = f'<a href="{url}" target="_blank" rel="noopener">{title_safe}</a>' if url else title_safe

        na_badge = '<span class="badge badge-na">N/A</span>'

        def badge(val, cls=""):
            if not val or val == "Not specified":
                return na_badge
            return f'<span class="badge {cls}">{esc(val)}</span>'

        num_img = esc(p.get('Num_Images', '')) if p.get('Num_Images', '') != 'Not specified' else na_badge
        epochs_val = esc(p.get('Epochs', '')) if p.get('Epochs', '') != 'Not specified' else na_badge
        batch_val = esc(p.get('Batch_Size', '')) if p.get('Batch_Size', '') != 'Not specified' else na_badge

        rows_html += f"""
        <tr>
            <td class="num">{i+1}</td>
            <td class="title-cell">{link}
                <div class="authors">{esc(p.get('Authors',''))}</div>
            </td>
            <td class="center">{esc(p.get('Year',''))}</td>
            <td>{badge(p.get('Database',''), 'badge-db')}</td>
            <td>{badge(p.get('Model',''), 'badge-model')}</td>
            <td class="center">{num_img}</td>
            <td class="center">{epochs_val}</td>
            <td class="center">{batch_val}</td>
            <td class="center">{badge(p.get('Accuracy',''), 'badge-metric')}</td>
            <td class="center">{badge(p.get('Precision',''), 'badge-metric')}</td>
            <td class="center">{badge(p.get('Recall',''), 'badge-metric')}</td>
            <td class="center">{badge(p.get('F1_Score',''), 'badge-metric')}</td>
            <td class="center">{badge(p.get('IoU',''), 'badge-metric')}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Systematic Review — CNNs in Chromosome Analysis (2020-2026)</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#0a0e1a;--bg2:#111827;--bg3:#1e293b;--surface:#1e2640;
  --border:#2d3a5c;--text:#e2e8f0;--text2:#94a3b8;--text3:#64748b;
  --accent:#6366f1;--accent2:#818cf8;--accent-glow:rgba(99,102,241,.25);
  --green:#22c55e;--cyan:#06b6d4;--amber:#f59e0b;--rose:#f43f5e;
  --purple:#a855f7;--teal:#14b8a6;
  --radius:12px;--radius-sm:8px;
}}
html{{scroll-behavior:smooth}}
body{{
  font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--text);
  line-height:1.6;-webkit-font-smoothing:antialiased;
}}
/* ─── Hero ─── */
.hero{{
  position:relative;overflow:hidden;
  background:linear-gradient(135deg,#0f172a 0%,#1e1b4b 40%,#312e81 70%,#1e1b4b 100%);
  padding:4rem 2rem 3rem;text-align:center;
  border-bottom:1px solid var(--border);
}}
.hero::before{{
  content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse 80% 50% at 50% 0%,rgba(99,102,241,.2),transparent);
  pointer-events:none;
}}
.hero h1{{
  font-size:clamp(1.8rem,4vw,2.8rem);font-weight:800;
  background:linear-gradient(135deg,#c7d2fe,#818cf8,#6366f1);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  letter-spacing:-.02em;margin-bottom:.5rem;position:relative;
}}
.hero .subtitle{{color:var(--text2);font-size:1.05rem;font-weight:400;position:relative}}
.hero .meta{{margin-top:1.2rem;display:flex;gap:1.5rem;justify-content:center;flex-wrap:wrap;position:relative}}
.hero .meta-item{{
  background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.08);
  border-radius:var(--radius);padding:.6rem 1.2rem;
}}
.hero .meta-item .label{{font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;color:var(--text3)}}
.hero .meta-item .value{{font-size:1.6rem;font-weight:700;color:var(--accent2)}}

/* ─── Container ─── */
.container{{max-width:1500px;margin:0 auto;padding:2rem 1.5rem}}

/* ─── Section ─── */
.section-title{{
  font-size:1.35rem;font-weight:700;margin-bottom:1.5rem;
  display:flex;align-items:center;gap:.6rem;
}}
.section-title .icon{{font-size:1.4rem}}

/* ─── Charts Grid ─── */
.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:1.5rem;margin-bottom:2.5rem}}
.chart-card{{
  background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  padding:1.5rem;transition:transform .2s,box-shadow .2s;
}}
.chart-card:hover{{transform:translateY(-2px);box-shadow:0 8px 30px rgba(0,0,0,.3)}}
.chart-card h3{{font-size:.95rem;font-weight:600;margin-bottom:1rem;color:var(--text2)}}
.chart-wrap{{position:relative;height:280px}}

/* ─── KPI Row ─── */
.kpi-row{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:1rem;margin-bottom:2.5rem}}
.kpi{{
  background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  padding:1.2rem 1.5rem;text-align:center;transition:transform .2s;
}}
.kpi:hover{{transform:translateY(-2px)}}
.kpi .kpi-value{{font-size:2rem;font-weight:800;background:linear-gradient(135deg,var(--accent),var(--cyan));-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.kpi .kpi-label{{font-size:.75rem;text-transform:uppercase;letter-spacing:.08em;color:var(--text3);margin-top:.3rem}}

/* ─── Table ─── */
.table-section{{margin-bottom:3rem}}
.table-controls{{
  display:flex;gap:1rem;margin-bottom:1rem;flex-wrap:wrap;align-items:center;
}}
.search-box{{
  flex:1;min-width:250px;padding:.65rem 1rem .65rem 2.6rem;
  background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius-sm);
  color:var(--text);font-size:.9rem;outline:none;transition:border-color .2s;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='%2364748b' viewBox='0 0 16 16'%3E%3Cpath d='M11.742 10.344a6.5 6.5 0 1 0-1.397 1.398h-.001l3.85 3.85a1 1 0 0 0 1.415-1.414l-3.85-3.85zm-5.242.156a5 5 0 1 1 0-10 5 5 0 0 1 0 10z'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:.8rem center;
}}
.search-box:focus{{border-color:var(--accent)}}
.filter-btn{{
  padding:.65rem 1.1rem;background:var(--bg2);border:1px solid var(--border);
  border-radius:var(--radius-sm);color:var(--text2);font-size:.8rem;cursor:pointer;
  transition:all .2s;
}}
.filter-btn:hover,.filter-btn.active{{background:var(--accent);color:#fff;border-color:var(--accent)}}

.table-wrap{{
  overflow-x:auto;border:1px solid var(--border);border-radius:var(--radius);
  background:var(--surface);
}}
table{{width:100%;border-collapse:collapse;font-size:.82rem}}
thead{{background:var(--bg3);position:sticky;top:0;z-index:2}}
th{{
  padding:.8rem .7rem;text-align:left;font-weight:600;color:var(--text2);
  text-transform:uppercase;font-size:.68rem;letter-spacing:.06em;
  border-bottom:2px solid var(--border);white-space:nowrap;cursor:pointer;
  user-select:none;transition:color .2s;
}}
th:hover{{color:var(--accent2)}}
th .sort-arrow{{font-size:.6rem;margin-left:.3rem;opacity:.4}}
td{{padding:.7rem .7rem;border-bottom:1px solid rgba(255,255,255,.04);vertical-align:top}}
tr:hover td{{background:rgba(99,102,241,.04)}}
.num{{color:var(--text3);font-size:.75rem;width:2.5rem;text-align:center}}
.center{{text-align:center}}
.title-cell{{max-width:350px;min-width:200px}}
.title-cell a{{color:var(--accent2);text-decoration:none;font-weight:500;line-height:1.4}}
.title-cell a:hover{{text-decoration:underline}}
.authors{{font-size:.72rem;color:var(--text3);margin-top:.2rem;line-height:1.3}}

/* ─── Badges ─── */
.badge{{
  display:inline-block;padding:.15rem .5rem;border-radius:4px;
  font-size:.72rem;font-weight:500;white-space:nowrap;
}}
.badge-na{{background:rgba(255,255,255,.05);color:var(--text3)}}
.badge-db{{background:rgba(6,182,212,.12);color:var(--cyan)}}
.badge-model{{background:rgba(168,85,247,.12);color:var(--purple)}}
.badge-metric{{background:rgba(34,197,94,.12);color:var(--green)}}

/* ─── Footer ─── */
.footer{{
  text-align:center;padding:2rem;color:var(--text3);font-size:.78rem;
  border-top:1px solid var(--border);margin-top:2rem;
}}

/* ─── Scroll to top ─── */
.scroll-top{{
  position:fixed;bottom:2rem;right:2rem;width:44px;height:44px;
  background:var(--accent);color:#fff;border:none;border-radius:50%;
  cursor:pointer;font-size:1.2rem;display:none;align-items:center;
  justify-content:center;box-shadow:0 4px 20px var(--accent-glow);
  transition:transform .2s;z-index:100;
}}
.scroll-top:hover{{transform:scale(1.1)}}
.scroll-top.visible{{display:flex}}

/* ─── Responsive ─── */
@media(max-width:768px){{
  .hero{{padding:2.5rem 1rem 2rem}}
  .charts{{grid-template-columns:1fr}}
  .kpi-row{{grid-template-columns:repeat(2,1fr)}}
  .container{{padding:1.5rem 1rem}}
}}
</style>
</head>
<body>

<!-- ═══════ HERO ═══════ -->
<header class="hero">
  <h1>🧬 Systematic Literature Review</h1>
  <p class="subtitle">Convolutional Neural Networks in Chromosome Identification, Segmentation &amp; Classification (2020–2026)</p>
  <div class="meta">
    <div class="meta-item"><div class="label">Total Papers</div><div class="value">{total}</div></div>
    <div class="meta-item"><div class="label">Databases</div><div class="value">{len(by_db)}</div></div>
    <div class="meta-item"><div class="label">Year Range</div><div class="value">2020–2026</div></div>
    <div class="meta-item"><div class="label">Generated</div><div class="value" style="font-size:1rem">{now}</div></div>
  </div>
</header>

<div class="container">

  <!-- ═══════ KPI ═══════ -->
  <div class="kpi-row">
    <div class="kpi"><div class="kpi-value">{total}</div><div class="kpi-label">Papers Found</div></div>
    <div class="kpi"><div class="kpi-value">{len(by_model)}</div><div class="kpi-label">Distinct Models</div></div>
    <div class="kpi"><div class="kpi-value">{f"{sum(specified_acc)/len(specified_acc):.1f}%" if specified_acc else "N/A"}</div><div class="kpi-label">Avg Accuracy (reported)</div></div>
    <div class="kpi"><div class="kpi-value">{model_sorted[0][0] if model_sorted else "N/A"}</div><div class="kpi-label">Most Used Model</div></div>
  </div>

  <!-- ═══════ CHARTS ═══════ -->
  <h2 class="section-title"><span class="icon">📊</span> Distribution Overview</h2>
  <div class="charts">
    <div class="chart-card">
      <h3>Publications by Year</h3>
      <div class="chart-wrap"><canvas id="chartYear"></canvas></div>
    </div>
    <div class="chart-card">
      <h3>Papers by Database Source</h3>
      <div class="chart-wrap"><canvas id="chartDB"></canvas></div>
    </div>
    <div class="chart-card">
      <h3>Most Frequently Used Models</h3>
      <div class="chart-wrap"><canvas id="chartModels"></canvas></div>
    </div>
  </div>

  <!-- ═══════ TABLE ═══════ -->
  <section class="table-section">
    <h2 class="section-title"><span class="icon">📄</span> All Papers</h2>
    <div class="table-controls">
      <input class="search-box" id="searchInput" type="text" placeholder="Search by title, model, year, database…">
      <button class="filter-btn active" data-filter="all">All</button>
      <button class="filter-btn" data-filter="Semantic Scholar">Semantic Scholar</button>
      <button class="filter-btn" data-filter="arXiv">arXiv</button>
      <button class="filter-btn" data-filter="PubMed">PubMed</button>
    </div>
    <div class="table-wrap">
      <table id="papersTable">
        <thead>
          <tr>
            <th>#</th>
            <th>Title / Authors</th>
            <th>Year <span class="sort-arrow">▲</span></th>
            <th>Source</th>
            <th>Model(s)</th>
            <th>Images</th>
            <th>Epochs</th>
            <th>Batch</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
            <th>IoU</th>
          </tr>
        </thead>
        <tbody id="tableBody">
          {rows_html}
        </tbody>
      </table>
    </div>
  </section>
</div>

<button class="scroll-top" id="scrollTop" title="Back to top">↑</button>

<footer class="footer">
  Generated by Research Assistant &middot; Data extracted from Semantic Scholar, arXiv &amp; PubMed APIs &middot; {now}
</footer>

<script>
// ─── Charts ───
const yearCtx = document.getElementById('chartYear').getContext('2d');
new Chart(yearCtx, {{
  type: 'bar',
  data: {{
    labels: {year_labels},
    datasets: [{{
      label: 'Papers',
      data: {year_values},
      backgroundColor: 'rgba(99,102,241,.6)',
      borderColor: '#6366f1',
      borderWidth: 1,
      borderRadius: 6,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: '#94a3b8' }}, grid: {{ color: 'rgba(255,255,255,.05)' }} }},
      y: {{ beginAtZero: true, ticks: {{ color: '#94a3b8', stepSize: 1 }}, grid: {{ color: 'rgba(255,255,255,.05)' }} }}
    }}
  }}
}});

const dbCtx = document.getElementById('chartDB').getContext('2d');
new Chart(dbCtx, {{
  type: 'doughnut',
  data: {{
    labels: {db_labels},
    datasets: [{{
      data: {db_values},
      backgroundColor: ['#6366f1','#06b6d4','#f59e0b','#22c55e','#f43f5e'],
      borderColor: '#1e2640',
      borderWidth: 3,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ position: 'bottom', labels: {{ color: '#94a3b8', padding: 16 }} }} }}
  }}
}});

const modelCtx = document.getElementById('chartModels').getContext('2d');
new Chart(modelCtx, {{
  type: 'bar',
  data: {{
    labels: {model_labels},
    datasets: [{{
      label: 'Count',
      data: {model_values},
      backgroundColor: 'rgba(168,85,247,.5)',
      borderColor: '#a855f7',
      borderWidth: 1,
      borderRadius: 6,
    }}]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ beginAtZero: true, ticks: {{ color: '#94a3b8', stepSize: 1 }}, grid: {{ color: 'rgba(255,255,255,.05)' }} }},
      y: {{ ticks: {{ color: '#94a3b8', font: {{ size: 11 }} }}, grid: {{ display: false }} }}
    }}
  }}
}});

// ─── Search & Filter ───
const searchInput = document.getElementById('searchInput');
const filterBtns = document.querySelectorAll('.filter-btn');
const tbody = document.getElementById('tableBody');
let activeFilter = 'all';

function applyFilters() {{
  const q = searchInput.value.toLowerCase();
  const rows = tbody.querySelectorAll('tr');
  rows.forEach(row => {{
    const text = row.textContent.toLowerCase();
    const matchSearch = !q || text.includes(q);
    const matchFilter = activeFilter === 'all' || text.includes(activeFilter.toLowerCase());
    row.style.display = (matchSearch && matchFilter) ? '' : 'none';
  }});
}}
searchInput.addEventListener('input', applyFilters);
filterBtns.forEach(btn => {{
  btn.addEventListener('click', () => {{
    filterBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    activeFilter = btn.dataset.filter;
    applyFilters();
  }});
}});

// ─── Scroll to top ───
const scrollBtn = document.getElementById('scrollTop');
window.addEventListener('scroll', () => {{
  scrollBtn.classList.toggle('visible', window.scrollY > 400);
}});
scrollBtn.addEventListener('click', () => window.scrollTo({{ top: 0, behavior: 'smooth' }}));
</script>
</body>
</html>"""

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  [HTML] Report saved: {filename}")


# ──────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Systematic Review -- CNN + Chromosome (2020-2026)")
    print("=" * 60)

    papers = []

    papers.extend(fetch_semantic_scholar())
    time.sleep(2)
    papers.extend(fetch_arxiv())
    time.sleep(2)
    papers.extend(fetch_pubmed())

    papers = deduplicate(papers)
    papers.sort(key=lambda p: (str(p.get('Year', '')), p.get('Title', '')), reverse=True)

    print(f"\n  Total unique papers: {len(papers)}")

    save_csv(papers)
    generate_html(papers)

    print("\n  [DONE] Open 'chromosome_cnn_report.html' in your browser.")
    print("=" * 60)

if __name__ == "__main__":
    main()
