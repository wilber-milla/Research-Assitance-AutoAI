"""
Merge multiple CSV extraction runs and regenerate the HTML report.
Uses the generate_html and deduplicate functions from extract_papers.py
"""
import csv
import re
import os

# Import functions from our main script
from extract_papers import deduplicate, generate_html, save_csv, CSV_FIELDS

def load_csv(filename):
    papers = []
    if not os.path.exists(filename):
        print(f"  [!] File not found: {filename}")
        return papers
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            papers.append(dict(row))
    print(f"  Loaded {len(papers)} papers from {filename}")
    return papers

def main():
    print("=" * 60)
    print("  Merging paper datasets...")
    print("=" * 60)

    # Load the current CSV (55 papers: 9 arXiv + 47 PubMed)
    current = load_csv("chromosome_cnn_papers.csv")

    # Load the backup from the previous run that had 245 papers
    # (we'll check if there's a backup, otherwise just use current)
    backup = load_csv("chromosome_cnn_papers_backup.csv") if os.path.exists("chromosome_cnn_papers_backup.csv") else []

    all_papers = current + backup
    all_papers = deduplicate(all_papers)
    all_papers.sort(key=lambda p: (str(p.get('Year', '')), p.get('Title', '')), reverse=True)

    print(f"\n  Total unique papers after merge: {len(all_papers)}")

    save_csv(all_papers)
    generate_html(all_papers)

    print("\n  [DONE] Merged report generated.")
    print("=" * 60)

if __name__ == "__main__":
    main()
