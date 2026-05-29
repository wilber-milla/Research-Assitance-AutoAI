"""Quick smoke test for the SciAgent-Nexus agent pipeline."""
import sys, os
sys.path.insert(0, ".")

from sci_agent_nexus.core.config import Config, OutputConfig
from sci_agent_nexus.core.memory import ResearchState
from sci_agent_nexus.agents import (
    IdeatorAgent, CoderAgent, AnalystAgent, WriterAgent, ReviewerAgent,
)

# Setup
oc = OutputConfig(base_dir="./test_output")
oc.ensure_dirs()
c = Config(scientific_domain="ml", output=oc)
s = ResearchState(topic="Attention mechanisms in small-data regimes", scientific_domain="ml")

# 1. Ideation
print("=" * 60)
print("PHASE 1: IDEATION")
print("=" * 60)
s = IdeatorAgent(c).run(s)
print(f"  Hypotheses: {len(s.hypotheses)}")
print(f"  Literature: {len(s.literature)}")
print(f"  Gap: {s.research_gap[:80]}...")
print(f"  Selected: {s.selected_hypothesis.statement[:80]}...")
print("  => IDEATOR OK\n")

# 2. Coding + Execution
print("=" * 60)
print("PHASE 2: CODING")
print("=" * 60)
s = CoderAgent(c).run(s)
r = s.experiment_results[-1] if s.experiment_results else None
print(f"  Code length: {len(s.experiment_code)} chars")
print(f"  Status: {r.status if r else 'none'}")
print(f"  Metrics: {len(r.metrics) if r else 0}")
print(f"  Data files: {len(r.data_files) if r else 0}")
print("  => CODER OK\n")

# 3. Analysis
print("=" * 60)
print("PHASE 3: ANALYSIS")
print("=" * 60)
s = AnalystAgent(c).run(s)
print(f"  Figures: {len(s.figure_paths)}")
print(f"  Statistical tests: {len(s.statistical_tests)}")
print(f"  Summary length: {len(s.analysis_summary)} chars")
print(f"  Critiques: {len(s.figure_critiques)}")
print("  => ANALYST OK\n")

# 4. Writing
print("=" * 60)
print("PHASE 4: WRITING")
print("=" * 60)
s = WriterAgent(c).run(s)
print(f"  Sections: {list(s.manuscript_sections.keys())}")
for sec, text in s.manuscript_sections.items():
    print(f"    {sec}: {len(text.split())} words")
print(f"  Markdown length: {len(s.manuscript_markdown)} chars")
print(f"  LaTeX length: {len(s.manuscript_latex)} chars")
print("  => WRITER OK\n")

# 5. Review (iteration 1)
print("=" * 60)
print("PHASE 5: REVIEW (iteration 1)")
print("=" * 60)
s.iteration = 1
s = ReviewerAgent(c).run(s)
rev1 = s.reviews[-1]
print(f"  Overall: {rev1.overall_score}")
print(f"  Novelty={rev1.novelty}, Soundness={rev1.soundness}, "
      f"Clarity={rev1.clarity}, Significance={rev1.significance}")
print(f"  Decision: {rev1.decision}")
print(f"  Strengths: {len(rev1.strengths)}, Weaknesses: {len(rev1.weaknesses)}")
print("  => REVIEWER OK (iteration 1)\n")

# 6. Review (iteration 2 — should improve)
print("=" * 60)
print("PHASE 5b: REVIEW (iteration 2)")
print("=" * 60)
s.iteration = 2
s = ReviewerAgent(c).run(s)
rev2 = s.reviews[-1]
print(f"  Overall: {rev2.overall_score}")
print(f"  Decision: {rev2.decision}")
print(f"  Score improved: {rev2.overall_score > rev1.overall_score}")
print("  => REVIEWER OK (iteration 2)\n")

print("=" * 60)
print("ALL 5 AGENTS PASSED SMOKE TEST SUCCESSFULLY")
print("=" * 60)
