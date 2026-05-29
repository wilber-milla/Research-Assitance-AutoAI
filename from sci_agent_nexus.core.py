from sci_agent_nexus.core.orchestrator import Orchestrator
from sci_agent_nexus.core.config import Config

config = Config()
config.scientific_domain = "ml"  # "ml", "genomics", "general"

orchestrator = Orchestrator(config)
state = orchestrator.run("Effect of batch normalization on transformer convergence")
streamlit run sci_agent_nexus/web_ui.py
