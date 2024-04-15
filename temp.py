import taipy as tp
from taipy.gui import Gui, State
import pandas as pd

scenarios = tp.get_scenarios()
scenario_names = [scenario.name for scenario in scenarios]
scenario_results = [scenario.results.read() for scenario in scenarios]
print(scenario_results[0].columns)
