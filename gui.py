import taipy as tp

scenarios = tp.get_scenarios()

scenario = scenarios[0]

print(f"Scenario name: {scenario.name}")

print(f"Model name: {scenario.model_name.read()}")
