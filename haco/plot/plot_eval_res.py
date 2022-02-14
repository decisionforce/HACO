import pandas

from drivingforce.process import smooth, parse, get_termination, filter_nan
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from drivingforce.process.parse import parse

if __name__ == "__main__":
    file_path_1 = "./no_lag.json"
    file_path_2 = "./egpo_human.json"

    flat_result = []
    parse_res = parse("/home/liquanyi/plot/0922_ppo_on_human_env")
    for file_path in [file_path_1, file_path_2]:
        with open(file_path, "r") as file:
            result = json.load(file)


        for key, value in result.items():
            for episode in value:
                episode["steps"] = int(key)*100
                episode["label"] = "egpo without lag" if file_path==file_path_1 else "egpo"
                flat_result.append(episode)

    data = pd.DataFrame(flat_result)
    plot_data = pandas.concat([data, parse_res])
    for type in ["success", "reward", "cost"]:
        sns.set("paper", "darkgrid")
        plt.figure(figsize=(12, 9), dpi=200)
        if type == "success":
            plt.ylim(0, 1)
        sns.lineplot(
            data=plot_data,
            y=type,
            #     y="episode_reward_mean",
            x="steps",
            # ci="sd",
            palette="colorblind",
            hue="label",
        )
        plt.savefig("{}.png".format(type))

