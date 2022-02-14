import os


def get_env_num(x):
    return int(x.split("=")[1].split(",")[0])


success_env_num = {}
success_env = {}
for file in sorted(os.listdir("evaluate_results"), key=lambda x: get_env_num(x)):
    if "tmp" not in file:
        with open("evaluate_results/{}".format(file), "r") as f:
            record = f.readlines()
        # print("Evaluating {}".format(file))
        env_num = get_env_num(file)
        if env_num not in success_env_num.keys():
            success_env_num[env_num] = []
        success = 0
        for i in record[1:]:
            env_index = int(i.split(",")[0])
            if env_index not in success_env.keys():
                success_env[env_index] = {
                    "BATCH_PER_ITER": 0,
                    "F": 0,
                }
            if "True" in i:
                success += 1
                success_env[env_index]["BATCH_PER_ITER"] += 1
            else:
                success_env[env_index]["F"] += 1
        success_env_num[env_num].append(success)
for env_num in success_env_num.keys():
    success_env_num[env_num] = sum(success_env_num[env_num]) / len(success_env_num[env_num]) / 70
for env_index in success_env.keys():
    success_env[env_index] = round(
        success_env[env_index]["BATCH_PER_ITER"] / (success_env[env_index]["BATCH_PER_ITER"] + success_env[env_index]["F"]), 4)
print(success_env_num)
print(success_env)
