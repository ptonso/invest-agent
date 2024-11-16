
import os
from datetime import datetime
from src.Experiment import Experiment
from src.Configurations import Configurations


def main():

    describe_flag = True

    import torch
    torch.autograd.set_detect_anomaly(True)


    results_dir = "results"
    today_flag = datetime.today().strftime('%Y-%m-%d')
    exp_name = f"baseline_{today_flag}"

    os.makedirs(f"{results_dir}/{exp_name}", exist_ok=True)

    exp_path = f"{results_dir}/{exp_name}/config.json"
    savefig = f"{results_dir}/{exp_name}/plot.png"
    action_csv = f"{results_dir}/{exp_name}/actions.csv"

    config = Configurations(describe_flag=describe_flag,
                            savefig=savefig,
                            action_csv=action_csv)
    
    exp = Experiment(exp_name, config)
    exp.run()

    config.save_config(exp, filepath=exp_path)


if __name__ == "__main__":
    main()