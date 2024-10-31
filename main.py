
from src.Experiment import Experiment
from src.Configurations import Configurations


def main():

    describe_flag = True

    import torch
    torch.autograd.set_detect_anomaly(True)


    results_dir = "data/results"
    exp_name = "baseline"
    start_date = "2005-01-03"
    exp_path = f"{results_dir}/{exp_name}.json"
    savefig = f"{results_dir}/{exp_name}.png"

    config = Configurations(describe_flag=describe_flag,
                            savefig=savefig)
    
    exp = Experiment(exp_name, config)
    exp.run()

    config.save_config(exp, filepath=exp_path)


if __name__ == "__main__":
    main()