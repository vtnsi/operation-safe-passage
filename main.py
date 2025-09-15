import argparse
import os
from operation_safe_passage.environment.map_generator import MapGenerator
from operation_safe_passage.policies.osp_rl import OSPReinforcementLearning



def main():
    parser = argparse.ArgumentParser(
        prog="FAIREST",
        usage="%(prog)s [options]",
        epilog="Mode options are: main (default), map, train, and ugv"
    )
    parser.add_argument("mode", default="main")
    args = parser.parse_args()
    if args.mode == "map":
        endpoint = MapGenerator(
            parameters_file=os.path.join("config", "params.json"),
            output_dir="output"
        )
    elif args.mode == "rl":
        endpoint = OSPReinforcementLearning(
            param_path="config/params.json",
            network_path="config/network.json",
            max_iters=2000,
            verbose=True,
        )
    endpoint.run()


if __name__ == '__main__':
    main()
