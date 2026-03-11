import numpy as np
import os
import subprocess
import shutil
import sys
import tempfile
import time
import yaml

import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import util.arg_parser as arg_parser
from util.logger import Logger
import util.mp_util as mp_util
import util.util as util

import torch

def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)
    return

def load_args(argv):
    args = arg_parser.ArgParser()
    args.load_args(argv[1:])

    arg_file = args.parse_string("arg_file")
    if (arg_file != ""):
        succ = args.load_file(arg_file)
        assert succ, Logger.print("Failed to load args from: " + arg_file)

    return args

def build_env(args, num_envs, device, visualize):
    env_file = args.parse_string("env_config")
    engine_file = args.parse_string("engine_config")
    record_video = args.parse_bool("video", False)
    motion_file = args.parse_string("motion_file", "")

    if (motion_file != ""):
        with open(env_file, "r") as stream:
            env_config = yaml.safe_load(stream)

        env_config["motion_file"] = motion_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
            yaml.safe_dump(env_config, tmp_file, sort_keys=False)
            tmp_env_file = tmp_file.name
        env_file = tmp_env_file
    
    env = env_builder.build_env(env_file, engine_file, num_envs, device, visualize=visualize, record_video=record_video)

    if (motion_file != ""):
        os.remove(tmp_env_file)

    return env

def build_agent(args, env, device):
    agent_file = args.parse_string("agent_config")
    agent = agent_builder.build_agent(agent_file, env, device)
    return agent

def train(agent, max_samples, max_iters, out_dir, save_int_models, logger_type):
    agent.train_model(max_samples=max_samples, max_iters=max_iters, out_dir=out_dir, 
                      save_int_models=save_int_models, logger_type=logger_type)
    return

def test(agent, test_episodes):
    result = agent.test_model(num_episodes=test_episodes)
    
    Logger.print("Mean Return: {}".format(result["mean_return"]))
    Logger.print("Mean Episode Length: {}".format(result["mean_ep_len"]))
    Logger.print("Episodes: {}".format(result["num_eps"]))
    return result

def save_config_files(args, out_dir):
    engine_file = args.parse_string("engine_config")
    if (engine_file != ""):
        copy_file_to_dir(engine_file, "engine_config.yaml", out_dir)

    env_file = args.parse_string("env_config")
    if (env_file != ""):
        copy_file_to_dir(env_file, "env_config.yaml", out_dir)

    agent_file = args.parse_string("agent_config")
    if (agent_file != ""):
        copy_file_to_dir(agent_file, "agent_config.yaml", out_dir)
    return

def create_output_dir(out_dir):
    if (mp_util.is_root_proc()):
        if (out_dir != "" and (not os.path.exists(out_dir))):
            os.makedirs(out_dir, exist_ok=True)
    return

def copy_file_to_dir(in_path, out_filename, output_dir):
    out_file = os.path.join(output_dir, out_filename)
    shutil.copy(in_path, out_file)
    return

def set_rand_seed(args):
    rand_seed_key = "rand_seed"

    if (args.has_key(rand_seed_key)):
        rand_seed = args.parse_int(rand_seed_key)
    else:
        rand_seed = np.uint64(time.time() * 256)
        
    rand_seed += np.uint64(41 * mp_util.get_proc_rank())
    print("Setting seed: {}".format(rand_seed))
    util.set_rand_seed(rand_seed)
    return

def run(rank, num_procs, device, master_port, args):
    mode = args.parse_string("mode", "train")
    num_envs = args.parse_int("num_envs", 1)
    visualize = args.parse_bool("visualize", True)
    logger_type = args.parse_string("logger", "txt")
    model_file = args.parse_string("model_file", "")

    out_dir = args.parse_string("out_dir", "output/")
    save_int_models = args.parse_bool("save_int_models", False)
    max_samples = args.parse_int("max_samples", np.iinfo(np.int64).max)
    max_iters = args.parse_int("max_iters", 30000)

    mp_util.init(rank, num_procs, device, master_port)

    set_rand_seed(args)
    set_np_formatting()
    create_output_dir(out_dir)

    env = build_env(args, num_envs, device, visualize)
    agent = build_agent(args, env, device)

    if (model_file != ""):
        agent.load(model_file)

    if (mode == "train"):
        save_config_files(args, out_dir)
        train(agent=agent, max_samples=max_samples, max_iters=max_iters, out_dir=out_dir, 
              save_int_models=save_int_models, logger_type=logger_type)
        
    elif (mode == "test"):
        test_episodes = args.parse_int("test_episodes", np.iinfo(np.int64).max)
        test(agent=agent, test_episodes=test_episodes)

    else:
        assert(False), "Unsupported mode: {}".format(mode)

    return

def main(argv):
    root_rank = 0
    args = load_args(argv)

    input_folder = args.parse_string("input_folder", "")
    if (input_folder != ""):
        out_dir_root = args.parse_string("out_dir", "output/")
        os.makedirs(out_dir_root, exist_ok=True)

        motion_files = [
            os.path.join(input_folder, f)
            for f in sorted(os.listdir(input_folder))
            if f.lower().endswith(".pkl")
        ]

        assert(len(motion_files) > 0), "No .pkl files found in input_folder: {}".format(input_folder)

        print("\n" + "=" * 60)
        print("🚀 BATCH TRAIN START")
        print("=" * 60)
        print("Input folder:  {}".format(input_folder))
        print("Output root:   {}".format(out_dir_root))
        print("Num motions:   {}".format(len(motion_files)))
        print("=" * 60 + "\n")

        base_tokens = argv[1:]

        def _remove_arg(tokens, key):
            out = []
            i = 0
            target = "--" + key
            while (i < len(tokens)):
                if (tokens[i] == target):
                    i += 1
                    while (i < len(tokens) and not tokens[i].startswith("--")):
                        i += 1
                else:
                    out.append(tokens[i])
                    i += 1
            return out

        base_tokens = _remove_arg(base_tokens, "input_folder")
        base_tokens = _remove_arg(base_tokens, "motion_file")
        base_tokens = _remove_arg(base_tokens, "out_dir")

        success = 0
        failed = []

        for i, motion_file in enumerate(motion_files):
            motion_name = os.path.splitext(os.path.basename(motion_file))[0]
            curr_out_dir = os.path.join(out_dir_root, motion_name)
            os.makedirs(curr_out_dir, exist_ok=True)

            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                *base_tokens,
                "--input_folder", "",
                "--motion_file", motion_file,
                "--out_dir", curr_out_dir,
            ]

            print("[{}/{}] Training motion: {}".format(i + 1, len(motion_files), motion_file))
            result = subprocess.run(cmd)

            if (result.returncode == 0):
                success += 1
            else:
                failed.append((motion_file, result.returncode))

        print("\n" + "=" * 60)
        print("📌 BATCH TRAIN SUMMARY")
        print("=" * 60)
        print("✅ Succeeded: {}/{}".format(success, len(motion_files)))
        print("❌ Failed:    {}/{}".format(len(failed), len(motion_files)))
        if (len(failed) > 0):
            print("-" * 60)
            for f, code in failed:
                print("• {} (exit={})".format(f, code))
        print("=" * 60 + "\n")
        return

    master_port = args.parse_int("master_port", None)
    devices = args.parse_strings("devices", ["cuda:0"])
    
    num_workers = len(devices)
    assert(num_workers > 0)
    
    # if master port is not specified, then pick a random one
    if (master_port is None):
        master_port = np.random.randint(6000, 7000)

    torch.multiprocessing.set_start_method("spawn")

    processes = []
    for rank in range(1, num_workers):
        curr_device = devices[rank]
        proc = torch.multiprocessing.Process(target=run, args=[rank, num_workers, curr_device, master_port, args])
        proc.start()
        processes.append(proc)
    
    root_device = devices[0]
    run(root_rank, num_workers, root_device, master_port, args)

    for proc in processes:
        proc.join()
       
    return

if __name__ == "__main__":
    main(sys.argv)