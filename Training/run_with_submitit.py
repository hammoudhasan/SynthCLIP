# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
A script to run multinode training with submitit.
"""
import argparse
import os
import uuid
from pathlib import Path

import main as main_slip
import submitit


def parse_args():
    parser = main_slip.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for SLIP pre-training", parents=[parser])
    parser.add_argument(
        "--ngpus", default=2, type=int, help="Number of gpus to request on each node"
    )
    parser.add_argument(
        "--nodes", default=8, type=int, help="Number of nodes to request"
    )
    parser.add_argument("--timeout", default=1440, type=int, help="Duration of the job")
    parser.add_argument(
        "--job_dir",
        default="slurm",
        type=str,
        help="Job dir. Leave empty for automatic.",
    )
    parser.add_argument("--gpu_type", default="v100", type=str, help="GPU name.")

    return parser.parse_args()


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs("output", exist_ok=True)
    init_file = Path(os.path.abspath("output")) / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main as main_slip

        self._setup_gpu_args()
        main_slip.main(self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file().as_uri()
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(
            str(self.args.output_dir).replace("%j", str(job_env.job_id))
        )
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()
    os.makedirs(args.job_dir, exist_ok=True)

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    kwargs = {}
    kwargs["slurm_constraint"] = args.gpu_type
    kwargs["slurm_setup"] = ["module load cuda/11.8"]

    executor.update_parameters(
        mem_gb=48 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=6,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_signal_delay_s=120,
        **kwargs,
    )

    executor.update_parameters(name="synthclip")

    args.dist_url = get_init_file().as_uri()

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()
