# -*- coding: utf-8 -*-
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
RUN = os.path.join(ROOT, "totol", "run_agent_al.py")
ENV = os.environ.get("CONDA_ENV", "gnot_cuda11")


def run(cmd: str):
    print(f"$ {cmd}")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print(f"Command failed: {cmd}")
        sys.exit(ret.returncode)


def main():
    data_dir = "/home/v-wenliao/gnot/GNOT/data/al_bz"
    os.makedirs(data_dir, exist_ok=True)

    # 1) 生成合成数据 + bandit
    # Use conda to ensure correct environment
    run(f"conda run -n {ENV} python {RUN} --data_update_dir {data_dir} --create_synth --mode bandit --select_num 20")

    # 2) LLM回退（无API）模式
    run(f"conda run -n {ENV} python {RUN} --data_update_dir {data_dir} --mode llm --select_num 20")

    # 3) 混合模式
    run(f"conda run -n {ENV} python {RUN} --data_update_dir {data_dir} --mode hybrid --select_num 20")


if __name__ == '__main__':
    main()
