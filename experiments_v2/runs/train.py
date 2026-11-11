"""并行采样完整回合，训练只控制飞行的 MLP MAPPO。"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
from time import perf_counter
import tomllib
from types import SimpleNamespace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from environment_v2.settings import load_settings
from environment_v2.simulator import Simulator
from experiments_v2.runs.metrics import EpisodeMetrics, summarize
from methods_v2.baselines.policies import choose_greedy, choose_heft, sample_flight
from methods_v2.ers import ERS
from methods_v2.learning.mappo import MAPPO
from methods_v2.mappo.buffer import FlightRolloutBuffer
from methods_v2.mappo.features import FeatureBuilder
from methods_v2.mappo.policy import Policy
from methods_v2.mappo.reward import compute_reward


def initialize_worker(settings, args):
    """每个采样进程只建立一次环境、特征构造器和策略。"""
    global worker_state
    torch.set_num_threads(1)
    simulator = Simulator(settings)
    builder = FeatureBuilder(settings, ERS(simulator.router, settings), args.user_info)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.train_seed)
        policy = Policy(builder, args)
    worker_state = simulator, builder, policy, args


def tensor_state(state):
    """把普通 NumPy 参数副本转回 PyTorch state_dict。"""
    return {name: torch.from_numpy(value) for name, value in state.items()}


def numpy_state(module):
    """复制参数到普通 NumPy 数组，避免跨进程共享张量文件句柄。"""
    return {name: value.detach().cpu().numpy().copy()
            for name, value in module.state_dict().items()}


def collect_episode(job):
    """装载本批参数，用飞行 actor 和固定 HEFT 放置运行完整回合。"""
    started = perf_counter()
    seed, episode, actor, critic, deterministic = job
    simulator, builder, policy, args = worker_state
    policy.actor.load_state_dict(tensor_state(actor))
    policy.critic.load_state_dict(tensor_state(critic))
    generator = None
    if not deterministic:
        sampling_seed = np.random.SeedSequence([seed, episode]).generate_state(1, dtype=np.uint64)[0]
        generator = torch.Generator().manual_seed(int(sampling_seed))
    snapshot = simulator.reset(seed, episode)
    metrics, samples, rewards, values = EpisodeMetrics(), [], [], []
    policy.actor.eval()
    policy.critic.eval()
    while snapshot is not None:
        features = builder.build(snapshot)
        flight, sample, value = policy.act(features, generator, deterministic=deterministic)
        placements = choose_heft(snapshot, builder.ers)
        result, following = simulator.step(flight, placements)
        reward = compute_reward(result, simulator.settings.scheduling.deadline_s, args.expected_arrivals)
        metrics.add(snapshot, result, placements)
        metrics.reward += reward
        if not deterministic:
            samples.append(sample)
            rewards.append(reward)
            values.append(value)
        snapshot = following
    return dict(episode=episode, samples=samples, rewards=rewards, values=values,
                metrics=metrics, elapsed_s=perf_counter() - started)


def train(settings, args, updates, seed_output, emit, checkpoint):
    """使用常驻进程池采样与评估，按回合编号顺序更新。"""
    numpy_seed, torch_seed = np.random.SeedSequence(args.train_seed).spawn(2)
    rng = np.random.default_rng(numpy_seed)
    generator = torch.Generator().manual_seed(int(torch_seed.generate_state(1, dtype=np.uint64)[0]))
    builder = FeatureBuilder(
        settings, ERS(Simulator(settings).router, settings), args.user_info,
    )
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(generator.get_state())
        policy = Policy(builder, args)
    trainer = MAPPO(policy, args)
    episode, first_update = 0, 1
    if checkpoint is not None:
        policy.actor.load_state_dict(checkpoint["actor"])
        policy.critic.load_state_dict(checkpoint["critic"])
        trainer.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        trainer.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        trainer.value_normalizer.load_state_dict(checkpoint["value_normalizer"])
        rng.bit_generator.state = checkpoint["numpy_rng"]
        episode, first_update = checkpoint["next_episode"], checkpoint["update"] + 1
    started = perf_counter()
    with mp.get_context("spawn").Pool(args.rollout_workers, initializer=initialize_worker,
                                      initargs=(settings, args)) as pool:
        for update in range(first_update, updates + 1):
            buffer, metrics = FlightRolloutBuffer(args), []
            actor, critic = numpy_state(policy.actor), numpy_state(policy.critic)
            jobs = [(args.train_seed, index, actor, critic, False)
                    for index in range(episode, episode + args.episodes_per_update)]
            for result in pool.map(collect_episode, jobs, chunksize=1):
                item = result["metrics"]
                buffer.add_episode(result["samples"], result["rewards"], result["values"])
                metrics.append(item)
                emit(dict(kind="train_episode", update=update, seed=args.train_seed,
                          episode=result["episode"], elapsed_s=result["elapsed_s"], **item.report()))
            episode += args.episodes_per_update
            sampling_s = perf_counter() - started
            updating = perf_counter()
            losses = trainer.train(buffer, rng)
            update_s = perf_counter() - updating
            mean, variance = trainer.value_normalizer.running_mean_var()
            emit(dict(kind="update", update=update, elapsed_s=perf_counter() - started,
                      sampling_s=sampling_s, update_s=update_s,
                      flight_log_std=policy.actor.log_std_values().detach().cpu().tolist(),
                      **summarize(metrics), **losses,
                      valuenorm_mean=mean.tolist(), valuenorm_variance=variance.tolist()))
            if update % args.evaluation_interval == 0:
                evaluation = []
                actor, critic = numpy_state(policy.actor), numpy_state(policy.critic)
                jobs = [(args.evaluation_seed, index, actor, critic, True)
                        for index in range(args.evaluation_episodes)]
                for result in pool.map(collect_episode, jobs, chunksize=1):
                    item = result["metrics"]
                    evaluation.append(item)
                    emit(dict(kind="evaluation_episode", update=update, seed=args.evaluation_seed,
                              episode=result["episode"], elapsed_s=result["elapsed_s"], **item.report()))
                emit(dict(kind="evaluation", update=update, **summarize(evaluation)))
            torch.save(dict(update=update, next_episode=episode, args=vars(args),
                            actor=policy.actor.state_dict(), critic=policy.critic.state_dict(),
                            actor_optimizer=trainer.actor_optimizer.state_dict(),
                            critic_optimizer=trainer.critic_optimizer.state_dict(),
                            value_normalizer=trainer.value_normalizer.state_dict(),
                            numpy_rng=rng.bit_generator.state),
                       seed_output / "checkpoints" / f"checkpoint_{update:04d}.pt")
            started = perf_counter()


def check_reward(settings, args, emit):
    """贪心回合核对负奖励还原后的平均归一化时延。"""
    simulator = Simulator(settings)
    snapshot = simulator.reset(args.train_seed, 0)
    ers = ERS(simulator.router, settings)
    flight_rng = np.random.default_rng(np.random.SeedSequence([args.train_seed, 0, 101]))
    metrics = EpisodeMetrics()
    while snapshot is not None:
        placements = choose_greedy(snapshot, ers)
        result, following = simulator.step(sample_flight(snapshot, flight_rng), placements)
        metrics.add(snapshot, result, placements)
        metrics.reward += compute_reward(result, settings.scheduling.deadline_s, args.expected_arrivals)
        snapshot = following
    emit(dict(kind="reward_check", seed=args.train_seed, episode=0, offers=metrics.offers,
              restored_delay=-metrics.reward * args.expected_arrivals / metrics.offers,
              mean_normalized_delay=metrics.delay_s / (metrics.offers * settings.scheduling.deadline_s),
              invalid_placements=metrics.invalid))


def toml_value(value):
    """把方法配置中的标量写成 TOML。"""
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def copy_config(config, method_config, output, args):
    """保存运行时环境 TOML 和含命令行覆盖值的方法 TOML。"""
    target = output / "config"
    target.mkdir(parents=True, exist_ok=True)
    for path in config.glob("*.toml"):
        shutil.copy2(path, target / path.name)
    methods = target / "methods"
    methods.mkdir(exist_ok=True)
    values = vars(args)
    lines = []
    for line in method_config.read_text(encoding="utf-8").splitlines():
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        lines.append(f"{key} = {toml_value(values[key])}" if key in values else line)
    (methods / method_config.name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(name, command, output):
    """汇总当前实验目录中每个种子的最后一次更新、评估或奖励核对。"""
    seeds = {}
    for metrics_path in sorted(output.glob("seed*/metrics.jsonl")):
        latest = {}
        for line in metrics_path.open(encoding="utf-8"):
            row = json.loads(line)
            if row["kind"] in ("update", "evaluation", "reward_check"):
                latest[row["kind"]] = row
        seeds[metrics_path.parent.name] = latest
    (output / "summary.json").write_text(json.dumps(
        dict(name=name, command=command, seeds=seeds), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_args(options):
    """读取方法 TOML 或 checkpoint，并应用本次运行覆盖值。"""
    checkpoint = None
    if options.resume is None:
        with options.method_config.open("rb") as stream:
            args = SimpleNamespace(**tomllib.load(stream))
    else:
        checkpoint = torch.load(options.resume, map_location="cpu", weights_only=False)
        args = SimpleNamespace(**checkpoint["args"])
    if options.seed is not None:
        args.train_seed = options.seed
    if options.flight_actor is not None:
        args.flight_actor = options.flight_actor
    if options.user_info is not None:
        args.user_info = options.user_info
    if options.episodes_per_update is not None:
        args.episodes_per_update = options.episodes_per_update
    if options.rollout_workers is not None:
        args.rollout_workers = options.rollout_workers
    return args, checkpoint


def main():
    """选择正式训练、奖励核对或固定规模冒烟并写入对应目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "check-reward", "smoke"))
    parser.add_argument("--name")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config")
    parser.add_argument("--method-config", type=Path, default=PROJECT_ROOT / "config/methods/mappo.toml")
    parser.add_argument("--updates", type=int, help="累计更新次数目标，续训时也使用总次数")
    parser.add_argument("--episodes-per-update", type=int)
    parser.add_argument("--rollout-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--flight-actor", choices=("shared", "per_uav", "shared_id"))
    parser.add_argument("--user-info", choices=("full", "quadrant", "count"))
    parser.add_argument("--resume", type=Path, help="从 checkpoint 接着训练")
    options = parser.parse_args()
    if options.command == "train" and options.name is None:
        parser.error("train 需要 --name")
    if options.command == "train" and options.updates is None:
        parser.error("train 需要 --updates")
    args, checkpoint = load_args(options)
    if args.train_seed == 1:
        parser.error("训练种子 1 留给固定评估集")
    if options.command == "smoke":
        args.episodes_per_update = 2
        args.rollout_workers = 2
        options.updates = 2
    torch.set_num_threads(args.training_threads)
    settings = load_settings(options.config)
    if options.command == "train":
        name = options.name
        output = PROJECT_ROOT / "output/runs" / name
    elif options.command == "check-reward":
        name = "reward"
        output = PROJECT_ROOT / "output/checks/reward"
        if output.exists():
            shutil.rmtree(output)
    else:
        name = f"smoke-{args.flight_actor}"
        output = PROJECT_ROOT / "output/checks/smoke" / args.flight_actor
        if output.exists():
            shutil.rmtree(output)
    seed_output = output / f"seed{args.train_seed}"
    (seed_output / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    copy_config(options.config, options.method_config, output, args)
    metrics_path = seed_output / "metrics.jsonl"
    mode = "a" if options.resume is not None else "w"
    with metrics_path.open(mode, encoding="utf-8") as log:
        def emit(row):
            line = json.dumps(row, ensure_ascii=False)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        emit(dict(kind="configuration", name=name, command=options.command,
                  output=str(output), method=vars(args), ppo_action_fields=["flight"],
                  rollout_sample_fields=["observations", "state", "flight", "log_probs"],
                  environment={key: vars(value) for key, value in settings._asdict().items()}))
        if options.command in ("train", "smoke"):
            train(settings, args, options.updates, seed_output, emit, checkpoint)
        else:
            check_reward(settings, args, emit)
    write_summary(name, options.command, output)


if __name__ == "__main__":
    main()
