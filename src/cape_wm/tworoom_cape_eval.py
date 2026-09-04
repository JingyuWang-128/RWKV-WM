from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import binomtest

from .adapters.lewm import LeWorldModelAdapter
from .baseline import (
    _final_positions,
    _package_version,
    _write_json,
    prepare_resources,
    select_pairs,
)
from .calibration import load_duration_calibrator
from .cem import CEMConfig
from .checkpoints import load_risk_model
from .comparison_eval import _assert_reference_pairs, _load_macro, _load_trm
from .config import load_config
from .generators import (
    AnchoredMacroGenerator,
    CEMLowLevelController,
    DirectRolloutGenerator,
    MacroCEMGenerator,
    OfficialCEMLowLevelController,
    TopologyWaypointGenerator,
    TrustedAnchorGenerator,
)
from .planner import CAPEPlanner, PlannerConfig
from .policy import StableWorldModelCAPEPolicy
from .reachability import (
    CalibratedReachabilityScorer,
    MultiHorizonReachabilityCEM,
    ReachabilityAdvantageCalibrator,
    ReachabilityAdvantagePlanner,
)
from .risk import ConstantRiskEstimator, DurationCalibratedRiskEstimator
from .torch_wrappers import (
    TorchMacroDynamics,
    TorchPairwiseHybridCost,
    TorchRiskPredictor,
)
from .tworoom_risk_collection import select_split_attempts


def _diagnostic_summary(policy: StableWorldModelCAPEPolicy) -> dict[str, Any]:
    histories = [item for history in policy.diagnostic_history for item in history]
    events = Counter(item.event.value for item in histories)
    triggers = Counter(
        item.trigger_event.value for item in histories if item.trigger_event is not None
    )
    durations = Counter(
        str(item.chosen_duration) for item in histories if item.chosen_duration is not None
    )
    anchor_gaps: list[float] = []
    anchor_decisions = 0
    macro_overrides = 0
    for item in histories:
        anchors = [candidate for candidate in item.candidates if candidate.get("trusted_anchor")]
        if not anchors:
            continue
        anchor_decisions += 1
        anchor_score = max(float(candidate["score"]) for candidate in anchors)
        macro_scores = [
            float(candidate["score"])
            for candidate in item.candidates
            if not candidate.get("trusted_anchor") and candidate.get("feasible")
        ]
        if macro_scores:
            anchor_gaps.append(max(macro_scores) - anchor_score)
        if item.chosen_duration != int(anchors[0]["duration"]):
            macro_overrides += 1
    return {
        "decisions": len(histories),
        "events": dict(sorted(events.items())),
        "triggers": dict(sorted(triggers.items())),
        "chosen_durations": dict(sorted(durations.items())),
        "fallback_decisions": int(sum(item.fallback_used for item in histories)),
        "candidate_decisions": int(sum(item.candidate_count > 0 for item in histories)),
        "feasible_candidates": int(sum(item.feasible_count for item in histories)),
        "anchor_decisions": anchor_decisions,
        "macro_override_decisions": macro_overrides,
        "macro_anchor_score_gap_quantiles": (
            {
                str(quantile): float(np.quantile(anchor_gaps, quantile))
                for quantile in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
            }
            if anchor_gaps
            else {}
        ),
        "reported_model_transitions": float(sum(item.step_planning_cost for item in histories)),
    }


def run_tworoom_cape(args: argparse.Namespace, resources: Any | None = None) -> dict[str, Any]:
    import stable_worldmodel as swm

    if args.audit_output is not None and args.candidate_records_output is not None:
        raise ValueError("audit and candidate collection outputs are mutually exclusive")
    if args.candidate_records_output is not None and (
        args.planner_mode != "cape" or args.split_name not in {"train", "calibration"}
    ):
        raise ValueError("candidate collection requires CAPE mode and a train/calibration split")
    if args.craft_records_output is not None and args.planner_mode != "craft":
        raise ValueError("CRAFT progress collection requires CRAFT planner mode")

    experiment_config = load_config(args.experiment_config)
    phase_config = load_config(args.phase_config)
    comparison_config = load_config(args.comparison_config)
    experiment = dict(experiment_config["experiment"])
    experiment.update(
        {
            "name": "cape_wm_tworooms",
            "selection_seed": int(args.selection_seed),
            "goal_offset": int(args.goal_offset),
            "eval_budget": int(args.eval_budget),
            "num_eval": int(args.num_eval),
        }
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    resources = resources or prepare_resources(
        data=args.data,
        weights=args.weights,
        experiment_config=args.experiment_config,
        runtime_config=args.runtime_config,
        cache_dir=output / "cache",
        device_override=args.device,
    )
    if args.split_name is None:
        pairs, selected_rows = select_pairs(
            resources.dataset.get_col_data("ep_idx"),
            resources.dataset.get_col_data("step_idx"),
            resources.dataset.lengths,
            goal_offset=int(experiment["goal_offset"]),
            num_eval=int(experiment["num_eval"]),
            seed=int(experiment["selection_seed"]),
        )
    else:
        if not args.skip_reference_check:
            raise ValueError("selection-split runs must use --skip-reference-check")
        split = json.loads(args.split_manifest.read_text())
        attempts = select_split_attempts(
            resources.dataset.get_col_data("ep_idx"),
            resources.dataset.get_col_data("step_idx"),
            resources.dataset.lengths,
            split[args.split_name],
            duration=int(experiment["goal_offset"]),
            count=int(experiment["num_eval"]),
            seed=int(experiment["selection_seed"]),
        )
        from .baseline import DatasetPair

        pairs = [
            DatasetPair(
                pair_id=item.record_id.replace("tworoom-risk", f"tworoom-{args.split_name}"),
                episode_index=item.episode_index,
                start_step=item.start_step,
                goal_offset=item.duration,
            )
            for item in attempts
        ]
        selected_rows = np.empty(0, dtype=np.int64)
    reference = (
        None
        if args.skip_reference_check
        else _assert_reference_pairs(
            args.reference_matrix,
            int(experiment["goal_offset"]),
            int(experiment["selection_seed"]),
            pairs,
        )
    )
    _write_json(
        output / "pairs.json",
        {
            "selection_seed": int(experiment["selection_seed"]),
            "selected_rows": selected_rows.tolist(),
            "reference_flat_mpc_manifest": reference,
            "pairs": [asdict(pair) for pair in pairs],
        },
    )

    trm = None
    trm_payload = None
    if args.planner_mode != "craft":
        trm, trm_payload = _load_trm(args.checkpoint_dir / "trm.pt", resources.device)
    macro = None
    macro_payload = None
    risk = None
    if args.planner_mode == "cape":
        macro_checkpoint = (
            args.macro_checkpoint
            if args.macro_checkpoint is not None
            else args.checkpoint_dir / "macro.pt"
        )
        macro, macro_payload = _load_macro(macro_checkpoint, resources.device)
        risk_model, risk_config = load_risk_model(args.risk_checkpoint, resources.device)
        risk_predictor = TorchRiskPredictor(
            risk_model,
            resources.device,
            miss_scale=float(risk_config.get("miss_scale", 1.0)),
        )
        probability_thresholds = None
        if args.probability_only_filter:
            if args.probability_calibration_artifact is None:
                raise ValueError(
                    "--probability-only-filter requires --probability-calibration-artifact"
                )
            probability_payload = json.loads(args.probability_calibration_artifact.read_text())
            probability_thresholds = {
                int(duration): float(item["threshold"])
                for duration, item in probability_payload["by_duration"].items()
            }
        risk = DurationCalibratedRiskEstimator(
            risk_predictor,
            load_duration_calibrator(args.calibration_artifact),
            probability_thresholds=probability_thresholds,
        )
    goal_metric = None
    if trm is not None:
        trm_cfg = comparison_config["trm"]
        goal_metric = TorchPairwiseHybridCost(
            trm,
            trm_weight=float(trm_cfg["hybrid_trm_weight"]),
            l2_weight=float(trm_cfg["hybrid_l2_weight"]),
            device=resources.device,
        )
    action_scaler = resources.process["action"]
    action_low = action_scaler.transform(np.full((1, 2), -1.0, dtype=np.float32))[0]
    action_high = action_scaler.transform(np.full((1, 2), 1.0, dtype=np.float32))[0]
    action_block = int(experiment_config["planner"]["action_block"])
    adapter = LeWorldModelAdapter(
        resources.model,
        action_shape=(2,),
        transform=resources.image_transform,
        goal_metric=goal_metric,
        device=resources.device,
        action_block=action_block,
        action_low=action_low,
        action_high=action_high,
    )
    macro_dynamics = (
        TorchMacroDynamics(
            macro,
            macro_dim=int(comparison_config["macro"]["macro_dim"]),
            device=resources.device,
        )
        if macro is not None
        else None
    )
    phase_planner = phase_config["planner"]
    low_cem_cfg = phase_config["cem"]["low_level"]
    macro_cem_cfg = phase_config["cem"]["macro"]
    empirical_macro_banks = None
    if args.macro_bank is not None:
        archive = np.load(args.macro_bank)
        empirical_macro_banks = {
            int(duration): np.asarray(archive[duration], dtype=np.float32)
            for duration in archive.files
        }
    topology_payload = (
        np.load(args.topology_artifact) if args.topology_artifact is not None else None
    )
    craft_model = None
    craft_calibrator = None
    craft_config = None
    if args.planner_mode == "craft":
        from .models import (
            ContinuousReachabilityDistribution,
            DirectedReachabilityDistribution,
        )

        craft_payload = torch.load(
            args.craft_checkpoint, map_location=resources.device, weights_only=False
        )
        craft_model_config = craft_payload["config"]
        if craft_model_config.get("family", "categorical") == "continuous_lognormal":
            craft_model = ContinuousReachabilityDistribution(
                latent_dim=int(craft_model_config["latent_dim"]),
                horizon_bins=tuple(map(int, craft_model_config["horizon_bins"])),
                hidden_dim=int(craft_model_config["hidden_dim"]),
                depth=int(craft_model_config["depth"]),
                min_log_scale=float(craft_model_config["min_log_scale"]),
                max_log_scale=float(craft_model_config["max_log_scale"]),
            ).to(resources.device)
        else:
            craft_model = DirectedReachabilityDistribution(
                latent_dim=int(craft_model_config["latent_dim"]),
                horizon_bins=tuple(map(int, craft_model_config["horizon_bins"])),
                hidden_dim=int(craft_model_config["hidden_dim"]),
                depth=int(craft_model_config["depth"]),
                overflow_horizon=int(craft_model_config["overflow_horizon"]),
            ).to(resources.device)
        craft_model.load_state_dict(craft_payload["modules"]["reachability"])
        craft_model.eval().requires_grad_(False)
        craft_config = load_config(args.craft_config)
        durations = tuple(map(int, craft_config["calibration"]["durations"]))
        if args.craft_calibration is not None:
            craft_calibrator = ReachabilityAdvantageCalibrator.load(args.craft_calibration)
        elif args.craft_uncalibrated_smoke:
            repeated = np.repeat(np.asarray(durations, dtype=np.int64), 2)
            craft_calibrator = ReachabilityAdvantageCalibrator(
                alpha=float(craft_config["calibration"]["alpha"])
            ).fit(np.zeros(len(repeated)), np.zeros(len(repeated)), repeated)
        else:
            raise ValueError(
                "CRAFT requires --craft-calibration; use --craft-uncalibrated-smoke only "
                "for non-reportable diagnostics"
            )
    planner_index = 0
    upstream_cem = experiment_config["cem"]
    degenerate_duration = int(experiment_config["planner"]["horizon"]) * action_block
    shared_generator = torch.Generator(device=resources.device).manual_seed(
        int(experiment["selection_seed"])
    )

    def planner_factory() -> CAPEPlanner | ReachabilityAdvantagePlanner:
        nonlocal planner_index
        seed = int(experiment["selection_seed"]) * 10_000 + planner_index
        planner_index += 1
        if args.planner_mode == "craft":
            assert craft_model is not None
            assert craft_calibrator is not None
            assert craft_config is not None
            craft_planner = craft_config["planner"]
            scorer = CalibratedReachabilityScorer(
                craft_model,
                craft_calibrator,
                tuple(map(int, craft_config["calibration"]["durations"])),
            )
            controller = MultiHorizonReachabilityCEM(
                adapter,
                scorer,
                action_block=int(craft_planner["action_block"]),
                samples=int(craft_planner["samples"]),
                elites=int(craft_planner["elites"]),
                iterations=int(craft_planner["iterations"]),
                variance_scale=float(craft_planner["variance_scale"]),
                seed=seed,
                device=resources.device,
            )
            return ReachabilityAdvantagePlanner(adapter, controller)
        if args.planner_mode == "flat_trm_degenerate":
            low_level = OfficialCEMLowLevelController(
                adapter,
                horizon=degenerate_duration,
                action_block=action_block,
                samples=int(upstream_cem["num_samples"]),
                elites=int(upstream_cem["topk"]),
                iterations=int(upstream_cem["iterations"]),
                variance_scale=float(upstream_cem["variance_scale"]),
                generator=shared_generator,
            )
            return CAPEPlanner(
                adapter,
                DirectRolloutGenerator(adapter, low_level),
                ConstantRiskEstimator(tube_threshold=float("inf")),
                PlannerConfig(
                    durations=(degenerate_duration,),
                    alpha=float(phase_planner["alpha"]),
                    executability_threshold=float("inf"),
                    goal_threshold=-1.0,
                    subgoal_threshold=-1.0,
                    lambda_compute=0.0,
                    lambda_risk=0.0,
                    minimum_candidate_progress=float("-inf"),
                    replan_low_level=False,
                    adaptive_duration=False,
                    risk_filter_enabled=False,
                    tube_trigger_enabled=False,
                    stall_trigger_enabled=False,
                    subgoal_trigger_enabled=False,
                    remaining_risk_trigger_enabled=False,
                    monitor_interval=action_block,
                ),
            )
        if args.planner_mode == "topology_trm":
            if topology_payload is None:
                raise ValueError("topology_trm mode requires --topology-artifact")
            generator = TrustedAnchorGenerator(
                adapter,
                OfficialCEMLowLevelController(
                    adapter,
                    horizon=degenerate_duration,
                    action_block=action_block,
                    samples=int(upstream_cem["num_samples"]),
                    elites=int(upstream_cem["topk"]),
                    iterations=int(upstream_cem["iterations"]),
                    variance_scale=float(upstream_cem["variance_scale"]),
                    generator=shared_generator,
                ),
                duration=degenerate_duration,
            )
            generator = TopologyWaypointGenerator(
                generator,
                adapter,
                None,
                duration=int(args.topology_duration),
                latent_mean=topology_payload["latent_mean"],
                latent_scale=topology_payload["latent_scale"],
                position_weight=topology_payload["position_weight"],
                position_mean=topology_payload["position_mean"],
                wall_position=float(topology_payload["wall_position"]),
                door_position=float(topology_payload["door_position"]),
                side_offset=float(topology_payload["side_offset"]),
                door_left_latent=topology_payload["door_left_latent"],
                door_right_latent=topology_payload["door_right_latent"],
                direct_action_transform=action_scaler.transform,
                direct_goal_completion=bool(args.topology_direct_completion),
            )
            return CAPEPlanner(
                adapter,
                generator,
                ConstantRiskEstimator(tube_threshold=float("inf")),
                PlannerConfig(
                    durations=(degenerate_duration,),
                    alpha=float(phase_planner["alpha"]),
                    executability_threshold=float("inf"),
                    goal_threshold=-1.0,
                    subgoal_threshold=-1.0,
                    lambda_compute=0.0,
                    lambda_risk=0.0,
                    minimum_candidate_progress=float("-inf"),
                    replan_low_level=False,
                    adaptive_duration=False,
                    risk_filter_enabled=False,
                    tube_trigger_enabled=False,
                    stall_trigger_enabled=False,
                    subgoal_trigger_enabled=False,
                    remaining_risk_trigger_enabled=False,
                    monitor_interval=action_block,
                ),
            )
        if args.macro_low_level_mode == "official":
            low_level = OfficialCEMLowLevelController(
                adapter,
                horizon=max(map(int, phase_planner["durations"])),
                action_block=action_block,
                samples=int(upstream_cem["num_samples"]),
                elites=int(upstream_cem["topk"]),
                iterations=int(upstream_cem["iterations"]),
                variance_scale=float(upstream_cem["variance_scale"]),
                generator=torch.Generator(device=resources.device).manual_seed(seed + 1_000_000),
                fixed_horizon=False,
            )
        else:
            low_level = CEMLowLevelController(
                adapter,
                horizon=max(map(int, phase_planner["durations"])),
                cem=CEMConfig(
                    samples=int(low_cem_cfg["samples"]),
                    elites=int(low_cem_cfg["elites"]),
                    iterations=int(low_cem_cfg["iterations"]),
                    seed=seed,
                ),
            )
        assert macro_dynamics is not None and risk is not None
        generator = MacroCEMGenerator(
            adapter,
            macro_dynamics,
            low_level,
            physical_horizon=int(phase_planner["physical_horizon"]),
            macro_cem=CEMConfig(
                samples=int(macro_cem_cfg["samples"]),
                elites=int(macro_cem_cfg["elites"]),
                iterations=int(macro_cem_cfg["iterations"]),
                seed=seed,
                clip=empirical_macro_banks is None,
            ),
            model_transition_budget=(
                None
                if args.macro_low_level_mode == "official"
                else int(phase_planner["model_transition_budget_per_decision"])
            ),
            high_level_budget_fraction=float(phase_planner["high_level_budget_fraction"]),
            fallback_budget_fraction=float(phase_planner["fallback_budget_fraction"]),
            empirical_banks=empirical_macro_banks,
            empirical_residual_scale=float(args.macro_bank_residual_scale),
            empirical_discrete=bool(args.macro_bank_discrete),
        )
        if args.flat_trm_anchor:
            generator = AnchoredMacroGenerator(
                generator,
                OfficialCEMLowLevelController(
                    adapter,
                    horizon=degenerate_duration,
                    action_block=action_block,
                    samples=int(upstream_cem["num_samples"]),
                    elites=int(upstream_cem["topk"]),
                    iterations=int(upstream_cem["iterations"]),
                    variance_scale=float(upstream_cem["variance_scale"]),
                    generator=shared_generator,
                ),
                anchor_duration=degenerate_duration,
            )
        if topology_payload is not None:
            topology_controller = (
                OfficialCEMLowLevelController(
                    adapter,
                    horizon=int(args.topology_duration),
                    action_block=action_block,
                    samples=int(upstream_cem["num_samples"]),
                    elites=int(upstream_cem["topk"]),
                    iterations=int(upstream_cem["iterations"]),
                    variance_scale=float(upstream_cem["variance_scale"]),
                    generator=torch.Generator(device=resources.device).manual_seed(
                        seed + 2_000_000
                    ),
                )
                if args.topology_control_mode == "cem"
                else None
            )
            generator = TopologyWaypointGenerator(
                generator,
                adapter,
                topology_controller,
                duration=int(args.topology_duration),
                latent_mean=topology_payload["latent_mean"],
                latent_scale=topology_payload["latent_scale"],
                position_weight=topology_payload["position_weight"],
                position_mean=topology_payload["position_mean"],
                wall_position=float(topology_payload["wall_position"]),
                door_position=float(topology_payload["door_position"]),
                side_offset=float(topology_payload["side_offset"]),
                door_left_latent=topology_payload["door_left_latent"],
                door_right_latent=topology_payload["door_right_latent"],
                direct_action_transform=(
                    action_scaler.transform if args.topology_control_mode == "direct" else None
                ),
                direct_goal_completion=bool(args.topology_direct_completion),
            )
        return CAPEPlanner(
            adapter,
            generator,
            risk,
            PlannerConfig(
                durations=tuple(map(int, phase_planner["durations"])),
                alpha=float(phase_planner["alpha"]),
                executability_threshold=float(experiment["success_threshold"]),
                goal_threshold=-1.0,
                subgoal_threshold=float(phase_planner["subgoal_threshold"]),
                lambda_compute=float(args.lambda_compute),
                lambda_risk=float(args.lambda_risk),
                compute_cost_scale=float(phase_planner["compute_cost_scale"]),
                stall_patience=int(phase_planner["stall_patience"]),
                successes_to_expand=int(phase_planner["successes_to_expand"]),
                replan_low_level=False,
                adaptive_duration=bool(phase_planner["adaptive_duration"]),
                risk_filter_enabled=(
                    bool(phase_planner["risk_filter_enabled"]) and not args.disable_risk_filter
                ),
                endpoint_bound_filter_enabled=not args.probability_only_filter,
                tube_trigger_enabled=bool(phase_planner["tube_trigger_enabled"]),
                stall_trigger_enabled=bool(phase_planner["stall_trigger_enabled"]),
                subgoal_trigger_enabled=bool(phase_planner["subgoal_trigger_enabled"]),
                remaining_risk_trigger_enabled=bool(phase_planner["remaining_risk_trigger_enabled"])
                and not args.disable_risk_filter
                and not args.probability_only_filter,
                minimum_success_probability=float(phase_planner["minimum_success_probability"]),
                anchor_advantage_margin=float(args.anchor_advantage_margin),
                monitor_interval=action_block,
            ),
        )

    policy_kwargs = {
        "planner_factory": planner_factory,
        "num_envs": int(experiment["num_eval"]),
        "action_postprocess": lambda value: action_scaler.inverse_transform(value),
    }
    if args.candidate_records_output is not None:
        from .tworoom_audit import LatentPositionIndex
        from .tworoom_candidate_collection import TwoRoomCandidateCollectionPolicy

        policy = TwoRoomCandidateCollectionPolicy(
            **policy_kwargs,
            position_index=LatentPositionIndex(
                args.audit_cache, resources.data_path, resources.device
            ),
            action_scaler=action_scaler,
            action_block=action_block,
            record_prefix=(
                f"{args.split_name}:seed{experiment['selection_seed']}:"
                f"offset{experiment['goal_offset']}"
            ),
            success_threshold=float(experiment["success_threshold"]),
        )
    elif args.audit_output is None:
        policy = StableWorldModelCAPEPolicy(**policy_kwargs)
    else:
        from .tworoom_audit import LatentPositionIndex, TwoRoomSameCandidateAuditPolicy

        policy = TwoRoomSameCandidateAuditPolicy(
            **policy_kwargs,
            position_index=LatentPositionIndex(
                args.audit_cache, resources.data_path, resources.device
            ),
            action_scaler=action_scaler,
            success_threshold=float(experiment["success_threshold"]),
        )
    world = swm.World(
        env_name=str(experiment["environment"]),
        num_envs=int(experiment["num_eval"]),
        max_episode_steps=2 * int(experiment["eval_budget"]),
        image_shape=(int(experiment["image_size"]), int(experiment["image_size"])),
    )
    world.set_policy(policy)
    if resources.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resources.device)
    started_at = time.time()
    wall_start = time.perf_counter()
    try:
        metrics = world.evaluate(
            dataset=resources.dataset,
            episodes_idx=[pair.episode_index for pair in pairs],
            start_steps=[pair.start_step for pair in pairs],
            goal_offset=int(experiment["goal_offset"]),
            eval_budget=int(experiment["eval_budget"]),
            callables=[
                {"method": "_set_state", "args": {"state": {"value": "pos_agent"}}},
                {
                    "method": "_set_goal_state",
                    "args": {"goal_state": {"value": "goal_pos_agent"}},
                },
            ],
            video=None,
        )
        wall_seconds = time.perf_counter() - wall_start
        agents, goals, distances = _final_positions(world)
    finally:
        world.close()
    successes = np.asarray(metrics["episode_successes"], dtype=bool)
    interval = binomtest(int(successes.sum()), len(successes)).proportion_ci(
        confidence_level=0.95, method="exact"
    )
    with (output / "episodes.jsonl").open("w") as handle:
        for index, pair in enumerate(pairs):
            history = policy.diagnostic_history[index]
            handle.write(
                json.dumps(
                    {
                        **asdict(pair),
                        "selection_seed": int(experiment["selection_seed"]),
                        "success": bool(successes[index]),
                        "steps": int(policy.steps[index]),
                        "final_state_error": float(distances[index]),
                        "final_agent_position": agents[index],
                        "goal_position": goals[index],
                        "fallback_decisions": int(sum(item.fallback_used for item in history)),
                        "trigger_events": [
                            item.trigger_event.value
                            for item in history
                            if item.trigger_event is not None
                        ],
                        "chosen_durations": [
                            item.chosen_duration
                            for item in history
                            if item.chosen_duration is not None
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    craft_records: list[dict[str, Any]] = []
    if args.craft_records_output is not None:
        for environment_index, (pair, planner) in enumerate(
            zip(pairs, policy.planners, strict=True)
        ):
            for record in getattr(planner, "calibration_records", []):
                craft_records.append(
                    {
                        "pair_id": pair.pair_id,
                        "environment_index": environment_index,
                        **record,
                    }
                )
        args.craft_records_output.parent.mkdir(parents=True, exist_ok=True)
        with args.craft_records_output.open("w") as handle:
            for record in craft_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "status": "complete",
        "method": (
            "craft_wm"
            if args.planner_mode == "craft"
            else (
                "cape_wm"
                if args.planner_mode == "cape"
                else (
                    "topology_trm"
                    if args.planner_mode == "topology_trm"
                    else "cape_flat_trm_degenerate"
                )
            )
        ),
        "experiment": experiment,
        "selected_hyperparameters": {
            "lambda_compute": float(args.lambda_compute),
            "lambda_risk": float(args.lambda_risk),
            "risk_filter_enabled": (args.planner_mode == "cape" and not args.disable_risk_filter),
            "remaining_risk_trigger_enabled": (
                args.planner_mode == "cape" and not args.disable_risk_filter
            ),
            "planner_mode": args.planner_mode,
            "probability_only_filter": bool(args.probability_only_filter),
            "flat_trm_anchor": bool(args.flat_trm_anchor),
            "anchor_advantage_margin": float(args.anchor_advantage_margin),
            "macro_low_level_mode": args.macro_low_level_mode,
            "macro_bank_discrete": bool(args.macro_bank_discrete),
            "topology_waypoint": args.topology_artifact is not None,
            "topology_duration": int(args.topology_duration),
            "topology_control_mode": args.topology_control_mode,
            "topology_direct_completion": bool(args.topology_direct_completion),
            "craft_uncalibrated_smoke": bool(args.craft_uncalibrated_smoke),
        },
        "data_split": args.split_name or "final_test_pairs",
        "runtime": {
            "requested_device": resources.requested_device,
            "resolved_device": str(resources.device),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "stable_worldmodel": _package_version("stable-worldmodel"),
        },
        "assets": {
            "dataset": str(resources.data_path),
            "weights": str(resources.weights_path),
            "trm_checkpoint": (
                str((args.checkpoint_dir / "trm.pt").resolve())
                if trm_payload is not None
                else None
            ),
            "trm_best_epoch": (
                trm_payload["best_epoch"] if trm_payload is not None else None
            ),
            "macro_checkpoint": (
                str(macro_checkpoint.resolve()) if macro_payload is not None else None
            ),
            "macro_best_epoch": (
                macro_payload["best_epoch"] if macro_payload is not None else None
            ),
            "risk_checkpoint": (
                str(args.risk_checkpoint.resolve()) if args.planner_mode == "cape" else None
            ),
            "calibration_artifact": (
                str(args.calibration_artifact.resolve()) if args.planner_mode == "cape" else None
            ),
            "probability_calibration_artifact": (
                str(args.probability_calibration_artifact.resolve())
                if args.probability_calibration_artifact is not None
                else None
            ),
            "macro_bank": (str(args.macro_bank.resolve()) if args.macro_bank is not None else None),
            "topology_artifact": (
                str(args.topology_artifact.resolve())
                if args.topology_artifact is not None
                else None
            ),
            "craft_checkpoint": (
                str(args.craft_checkpoint.resolve()) if args.planner_mode == "craft" else None
            ),
            "craft_calibration": (
                str(args.craft_calibration.resolve())
                if args.craft_calibration is not None
                else None
            ),
            "reference_pairs": reference,
        },
        "results": {
            "num_tasks": len(successes),
            "successes": int(successes.sum()),
            "success_rate": float(successes.mean()),
            "success_rate_percent": 100.0 * float(successes.mean()),
            "success_rate_exact_95_ci": [float(interval.low), float(interval.high)],
            "mean_final_state_error": float(np.mean(distances)),
            "mean_steps": float(np.mean(policy.steps)),
            "wall_time_seconds": wall_seconds,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(resources.device))
                if resources.device.type == "cuda"
                else 0
            ),
        },
        "planning": _diagnostic_summary(policy),
        "candidate_collection": (
            {
                "records": len(policy.candidate_records),
                "output": str(args.candidate_records_output.resolve()),
            }
            if args.candidate_records_output is not None
            else None
        ),
        "craft_progress_collection": (
            {
                "records": len(craft_records),
                "output": str(args.craft_records_output.resolve()),
            }
            if args.craft_records_output is not None
            else None
        ),
        "started_at_unix": started_at,
        "finished_at_unix": time.time(),
    }
    _write_json(output / "summary.json", summary)
    if args.audit_output is not None:
        policy.save_audit(args.audit_output)
    if args.candidate_records_output is not None:
        policy.save_candidates(args.candidate_records_output)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paired Two-Room CAPE-WM evaluation")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints/tworoom_baselines")
    )
    parser.add_argument("--macro-checkpoint", type=Path)
    parser.add_argument(
        "--risk-checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/tworoom_cape/risk_scaled.pt"),
    )
    parser.add_argument(
        "--calibration-artifact",
        type=Path,
        default=Path("artifacts/checkpoints/tworoom_cape/risk_scaled.duration_calibration.json"),
    )
    parser.add_argument("--probability-calibration-artifact", type=Path)
    parser.add_argument("--macro-bank", type=Path)
    parser.add_argument("--macro-bank-residual-scale", type=float, default=0.25)
    parser.add_argument("--macro-bank-discrete", action="store_true")
    parser.add_argument("--topology-artifact", type=Path)
    parser.add_argument("--topology-duration", type=int, default=15)
    parser.add_argument("--topology-control-mode", choices=("cem", "direct"), default="cem")
    parser.add_argument("--topology-direct-completion", action="store_true")
    parser.add_argument(
        "--craft-checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/tworoom_craft/reachability.pt"),
    )
    parser.add_argument("--craft-calibration", type=Path)
    parser.add_argument(
        "--craft-config", type=Path, default=Path("configs/craft_phase_b.yaml")
    )
    parser.add_argument(
        "--craft-uncalibrated-smoke",
        action="store_true",
        help="non-reportable diagnostic with zero conformal correction",
    )
    parser.add_argument("--flat-trm-anchor", action="store_true")
    parser.add_argument("--anchor-advantage-margin", type=float, default=0.0)
    parser.add_argument(
        "--macro-low-level-mode",
        choices=("budgeted", "official"),
        default="budgeted",
    )
    parser.add_argument(
        "--reference-matrix",
        type=Path,
        default=Path("artifacts/results/lewm_tworooms_long_matrix_gpu_final"),
    )
    parser.add_argument(
        "--experiment-config", type=Path, default=Path("configs/lewm_tworooms_baseline.yaml")
    )
    parser.add_argument("--phase-config", type=Path, default=Path("configs/cape_phase_b.yaml"))
    parser.add_argument(
        "--comparison-config", type=Path, default=Path("configs/tworoom_baselines.yaml")
    )
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/runtime.yaml"))
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--goal-offset", type=int, default=100)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--num-eval", type=int, default=100)
    parser.add_argument("--lambda-compute", type=float, default=0.01)
    parser.add_argument("--lambda-risk", type=float, default=0.01)
    parser.add_argument(
        "--planner-mode",
        choices=("cape", "flat_trm_degenerate", "topology_trm", "craft"),
        default="cape",
    )
    parser.add_argument(
        "--disable-risk-filter",
        action="store_true",
        help="validation-only diagnostic ablation; do not use for frozen final runs",
    )
    parser.add_argument("--probability-only-filter", action="store_true")
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("artifacts/cache/tworoom_baselines/split_manifest.json"),
    )
    parser.add_argument("--split-name", choices=("train", "validation", "calibration"))
    parser.add_argument(
        "--skip-reference-check",
        action="store_true",
        help="smoke tests only; final paired runs must not use this flag",
    )
    parser.add_argument("--device")
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--candidate-records-output", type=Path)
    parser.add_argument("--craft-records-output", type=Path)
    parser.add_argument(
        "--audit-cache", type=Path, default=Path("artifacts/cache/tworoom_baselines")
    )
    return parser


def main() -> None:
    run_tworoom_cape(build_parser().parse_args())


if __name__ == "__main__":
    main()
