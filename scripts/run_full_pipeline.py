import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))
import multiuav_io as mio  # noqa: E402


# ============================================================
# ARGUMENTS
# ============================================================
#
# All arguments are optional and default to the historical behaviour, so
#
#     python scripts/run_full_pipeline.py
#
# still runs exactly as before (5-step live-Sionna PPO episode, NS-3 path
# taken from the NS3_PATH environment variable).

parser = argparse.ArgumentParser(
    description="Run the full ResQNet PPO -> Sionna -> JSON -> NS-3 pipeline."
)
parser.add_argument(
    "--steps",
    type=int,
    default=5,
    help="Number of PPO steps to run (default: 5).",
)
parser.add_argument(
    "--mode",
    choices=["sionna", "surrogate"],
    default="sionna",
    help="PPO feedback mode passed to ml/run_ppo.py (default: sionna).",
)
parser.add_argument(
    "--ns3-path",
    default=None,
    help="NS-3 installation path. Overrides the NS3_PATH environment "
    "variable when given.",
)
# Multi-UAV / scenario arguments. If NONE are given the pipeline runs the
# historical single-UAV / 10-user path unchanged.
parser.add_argument("--num-uavs", type=int, default=None)
parser.add_argument("--num-users", type=int, default=None)
parser.add_argument("--scenario", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--interference-model", choices=["co_channel", "orthogonal"], default="co_channel"
)
args = parser.parse_args()

MULTIUAV = (
    args.num_uavs is not None
    or args.num_users is not None
    or args.scenario is not None
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PPO_SCRIPT = PROJECT_ROOT / "ml" / "run_ppo.py"
POSITION_FILE = PROJECT_ROOT / "data" / "uav_positions.json"

# NS-3 path is configured through --ns3-path or the NS3_PATH environment
# variable. This makes the project portable across different computers.
NS3_PATH = args.ns3_path or os.environ.get("NS3_PATH")

if not NS3_PATH:
    raise RuntimeError(
        "NS3_PATH is not set. "
        "Please set it to your NS-3 installation path.\n"
        "Example:\n"
        "export NS3_PATH=/home/yourusername/ns-allinone-3.47/ns-3.47"
    )

NS3_ROOT = Path(NS3_PATH).expanduser().resolve()

if not NS3_ROOT.exists():
    raise FileNotFoundError(
        f"NS-3 directory does not exist: {NS3_ROOT}\n"
        "Please check your NS3_PATH environment variable."
    )

NS3_POSITION_FILE = NS3_ROOT / "data" / "uav_positions.json"
NS3_PROGRAM = "scratch/uav-demo"

RESULT_FILE = PROJECT_ROOT / "data" / "full_pipeline_results.json"


# ============================================================
# HELPER
# ============================================================

def run_command(command, cwd, env=None):
    print("\n" + "=" * 70)
    print("RUNNING:")
    print(" ".join(str(x) for x in command))
    print("=" * 70)

    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True
    )

    print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {result.returncode}"
        )

    return result.stdout


# ============================================================
# STEP 1: RUN PPO + SIONNA RT
# ============================================================

print("\n")
print("=" * 70)
print("        RESQNET FULL INTEGRATED PIPELINE")
print("=" * 70)

print(f"\n[1/3] Running PPO ({args.mode} feedback, {args.steps} steps)...")

env = os.environ.copy()

# Allows the project Sionna feedback engine to be imported
# while still using NVIDIA Sionna RT.
env["PYTHONPATH"] = (
    str(PROJECT_ROOT / "sionna")
    + os.pathsep
    + env.get("PYTHONPATH", "")
)

ppo_cmd = [
    "python",
    str(PPO_SCRIPT),
    "--mode",
    args.mode,
    "--steps",
    str(args.steps),
]
if MULTIUAV:
    ppo_cmd += [
        "--num-uavs", str(args.num_uavs if args.num_uavs is not None else 3),
        "--num-users", str(args.num_users if args.num_users is not None else 30),
        "--scenario", args.scenario or "blocked_urban",
        "--seed", str(args.seed),
        "--interference-model", args.interference_model,
    ]

ppo_output = run_command(ppo_cmd, cwd=PROJECT_ROOT, env=env)


# ============================================================
# STEP 2: READ PPO POSITION
# ============================================================

print("\n[2/3] Reading PPO-selected UAV position(s)...")

if not POSITION_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {POSITION_FILE}"
    )

with open(POSITION_FILE, "r") as f:
    position_data = json.load(f)

handoff = mio.read_uav_positions(POSITION_FILE)
uav_positions = handoff["uav_positions"]
num_uavs = handoff["num_uavs"]

# UAV 0 keeps the historical variable name / result shape.
uav_position = uav_positions[0]

print("\nPPO SELECTED UAV POSITION(S)")
print("---------------------------")
for k, p in enumerate(uav_positions):
    print(f"UAV {k + 1}: X = {p[0]}   Y = {p[1]}   Z = {p[2]}")


# ============================================================
# STEP 3: COPY POSITION TO NS-3
# ============================================================

print("\nCopying UAV position to NS-3...")

NS3_POSITION_FILE.parent.mkdir(parents=True, exist_ok=True)

shutil.copy2(
    POSITION_FILE,
    NS3_POSITION_FILE
)

print("Copied to:")
print(NS3_POSITION_FILE)


# ============================================================
# STEP 4: RUN NS-3
# ============================================================

print("\n[3/3] Running NS-3 network simulation...")

# Run the repository's ns3/uav-demo.cc, not whatever copy happens to be in
# the NS-3 scratch directory (the dashboard runner already does this).
NS3_SCRATCH_FILE = NS3_ROOT / "scratch" / "uav-demo.cc"
NS3_SCRATCH_FILE.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(PROJECT_ROOT / "ns3" / "uav-demo.cc", NS3_SCRATCH_FILE)
print(f"Synced ns3/uav-demo.cc -> {NS3_SCRATCH_FILE}")

ns3_output = run_command(
    [
        "./ns3",
        "run",
        NS3_PROGRAM
    ],
    cwd=NS3_ROOT
)


# ============================================================
# VERIFY POSITION HANDOFF
# ============================================================

def _pos_text(prefix, p):
    return f"{prefix}({p[0]:g}, {p[1]:g}, {p[2]:g})"


# UAV 0 always uses the historical "ML UAV Position:" line.
per_uav_handoff = [_pos_text("ML UAV Position: ", uav_positions[0]) in ns3_output]
if num_uavs > 1:
    for k, p in enumerate(uav_positions):
        per_uav_handoff.append(_pos_text(f"UAV {k + 1}: ", p) in ns3_output)

position_handoff_ok = all(per_uav_handoff)

print("\n")
print("=" * 70)
print("              POSITION HANDOFF")
print("=" * 70)

if position_handoff_ok:
    print(f"SUCCESS: NS-3 received all {num_uavs} PPO UAV position(s).")
else:
    missing = [
        i for i, ok in enumerate(per_uav_handoff) if not ok
    ]
    print(f"WARNING: Could not verify UAV position(s) {missing} in NS-3 output.")


# ============================================================
# PARSE NS-3 FLOW RESULTS
# ============================================================

flows = []

flow_pattern = re.compile(
    r"Flow:\s+(\S+)\s+->\s+(\S+)"
    r"(.*?)(?=\nFlow:|\n=+\n|\Z)",
    re.DOTALL
)

for match in flow_pattern.finditer(ns3_output):

    source = match.group(1)
    destination = match.group(2)
    block = match.group(3)

    flow = {
        "source": source,
        "destination": destination
    }

    if "No packets received." in block:
        flow["packets_received"] = 0
        flow["status"] = "NO_PACKETS_RECEIVED"
        # uav-demo.cc also reports what a failed flow sent, so its packets
        # count towards the aggregate loss below.
        sent = re.search(r"Packets Sent:\s+(\d+)", block)
        if sent:
            flow["packets_sent"] = int(sent.group(1))
            flow["packet_loss_percent"] = 100.0 if flow["packets_sent"] else 0.0
    else:
        sent = re.search(
            r"Packets Sent:\s+(\d+)",
            block
        )

        received = re.search(
            r"Packets Received:\s+(\d+)",
            block
        )

        throughput = re.search(
            r"Throughput:\s+([\d.eE+-]+)\s+Mbps",
            block
        )

        delay = re.search(
            r"Average Delay:\s+([\d.eE+-]+)\s+ms",
            block
        )

        loss = re.search(
            r"Packet Loss:\s+([\d.eE+-]+)\s+%",
            block
        )

        if sent:
            flow["packets_sent"] = int(sent.group(1))

        if received:
            flow["packets_received"] = int(received.group(1))

        if throughput:
            flow["throughput_mbps"] = float(
                throughput.group(1)
            )

        if delay:
            flow["average_delay_ms"] = float(
                delay.group(1)
            )

        if loss:
            flow["packet_loss_percent"] = float(
                loss.group(1)
            )

        flow["status"] = "OK"

    flows.append(flow)


# ============================================================
# MAP FLOWS TO SERVING UAV  (multi-UAV only)
# ============================================================

# "UAV k address: 10.1.1.X" lines are printed by ns3/uav-demo.cc when K > 1.
# For K == 1 the flow objects are left exactly as before (no uav_index key).
uav_addr_to_index = {}
if num_uavs > 1:
    for m in re.finditer(r"UAV\s+(\d+)\s+address:\s+(\S+)", ns3_output):
        uav_addr_to_index[m.group(2)] = int(m.group(1)) - 1
    for flow in flows:
        flow["uav_index"] = uav_addr_to_index.get(flow["destination"], 0)


# ============================================================
# CALCULATE SUMMARY
# ============================================================

successful_flows = [
    f for f in flows
    if f.get("packets_received", 0) > 0
]

total_flows = len(flows)
successful_count = len(successful_flows)
failed_count = total_flows - successful_count

# Calculate metrics only for flows that actually received packets.
if successful_flows:

    average_throughput = sum(
        f.get("throughput_mbps", 0)
        for f in successful_flows
    ) / successful_count

    average_delay = sum(
        f.get("average_delay_ms", 0)
        for f in successful_flows
    ) / successful_count

else:
    average_throughput = 0
    average_delay = 0

# Share of flows that delivered nothing. Kept under its historical key
# "overall_flow_loss_percent"; it is a flow-failure ratio, NOT packet loss.
overall_flow_loss = (
    failed_count / total_flows * 100
    if total_flows > 0
    else 100
)

# Packet-level aggregates over ALL flows (failed flows included).
total_sent = sum(f.get("packets_sent", 0) for f in flows)
total_received = sum(f.get("packets_received", 0) for f in flows)
flows_missing_sent = sum(1 for f in flows if "packets_sent" not in f)
packet_delivery_ratio = total_received / total_sent if total_sent > 0 else 0.0


# ============================================================
# SAVE COMBINED RESULTS
# ============================================================

results = {
    "pipeline": "ResQNet",
    "scene": "city_damaged.xml",

    # UAV 0 - unchanged shape so existing readers keep working.
    "uav_position": {
        "x": uav_position[0],
        "y": uav_position[1],
        "z": uav_position[2]
    },

    "position_handoff_to_ns3": position_handoff_ok,

    "ns3_summary": {
        "total_flows": total_flows,
        "successful_flows": successful_count,
        "failed_flows": total_flows - successful_count,
        "average_throughput_mbps": average_throughput,
        "average_delay_ms": average_delay,
        "overall_flow_loss_percent": overall_flow_loss,
        "flow_failure_percent": overall_flow_loss,
        "total_packets_sent": total_sent,
        "total_packets_received": total_received,
        "packet_delivery_ratio": packet_delivery_ratio,
        "aggregate_packet_loss_percent": (1.0 - packet_delivery_ratio) * 100.0,
        "flows_missing_packets_sent": flows_missing_sent,
        "metric_notes": {
            "average_throughput_mbps": "mean over delivering flows only; "
            "FlowMonitor IP-layer bytes (incl. IP+UDP headers) / "
            "(last rx - first tx)",
            "average_delay_ms": "mean one-way delay over delivering flows only",
            "overall_flow_loss_percent": "share of flows with zero received "
            "packets (same as flow_failure_percent), not packet loss",
            "aggregate_packet_loss_percent": "1 - total received / total sent, "
            "all flows",
        },
    },

    "flows": flows
}

if num_uavs > 1:
    # per-UAV flow breakdown
    per_uav = []
    for k in range(num_uavs):
        kf = [f for f in flows if f.get("uav_index") == k]
        ok = [f for f in kf if f.get("packets_received", 0) > 0]
        per_uav.append({
            "uav_index": k,
            "position": {
                "x": uav_positions[k][0],
                "y": uav_positions[k][1],
                "z": uav_positions[k][2],
            },
            "flows": len(kf),
            "delivering_flows": len(ok),
            "avg_throughput_mbps": (
                sum(f.get("throughput_mbps", 0) for f in ok) / len(ok) if ok else 0
            ),
            "avg_delay_ms": (
                sum(f.get("average_delay_ms", 0) for f in ok) / len(ok) if ok else 0
            ),
        })
    results["num_uavs"] = num_uavs
    results["uav_positions"] = [
        {"x": p[0], "y": p[1], "z": p[2]} for p in uav_positions
    ]
    results["associations"] = handoff["associations"]
    results["interference_model"] = args.interference_model
    results["scenario"] = args.scenario or "blocked_urban"
    results["per_uav"] = per_uav

with open(RESULT_FILE, "w") as f:
    json.dump(results, f, indent=2)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("             RESQNET PIPELINE COMPLETE")
print("=" * 70)

print("\n3D SCENE")
print("--------")
print("city_damaged.xml")

print("\nPPO UAV POSITION" + ("S" if num_uavs > 1 else ""))
print("----------------")
if num_uavs == 1:
    print(
        f"({uav_position[0]}, "
        f"{uav_position[1]}, "
        f"{uav_position[2]})"
    )
else:
    for k, p in enumerate(uav_positions):
        print(f"UAV {k + 1}: ({p[0]}, {p[1]}, {p[2]})")

print("\nNS-3 NETWORK")
print("------------")
print(f"Total flows:       {total_flows}")
print(f"Successful flows:  {successful_count}")
print(f"Failed flows:      {total_flows - successful_count}")
print(f"Avg throughput:    {average_throughput:.4f} Mbps")
print(f"Avg delay:         {average_delay:.4f} ms")
print(f"Overall flow loss: {overall_flow_loss:.2f} %  (flows with zero delivery)")
print(f"Packet delivery:   {total_received}/{total_sent} "
      f"({packet_delivery_ratio * 100:.2f} %)")
if num_uavs > 1:
    for pu in per_uav:
        print(f"  UAV {pu['uav_index'] + 1}: {pu['delivering_flows']}/{pu['flows']} "
              f"flows delivering, {pu['avg_throughput_mbps']:.4f} Mbps, "
              f"{pu['avg_delay_ms']:.4f} ms")

print("\nPOSITION HANDOFF")
print("----------------")
print(
    "PPO -> JSON -> NS-3: "
    + ("SUCCESS" if position_handoff_ok else "FAILED")
)

print("\nCombined results saved to:")
print(RESULT_FILE)

print("\n" + "=" * 70)
