import json
import os
import re
import shutil
import subprocess
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PPO_SCRIPT = PROJECT_ROOT / "ml" / "run_ppo.py"
POSITION_FILE = PROJECT_ROOT / "data" / "uav_positions.json"

# NS-3 path is configured through the NS3_PATH environment variable.
# This makes the project portable across different computers.
NS3_PATH = os.environ.get("NS3_PATH")

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

print("\n[1/3] Running PPO with live Sionna RT...")

env = os.environ.copy()

# Allows the project Sionna feedback engine to be imported
# while still using NVIDIA Sionna RT.
env["PYTHONPATH"] = (
    str(PROJECT_ROOT / "sionna")
    + os.pathsep
    + env.get("PYTHONPATH", "")
)

ppo_output = run_command(
    [
        "python",
        str(PPO_SCRIPT),
        "--mode",
        "sionna",
        "--steps",
        "5"
    ],
    cwd=PROJECT_ROOT,
    env=env
)


# ============================================================
# STEP 2: READ PPO POSITION
# ============================================================

print("\n[2/3] Reading PPO-selected UAV position...")

if not POSITION_FILE.exists():
    raise FileNotFoundError(
        f"Could not find {POSITION_FILE}"
    )

with open(POSITION_FILE, "r") as f:
    position_data = json.load(f)

uav_position = position_data["uav_positions"][0]

print("\nPPO SELECTED UAV POSITION")
print("-------------------------")
print(f"X = {uav_position[0]}")
print(f"Y = {uav_position[1]}")
print(f"Z = {uav_position[2]}")


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

expected_position_text = (
    f"ML UAV Position: "
    f"({uav_position[0]:g}, {uav_position[1]:g}, {uav_position[2]:g})"
)

position_handoff_ok = (
    expected_position_text in ns3_output
)

print("\n")
print("=" * 70)
print("              POSITION HANDOFF")
print("=" * 70)

if position_handoff_ok:
    print("SUCCESS: NS-3 received the PPO UAV position.")
else:
    print("WARNING: Could not verify the UAV position in NS-3 output.")


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

# A flow with no received packets is treated as 100% loss.
overall_flow_loss = (
    failed_count / total_flows * 100
    if total_flows > 0
    else 100
)


# ============================================================
# SAVE COMBINED RESULTS
# ============================================================

results = {
    "pipeline": "ResQNet",
    "scene": "city_damaged.xml",

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
        "overall_flow_loss_percent": overall_flow_loss
    },

    "flows": flows
}

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

print("\nPPO UAV POSITION")
print("----------------")
print(
    f"({uav_position[0]}, "
    f"{uav_position[1]}, "
    f"{uav_position[2]})"
)

print("\nNS-3 NETWORK")
print("------------")
print(f"Total flows:       {total_flows}")
print(f"Successful flows:  {successful_count}")
print(f"Failed flows:      {total_flows - successful_count}")
print(f"Avg throughput:    {average_throughput:.4f} Mbps")
print(f"Avg delay:         {average_delay:.4f} ms")
print(f"Overall flow loss: {overall_flow_loss:.2f} %")

print("\nPOSITION HANDOFF")
print("----------------")
print(
    "PPO -> JSON -> NS-3: "
    + ("SUCCESS" if position_handoff_ok else "FAILED")
)

print("\nCombined results saved to:")
print(RESULT_FILE)

print("\n" + "=" * 70)
