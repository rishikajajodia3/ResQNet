"""
Test passing explicit seed to PathSolver
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "sionna") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "sionna"))

import sionna.rt
from sionna.rt import load_scene, Transmitter, Receiver, PathSolver, PlanarArray
import numpy as np

scene = load_scene("sionna/scenes/city_damaged.xml")
for name, obj in scene.objects.items():
    if obj.radio_material is None:
        obj.radio_material = "itu_concrete"

scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V")
scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V")
scene.frequency = 2.1e9

solver = PathSolver()

def eval_pos(pos, seed=42):
    for name in list(scene.transmitters.keys()):
        scene.remove(name)
    for name in list(scene.receivers.keys()):
        scene.remove(name)
    
    tx = Transmitter(name="tx", position=[float(x) for x in pos])
    rx = Receiver(name="rx", position=[-20.0, -30.0, 1.5])
    scene.add(tx)
    scene.add(rx)
    
    paths = solver(scene, seed=seed)
    a_real, a_imag = paths.a
    a_complex = a_real.numpy() + 1j * a_imag.numpy()
    gain = float(np.sum(np.abs(a_complex) ** 2))
    return gain

print("Testing 5 consecutive evaluations of same position with seed=42:")
for i in range(5):
    g = eval_pos([-45.0, 50.0, 12.0], seed=42)
    print(f"Run {i+1}: Linear Gain = {g:.10e}")
