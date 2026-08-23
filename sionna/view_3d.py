from sionna.rt import (
    load_scene,
    Transmitter,
    Receiver,
    PlanarArray
)

SCENE = "/home/rishika/ResQNet/sionna/scenes/city_damaged.xml"

# Load damaged 3D city
scene = load_scene(SCENE)

scene.frequency = 2.1e9

# Antennas
scene.tx_array = PlanarArray(
    num_rows=1,
    num_cols=1,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="iso",
    polarization="V"
)

scene.rx_array = PlanarArray(
    num_rows=1,
    num_cols=1,
    vertical_spacing=0.5,
    horizontal_spacing=0.5,
    pattern="iso",
    polarization="V"
)

# PPO-selected UAV
uav = Transmitter(
    name="PPO_UAV",
    position=[-18.82, 48.41, 30.57]
)

scene.add(uav)

# Example survivors/users
users = [
    [-20, -30, 1.5],
    [-10, -20, 1.5],
    [0, -10, 1.5],
    [10, 0, 1.5],
    [20, 10, 1.5]
]

for i, pos in enumerate(users):
    rx = Receiver(
        name=f"user_{i+1}",
        position=pos
    )
    scene.add(rx)

print("3D SCENE LOADED")
print("PPO UAV:", [-18.82, 48.41, 30.57])
print("Objects:", len(scene.objects))
print("Opening interactive viewer...")

scene.preview()
