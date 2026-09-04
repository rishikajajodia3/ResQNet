import { Suspense, useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, useGLTF, Line, Html } from "@react-three/drei";
import { Panel } from "./Panel.jsx";
import { api } from "../api.js";
import { UAV_COLORS, stepUavPositions, ns3DeliveryByUser } from "../multiuav.js";

// Data coords are (x, y, z) with z = height. Everything sits in a group
// rotated -90 deg about X so data-z points up in the viewer.

function CityMesh({ url }) {
  const gltf = useGLTF(url);
  const scene = useMemo(() => gltf.scene.clone(true), [gltf]);
  scene.traverse((o) => {
    if (o.isMesh) {
      o.material.color?.set("#3b4654");
      o.material.metalness = 0.0;
      o.material.roughness = 1.0;
      o.material.flatShading = true;
    }
  });
  return <primitive object={scene} />;
}

function Marker({ position, color, size = 2.5, label, hollow }) {
  return (
    <group position={position}>
      <mesh>
        <sphereGeometry args={[size, 16, 16]} />
        {hollow ? (
          <meshBasicMaterial color={color} wireframe />
        ) : (
          <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.35} />
        )}
      </mesh>
      {label && (
        <Html
          center
          distanceFactor={280}
          style={{
            color: "#e6e9ee",
            font: "600 11px/1 ui-monospace, monospace",
            background: "rgba(11,14,18,0.82)",
            border: "1px solid #2a323d",
            borderRadius: 4,
            padding: "2px 5px",
            whiteSpace: "nowrap",
            transform: "translateY(-16px)",
          }}
        >
          {label}
        </Html>
      )}
    </group>
  );
}

function SceneContents({ variant, meta, step, ns3 }) {
  const users = step?.user_results || [];
  const uavPos = stepUavPositions(step);
  const numUavs = uavPos.length || 1;
  const uavTraj = meta.uav_trajectories || (meta.trajectory ? [meta.trajectory] : []);
  const finals = meta.final_positions || (meta.final_position ? [meta.final_position] : []);
  const inits = meta.initial_positions || (meta.initial_position ? [meta.initial_position] : []);
  const delivery = ns3DeliveryByUser(ns3);

  return (
    <group rotation={[-Math.PI / 2, 0, 0]}>
      <Suspense fallback={null}>
        <CityMesh url={api.geometryUrl(variant)} />
      </Suspense>

      {/* ground users: fill = Sionna connected, ring = NS-3 delivery */}
      {users.map((u, i) => {
        const servK = (u.serving_uav || 1) - 1;
        const servColor = UAV_COLORS[servK % UAV_COLORS.length];
        const sio = u.connected;
        const del = delivery && delivery[i];
        return (
          <group key={u.user_id || i}>
            {/* comm link to serving UAV */}
            {uavPos[servK] && (
              <Line
                points={[u.position, uavPos[servK]]}
                color={servColor}
                lineWidth={sio ? 1.4 : 0.7}
                dashed={!sio}
                dashSize={2}
                gapSize={2}
                transparent
                opacity={sio ? 0.55 : 0.25}
              />
            )}
            <Marker
              position={u.position}
              color={sio ? "#4caf7d" : "#d9645b"}
              size={2.0}
            />
            {/* NS-3 delivery ring (only when NS-3 data present) */}
            {delivery && (
              <mesh position={[u.position[0], u.position[1], u.position[2] + 0.05]} rotation={[-Math.PI / 2, 0, 0]}>
                <ringGeometry args={[2.7, 3.4, 20]} />
                <meshBasicMaterial color={del ? "#4caf7d" : "#d9645b"} transparent opacity={0.9} />
              </mesh>
            )}
          </group>
        );
      })}

      {/* per-UAV trajectory + altitude + marker */}
      {uavPos.map((p, k) => {
        const color = UAV_COLORS[k % UAV_COLORS.length];
        return (
          <group key={`uav-${k}`}>
            {uavTraj[k] && uavTraj[k].length > 1 && (
              <Line points={uavTraj[k]} color={color} lineWidth={2} />
            )}
            <Line points={[[p[0], p[1], 0], [p[0], p[1], p[2]]]} color={color} lineWidth={1.3} />
            <Marker
              position={p}
              color={color}
              size={3.2}
              label={`UAV ${k + 1}  ${p[0].toFixed(0)}, ${p[1].toFixed(0)}, ${p[2].toFixed(0)} m`}
            />
            {inits[k] && (
              <Marker position={inits[k]} color={color} size={2.4} hollow />
            )}
            {finals[k] && (
              <mesh position={[finals[k][0], finals[k][1], 0.2]} rotation={[-Math.PI / 2, 0, 0]}>
                <ringGeometry args={[4, 5.2, 32]} />
                <meshBasicMaterial color={color} />
              </mesh>
            )}
          </group>
        );
      })}

      {meta.operational_region && (
        <Line
          points={(() => {
            const { x, y } = meta.operational_region;
            return [
              [x[0], y[0], 0.15], [x[1], y[0], 0.15],
              [x[1], y[1], 0.15], [x[0], y[1], 0.15], [x[0], y[0], 0.15],
            ];
          })()}
          color="#3a4652"
          lineWidth={1}
        />
      )}
    </group>
  );
}

export function Scene3D({ variant, meta, traj, stepIndex, ns3 }) {
  const step = traj?.trajectory?.[stepIndex];
  const nUav = traj?.num_uavs || 1;
  return (
    <Panel
      title="3D disaster environment"
      hint={`${nUav} UAV${nUav > 1 ? "s" : ""} · real city geometry · real coordinates`}
      flush
    >
      <div className="scene-wrap">
        <Canvas camera={{ position: [190, 150, 190], fov: 42, near: 0.5, far: 8000 }}>
          <color attach="background" args={["#0b0e12"]} />
          <ambientLight intensity={0.7} />
          <directionalLight position={[120, 200, 80]} intensity={1.1} />
          <directionalLight position={[-150, 120, -100]} intensity={0.4} />
          <gridHelper args={[1400, 28, "#1c232c", "#161b22"]} />
          <Suspense fallback={null}>
            {meta && <SceneContents variant={variant} meta={meta} step={step} ns3={ns3} />}
          </Suspense>
          <OrbitControls target={[10, 10, 0]} maxDistance={2500} minDistance={20} makeDefault />
        </Canvas>

        <div className="scene-legend">
          {Array.from({ length: nUav }).map((_, k) => (
            <div className="row" key={k}>
              <span className="swatch" style={{ background: UAV_COLORS[k % UAV_COLORS.length] }} />
              UAV {k + 1} + trajectory
            </div>
          ))}
          <div className="row">
            <span className="swatch" style={{ background: "#4caf7d" }} /> user — Sionna connected
          </div>
          <div className="row">
            <span className="swatch" style={{ background: "#d9645b" }} /> user — Sionna not connected
          </div>
          <div className="row">
            <span className="swatch" style={{ border: "2px solid #4caf7d", background: "transparent" }} />
            ring = NS-3 packet delivery
          </div>
        </div>
      </div>

      <div className="scene-note">
        Geometry: {meta?.web_asset_crop?.half_extent_m ?? "?"} m crop of{" "}
        <span className="mono">
          {variant === "damaged" ? "city_damaged.obj" : "city_baseline.obj"}
        </span>{" "}
        ({meta?.web_asset_crop?.[variant]?.final_faces ?? "?"} triangles), original
        coordinates preserved. {meta?.geometry_honesty_note}{" "}
        Damage labels: {meta?.damage_labels?.available ? "shown" : "not shown — " + (meta?.damage_labels?.reason || "unavailable")}
      </div>
    </Panel>
  );
}

useGLTF.preload(api.geometryUrl("damaged"));
