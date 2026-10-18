import { Suspense, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, useGLTF, Line, Grid, Html } from "@react-three/drei";
import * as THREE from "three";
import { Panel } from "./Panel.jsx";
import { api } from "../api.js";
import {
  UAV_COLORS, C_OK, C_BAD, stepUavPositions, stepActions,
  ns3DeliveryByUser, perUavStats, uavMovement, fmt,
} from "../multiuav.js";

/* Data coords (x, y, z) with z = height. Everything lives inside a group
   rotated -90deg about X so data-z points up in the viewer.
   Presentation goal: neutral engineering / GIS digital twin — no glow,
   no emissive, no animated rings. Blue-family UAV identity only. */

export function CityMesh({ url }) {
  const gltf = useGLTF(url);
  const scene = useMemo(() => {
    const s = gltf.scene.clone(true);
    s.traverse((o) => {
      if (o.isMesh) {
        o.material = new THREE.MeshStandardMaterial({
          color: "#c9d2dd",
          roughness: 0.92,
          metalness: 0.0,
          flatShading: true,
        });
      }
    });
    return s;
  }, [gltf]);
  return <primitive object={scene} />;
}

function SelectionRing({ color }) {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, -0.02]}>
      <ringGeometry args={[4.4, 5.0, 48]} />
      <meshBasicMaterial color={color} transparent opacity={0.6} />
    </mesh>
  );
}

function Drone({ position, color, idx, selected, hovered, onOver, onOut, onClick }) {
  const ref = useRef();
  useFrame((_, dt) => {
    if (!ref.current) return;
    const target = selected ? 1.3 : hovered ? 1.12 : 1;
    ref.current.scale.lerp(new THREE.Vector3(target, target, target), Math.min(1, dt * 8));
  });
  return (
    <group position={position}>
      <group
        ref={ref}
        onPointerOver={(e) => { e.stopPropagation(); onOver(); }}
        onPointerOut={(e) => { e.stopPropagation(); onOut(); }}
        onClick={(e) => { e.stopPropagation(); onClick(); }}
      >
        <mesh>
          <boxGeometry args={[3.2, 3.2, 1.1]} />
          <meshStandardMaterial color={color} roughness={0.55} metalness={0.1} />
        </mesh>
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[3.0, 0.26, 8, 24]} />
          <meshStandardMaterial color={selected ? color : "#8a97a8"} roughness={0.6} metalness={0.05} />
        </mesh>
      </group>
      {selected && (
        <group position={[0, 0, -(position[2] ?? 0) + 0.12]}>
          <SelectionRing color={color} />
        </group>
      )}
      <Html center distanceFactor={320} style={labelStyle(selected)}>
        {`UAV ${String(idx + 1).padStart(2, "0")}`}
      </Html>
    </group>
  );
}

function labelStyle(strong) {
  return {
    color: strong ? "#1d4ed8" : "#344054",
    font: `${strong ? 600 : 500} 10px/1 ui-monospace, "SF Mono", monospace`,
    background: "rgba(255,255,255,0.94)",
    border: `1px solid ${strong ? "#bfd3f2" : "#d9e0e8"}`,
    borderRadius: 2,
    padding: "2px 5px",
    whiteSpace: "nowrap",
    transform: "translateY(-22px)",
    pointerEvents: "none",
  };
}

function UserNode({ u, servingPos, servColor, connected, delivery, selected, hovered, onOver, onOut, onClick }) {
  const ref = useRef();
  useFrame((_, dt) => {
    if (!ref.current) return;
    const target = selected ? 1.8 : hovered ? 1.4 : 1;
    ref.current.scale.lerp(new THREE.Vector3(target, target, target), Math.min(1, dt * 10));
  });
  return (
    <group>
      {servingPos && (
        <Line
          points={[u.position, servingPos]}
          color={connected ? servColor : "#98a2b3"}
          lineWidth={connected ? (selected ? 1.8 : 1) : 0.7}
          dashed={!connected}
          dashSize={2}
          gapSize={2}
          transparent
          opacity={selected ? 0.75 : connected ? 0.35 : 0.18}
        />
      )}
      <mesh
        ref={ref}
        position={u.position}
        onPointerOver={(e) => { e.stopPropagation(); onOver(); }}
        onPointerOut={(e) => { e.stopPropagation(); onOut(); }}
        onClick={(e) => { e.stopPropagation(); onClick(); }}
      >
        <sphereGeometry args={[1.4, 14, 14]} />
        <meshStandardMaterial color={connected ? C_OK : C_BAD} roughness={0.6} metalness={0.0} />
      </mesh>
      {delivery !== null && (
        <mesh position={[u.position[0], u.position[1], u.position[2] + 0.05]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[2.5, 3.1, 20]} />
          <meshBasicMaterial color={delivery ? C_OK : C_BAD} transparent opacity={selected ? 0.95 : 0.7} />
        </mesh>
      )}
    </group>
  );
}

function SceneContents({ variant, meta, step, ns3, selected, onSelect }) {
  const [hover, setHover] = useState(null);
  const users = step?.user_results || [];
  const uavPos = stepUavPositions(step);
  const uavTraj = meta.uav_trajectories || (meta.trajectory ? [meta.trajectory] : []);
  const inits = meta.initial_positions || (meta.initial_position ? [meta.initial_position] : []);
  const finals = meta.final_positions || (meta.final_position ? [meta.final_position] : []);
  const delivery = ns3DeliveryByUser(ns3);
  const setCursor = (on) => { document.body.style.cursor = on ? "pointer" : "default"; };

  return (
    <group rotation={[-Math.PI / 2, 0, 0]}>
      <Suspense fallback={null}>
        <CityMesh url={api.geometryUrl(variant)} />
      </Suspense>

      {meta.operational_region && (
        <Line
          points={(() => {
            const { x, y } = meta.operational_region;
            return [
              [x[0], y[0], 0.1], [x[1], y[0], 0.1],
              [x[1], y[1], 0.1], [x[0], y[1], 0.1], [x[0], y[0], 0.1],
            ];
          })()}
          color="#9aa6b4"
          lineWidth={1}
          dashed
          dashSize={4}
          gapSize={3}
        />
      )}

      {uavPos.map((p, k) => {
        const color = UAV_COLORS[k % UAV_COLORS.length];
        return (
          <group key={`traj-${k}`}>
            {uavTraj[k] && uavTraj[k].length > 1 && (
              <Line points={uavTraj[k]} color={color} lineWidth={1.4} transparent opacity={0.7} />
            )}
            <Line points={[[p[0], p[1], 0], [p[0], p[1], p[2]]]} color={color} lineWidth={1} transparent opacity={0.4} />
            {inits[k] && (
              <mesh position={inits[k]}>
                <sphereGeometry args={[1.5, 12, 12]} />
                <meshBasicMaterial color={color} wireframe />
              </mesh>
            )}
            {finals[k] && (
              <mesh position={[finals[k][0], finals[k][1], 0.15]} rotation={[-Math.PI / 2, 0, 0]}>
                <ringGeometry args={[3.4, 4.1, 32]} />
                <meshBasicMaterial color={color} transparent opacity={0.55} />
              </mesh>
            )}
          </group>
        );
      })}

      {users.map((u, i) => {
        const servK = (u.serving_uav || 1) - 1;
        return (
          <UserNode
            key={u.user_id || i}
            u={u}
            servingPos={uavPos[servK]}
            servColor={UAV_COLORS[servK % UAV_COLORS.length]}
            connected={u.connected}
            delivery={delivery ? delivery[i] : null}
            selected={selected?.type === "user" && selected.index === i}
            hovered={hover?.type === "user" && hover.index === i}
            onOver={() => { setHover({ type: "user", index: i }); setCursor(true); }}
            onOut={() => { setHover(null); setCursor(false); }}
            onClick={() => onSelect({ type: "user", index: i })}
          />
        );
      })}

      {uavPos.map((p, k) => (
        <Drone
          key={`uav-${k}`}
          position={p}
          idx={k}
          color={UAV_COLORS[k % UAV_COLORS.length]}
          selected={selected?.type === "uav" && selected.index === k}
          hovered={hover?.type === "uav" && hover.index === k}
          onOver={() => { setHover({ type: "uav", index: k }); setCursor(true); }}
          onOut={() => { setHover(null); setCursor(false); }}
          onClick={() => onSelect({ type: "uav", index: k })}
        />
      ))}
    </group>
  );
}

function Inspector({ traj, step, ns3, selected, onClose }) {
  if (!selected) return null;
  const delivery = ns3DeliveryByUser(ns3);

  if (selected.type === "uav") {
    const k = selected.index;
    const st = perUavStats(step)[k];
    const move = uavMovement(traj)[k];
    if (!st) return null;
    return (
      <div className="inspector">
        <header>
          <span className="sdot" style={{ background: UAV_COLORS[k % UAV_COLORS.length] }} />
          <span className="ins-title">UAV {String(k + 1).padStart(2, "0")}</span>
          <button onClick={onClose} aria-label="close">×</button>
        </header>
        <div className="ins-body">
          <span className="k">Users served</span><span className="v">{st.usersServed}</span>
          <span className="k">Connected</span><span className="v">{st.connected} / {st.usersServed}</span>
          <span className="k">Mean RSS</span><span className="v">{fmt(st.meanRss)} dBm</span>
          <span className="k">Mean SNR</span><span className="v">{fmt(st.meanSnr)} dB</span>
          <span className="k">Mean SINR</span><span className="v">{fmt(st.meanSinr)} dB</span>
          <span className="k">Position</span><span className="v">{st.position?.map((c) => c.toFixed(0)).join(", ")}</span>
          <span className="k">Altitude</span><span className="v">{fmt(st.position?.[2], 0)} m</span>
          <span className="k">Movement</span><span className="v">{typeof move === "number" ? move.toFixed(1) + " m" : "—"}</span>
        </div>
      </div>
    );
  }

  const i = selected.index;
  const u = (step?.user_results || [])[i];
  if (!u) return null;
  const del = delivery ? delivery[i] : null;
  return (
    <div className="inspector">
      <header>
        <span className={`sdot ${u.connected ? "ok" : "bad"}`} />
        <span className="ins-title">{(u.user_id || `user_${i + 1}`).replace("user_", "User ")}</span>
        <button onClick={onClose} aria-label="close">×</button>
      </header>
      <div className="ins-body">
        <span className="k">Position</span><span className="v">{u.position?.map((c) => c.toFixed(0)).join(", ")}</span>
        <span className="k">Serving UAV</span><span className="v">{u.serving_uav ? "UAV " + String(u.serving_uav).padStart(2, "0") : "UAV 01"}</span>
        <span className="k">RSS</span><span className="v">{fmt(u.rss_dbm)} dBm</span>
        <span className="k">SNR</span><span className="v">{fmt(u.snr_db ?? u.sinr_db)} dB</span>
        <span className="k">SINR</span><span className="v">{fmt(u.sinr_db)} dB</span>
        <span className="k">Paths</span><span className="v">{u.num_paths ?? "—"}</span>
        <span className="k">Sionna</span>
        <span className="v" style={{ color: u.connected ? "var(--ok)" : "var(--bad)" }}>
          {u.connected ? "connected" : "not connected"}
        </span>
        <span className="k">NS-3</span>
        <span className="v" style={{ color: del === null ? "var(--text-4)" : del ? "var(--ok)" : "var(--bad)" }}>
          {del === null ? "no data" : del ? "delivery" : "no delivery"}
        </span>
      </div>
    </div>
  );
}

export function Scene3D({ variant, meta, traj, stepIndex, ns3, selected, onSelect }) {
  const step = traj?.trajectory?.[stepIndex];
  const nUav = traj?.num_uavs || 1;
  const actions = stepActions(step);
  const movedCount = actions.filter((a) => a && a !== 0).length;

  return (
    <Panel
      hero
      flush
      title="Disaster digital twin"
      tag={<span className="tag ppo">{nUav} UAV</span>}
      right={
        <span className="hint">
          step {step?.step ?? 0} / {traj?.num_steps ?? 0} &nbsp;·&nbsp; {movedCount}/{nUav} UAV repositioning
        </span>
      }
    >
      <div className="scene-wrap">
        <Canvas
          dpr={[1, 1.75]}
          camera={{ position: [230, 175, 230], fov: 40, near: 0.5, far: 9000 }}
          gl={{ antialias: true }}
        >
          <color attach="background" args={["#eef2f7"]} />
          <fogExp2 attach="fog" args={["#eef2f7", 0.00075]} />
          <hemisphereLight args={["#e4ebf3", "#c6ccd4", 0.6]} />
          <ambientLight intensity={0.75} />
          <directionalLight position={[180, 260, 120]} intensity={0.8} color="#ffffff" />
          <directionalLight position={[-160, 120, -140]} intensity={0.25} color="#e6ecf3" />
          <Grid
            args={[3000, 3000]}
            cellSize={20}
            cellThickness={0.5}
            cellColor="#dfe4ea"
            sectionSize={100}
            sectionThickness={1}
            sectionColor="#c8d0da"
            fadeDistance={1100}
            fadeStrength={2.5}
            infiniteGrid
            position={[0, -0.1, 0]}
          />
          <Suspense fallback={null}>
            {meta && (
              <SceneContents
                variant={variant} meta={meta} step={step} ns3={ns3}
                selected={selected} onSelect={onSelect}
              />
            )}
          </Suspense>
          <OrbitControls
            makeDefault
            enableDamping
            dampingFactor={0.08}
            target={[10, 10, 0]}
            minDistance={25}
            maxDistance={2600}
            maxPolarAngle={Math.PI / 2.05}
          />
        </Canvas>

        <div className="scene-overlay scene-hud">
          <div className="hud-chip">{meta?.scenario || "legacy scenario"}</div>
          <div className="hud-chip">{traj?.mode === "surrogate" ? "Surrogate / free-space" : "Sionna RT"}</div>
          {nUav > 1 && <div className="hud-chip">{meta?.interference_model || "co_channel"}</div>}
        </div>

        <div className="scene-overlay scene-legend">
          {Array.from({ length: nUav }).map((_, k) => (
            <div className="row" key={k}>
              <span className="swatch" style={{ background: UAV_COLORS[k % UAV_COLORS.length] }} />
              UAV {String(k + 1).padStart(2, "0")} + trajectory
            </div>
          ))}
          <div className="row"><span className="swatch" style={{ background: C_OK }} /> Sionna connected</div>
          <div className="row"><span className="swatch" style={{ background: C_BAD }} /> Sionna not connected</div>
          <div className="row"><span className="swatch ring" style={{ border: `2px solid ${C_OK}` }} /> ring = NS-3 delivery</div>
        </div>

        <Inspector traj={traj} step={step} ns3={ns3} selected={selected} onClose={() => onSelect(null)} />
      </div>

      <div className="scene-note">
        {meta?.web_asset_crop?.half_extent_m ?? "?"} m crop of{" "}
        <span className="mono">{variant === "damaged" ? "city_damaged.obj" : "city_baseline.obj"}</span>{" "}
        (~{meta?.web_asset_crop?.[variant]?.final_faces ?? "?"} triangles), original coordinates
        preserved. {meta?.geometry_honesty_note} No physical coverage field is rendered — only the
        ray-traced per-user connectivity the pipeline returns.
      </div>
    </Panel>
  );
}

useGLTF.preload(api.geometryUrl("damaged"));
