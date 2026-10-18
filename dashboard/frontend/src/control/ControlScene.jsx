import { Suspense, useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Line, Grid, Html } from "@react-three/drei";
import * as THREE from "three";
import { CityMesh } from "../components/Scene3D.jsx";
import { api } from "../api.js";
import { sceneData, PDR_GOOD, PDR_MID, PDR_BAD } from "./model.js";

// Data coordinates (x, y, z = height) are the Sionna scene's own; the group is
// rotated -90 deg about X so z points up, exactly as in Scene3D.

const STATUS_TEXT = {
  initial: "initial", holding: "holding", moving: "moving", moved: "moved", reverted: "reverted",
};

function UavMarker({ u, selected, onSelect }) {
  const ref = useRef();
  const target = useMemo(() => new THREE.Vector3(...u.position), [u.position]);
  // glide to the backend position (display only; the position itself is the backend's)
  useFrame((_, dt) => {
    if (ref.current) ref.current.position.lerp(target, Math.min(1, dt * 3));
  });
  return (
    <group ref={ref} position={u.position}>
      <group onClick={(e) => { e.stopPropagation(); onSelect(u.index); }}
             onPointerOver={() => { document.body.style.cursor = "pointer"; }}
             onPointerOut={() => { document.body.style.cursor = "default"; }}>
        <mesh>
          <boxGeometry args={[4.2, 4.2, 1.4]} />
          <meshStandardMaterial color={u.color} roughness={0.5} />
        </mesh>
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[3.8, selected ? 0.55 : 0.3, 8, 28]} />
          <meshStandardMaterial color={selected ? "#111827" : u.status === "reverted" ? PDR_BAD
            : u.status === "moving" ? PDR_MID : "#8a97a8"} />
        </mesh>
      </group>
      {/* vertical stem to the ground shows the altitude */}
      <Line points={[[0, 0, 0], [0, 0, -u.position[2]]]} color={u.color} lineWidth={1} dashed dashSize={2}
            gapSize={1.5} transparent opacity={0.6} />
      <Html center distanceFactor={190} style={{ pointerEvents: "none" }}>
        <div className={`uav-label${selected ? " sel" : ""}`} style={{ borderColor: u.color }}>
          <b style={{ color: u.color }}>{u.id}</b> z {u.position[2].toFixed(0)} m
          <span className={`st st-${u.status}`}>{STATUS_TEXT[u.status] || u.status}</span>
        </div>
      </Html>
    </group>
  );
}

function Contents({ state, selected, onSelect, showLinks, showFootprints }) {
  const { uavs, users, trails } = useMemo(() => sceneData(state), [state]);
  const reg = state.scene?.region;
  return (
    <group rotation={[-Math.PI / 2, 0, 0]}>
      <Suspense fallback={null}>
        <CityMesh url={api.geometryUrl("damaged")} />
      </Suspense>
      {reg && (
        <Line
          points={[[reg.x[0], reg.y[0], 0.3], [reg.x[1], reg.y[0], 0.3], [reg.x[1], reg.y[1], 0.3],
                   [reg.x[0], reg.y[1], 0.3], [reg.x[0], reg.y[0], 0.3]]}
          color="#64748b" lineWidth={1} dashed dashSize={5} gapSize={4}
        />
      )}
      {showLinks && users.map((u) => u.servingPos && (
        <Line key={`l-${u.id}`} points={[u.position, u.servingPos]} color={u.color}
              lineWidth={selected === u.serving_uav - 1 ? 1.6 : 0.9} transparent
              opacity={selected === null || selected === u.serving_uav - 1 ? 0.55 : 0.12}
              dashed={u.pdr < 0.05} dashSize={2} gapSize={2} />
      ))}
      {users.map((u) => (
        <mesh key={u.id} position={u.position}>
          <sphereGeometry args={[1.7, 12, 12]} />
          <meshStandardMaterial color={u.color} roughness={0.6} />
        </mesh>
      ))}
      {trails.map((t, k) => t.length > 1 && (
        <Line key={`t-${k}`} points={t} color={uavs[k].color} lineWidth={1.4} transparent opacity={0.45} />
      ))}
      {showFootprints && uavs.map((u) => u.footprint_m > 0 && (
        <mesh key={`f-${u.id}`} position={[u.position[0], u.position[1], 0.4]}>
          <ringGeometry args={[Math.max(0, u.footprint_m - 0.8), u.footprint_m, 64]} />
          <meshBasicMaterial color={u.color} transparent opacity={selected === null || selected === u.index ? 0.5 : 0.15}
                             side={THREE.DoubleSide} />
        </mesh>
      ))}
      {uavs.map((u) => (
        <UavMarker key={u.id} u={u} selected={selected === u.index} onSelect={onSelect} />
      ))}
    </group>
  );
}

export function ControlScene({ state, selected, onSelect, showLinks = true, showFootprints = true }) {
  const c = state?.scene?.region
    ? [(state.scene.region.x[0] + state.scene.region.x[1]) / 2, (state.scene.region.y[0] + state.scene.region.y[1]) / 2]
    : [0, 0];
  // three.js coordinates of the data point (c0, c1, 0) after the -90 deg X rotation
  const target = [c[0], 10, -c[1]];
  return (
    <div className="ctl-scene">
      <Canvas dpr={[1, 1.75]} camera={{ position: [c[0] + 190, 170, -c[1] + 210], fov: 40, near: 0.5, far: 9000 }}
              onPointerMissed={() => onSelect(null)}>
        <color attach="background" args={["#eef2f7"]} />
        <hemisphereLight args={["#e4ebf3", "#c6ccd4", 0.6]} />
        <ambientLight intensity={0.75} />
        <directionalLight position={[180, 260, 120]} intensity={0.8} />
        <Grid args={[3000, 3000]} cellSize={20} cellColor="#dfe4ea" sectionSize={100} sectionColor="#c8d0da"
              fadeDistance={1400} infiniteGrid position={[0, -0.1, 0]} />
        {state?.uavs && (
          <Contents state={state} selected={selected} onSelect={onSelect}
                    showLinks={showLinks} showFootprints={showFootprints} />
        )}
        <OrbitControls makeDefault enableDamping target={target} minDistance={25} maxDistance={2600}
                       maxPolarAngle={Math.PI / 2.05} />
      </Canvas>
      <div className="ctl-legend">
        <div><span className="dot" style={{ background: PDR_GOOD }} /> survivor PDR ≥ target</div>
        <div><span className="dot" style={{ background: PDR_MID }} /> PDR ≥ coverage threshold</div>
        <div><span className="dot" style={{ background: PDR_BAD }} /> below threshold / no delivery (dashed link)</div>
        <div><span className="ring" /> ring: farthest user at PDR target (measured)</div>
        <div><span className="stem" /> dashed stem: UAV altitude</div>
      </div>
    </div>
  );
}
