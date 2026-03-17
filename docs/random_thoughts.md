# Random Thoughts

## FUSE's Relationship to VLAs and World Models

**Date:** 2026-03-13

### Where FUSE Sits

FUSE is a perception front-end. It produces structured, per-object 3D representations (label + centroid + complete point cloud) from stereo camera input. It doesn't plan actions or predict futures — it tells downstream systems *what's where* and *what shape it is*.

```
FUSE (perception)  →  per-object 3D state  →  VLA / World Model / Grasp Planner (decision)
```

### Connection to VLAs

VLAs (RT-2, OpenVLA, etc.) take visual observations + language instructions and output robot actions. They need to know object identity and position. FUSE provides exactly that — a FusedObject with label, 3D centroid, and complete geometry. Without a perception layer like this, a VLA is blind.

### Connection to World Models

World models predict what happens next given current state + action. Current world models (UniSim, GAIA-1, Dreamer) mostly work in 2D (predict future video frames). This breaks for manipulation because:
- 2D frames don't encode 3D position
- Can't simulate "push mug 5cm left" without 3D geometry
- Occlusion makes 2D prediction unreliable for contact-rich tasks

FUSE provides an object-centric 3D state that a world model could simulate from. The complete point cloud from Hunyuan3D matters — a partial front-facing scan isn't enough to predict contact physics (how a mug tips, rolls, nests against other objects). A 3D world model could take FUSE's output and predict state transitions under actions.

This connection isn't direct yet — nobody has built a 3D object-centric world model that consumes exactly our output format. But research is heading there (GROOT, CoRL 2023 shows object-centric 3D representations improve policy learning).

### Is FUSE a Research Contribution?

The pipeline pattern (camera → detect → segment → 3D extract) is standard. But the specific combination is novel:

1. **Generative 3D in a real-time robot pipeline** — SR3D (2025) is the closest prior work but uses TripoSR (solid blobs, no topology). We're the first to use a high-quality DiT-based generator (Hunyuan3D Full) and solve the latency problem with async cloud inference.

2. **Point cloud decoder (Phase 10)** — replacing a 17s volume decoder with a 1-3s cross-attention decoder while preserving topology. Applied ML, but the (latent, point cloud) distillation on a frozen DiT is a real result if it works.

3. **Systematic model evaluation** — tested 6 models, identified three architectural requirements for semantic 3D understanding (direct image conditioning + DiT global attention + sufficient capacity). That analysis has standalone value.

Systems contribution (integrating generative 3D into real-time perception) + applied ML contribution (decoder fine-tuning). Both publishable if results hold.

### Closest Existing Systems

| System | How it compares |
|--------|----------------|
| SR3D (2025) | Most similar. Grounded-SAM → TripoSR → grasp. But TripoSR produces solid blobs. |
| OK-Robot (NYU/Meta 2024) | Modular open-vocab pipeline but no shape completion at all. |
| ZeroGrasp (CVPR 2025) | End-to-end shape recon + grasp. Monolithic, not open-vocab. |
| ConceptGraphs (ICRA 2024) | Open-vocab 3D scene graphs. Scene-level, no shape completion. |

Nobody has published a system using a DiT-based image-to-3D generator in a robotic perception pipeline with async cloud architecture.

---

## The Full Vision: Perception → World Model → Dexterous Grasp

**Date:** 2026-03-13

### Three-Stage System

```
Stage 1: FUSE Perception (what we're building now)
  ZED camera → YOLOE seg → partial cloud + image
  → Hunyuan3D → complete point cloud aligned to camera frame
  → FusedObject per object (mug, table, etc.)

Stage 2: World Model + Simulator
  Complete point clouds (mug + table) → Poisson mesh → CoACD → MuJoCo
  → LEAP Hand model in MuJoCo Playground
  → Train manipulation policy / simulate hand-object interaction
  → Visualize training/inference in MuJoCo

Stage 3: Dexterous Grasp Planning Output
  → Per-finger contact points on the mug's actual geometry
  → Visualize: hand mesh with each finger placed on the surface
  → Execute on real robot
```

### World Models vs Simulators

They're different things. MuJoCo/Isaac Sim are physics simulators — deterministic, require explicit scene definition. World models are learned simulators — predict next state from experience, no explicit physics.

Current state: **learned world models aren't ready for dexterous manipulation.** DreamerV3 and TD-MPC2 work for coarse tasks but fail on contact-rich multi-finger scenarios. The field still uses physics simulators (MuJoCo Playground, IsaacGym). Learned world models will likely replace simulators eventually but not yet.

Three ways they interact:
1. Train world model inside sim (most common)
2. Replace sim entirely with learned model (DayDreamer — works for locomotion, not dexterous yet)
3. Hybrid — digital twin + physics sim (DreMa)

### Key Grasp Planning Systems

| System | Input | Hand | What it does |
|--------|-------|------|-------------|
| DexGraspNet 2.0 (2024) | Depth point cloud | LEAP Hand | 427M grasps, diffusion model, 90.7% real success |
| AnyDexGrasp (ICLR 2025) | Partial point cloud | Any hand | Hand-agnostic, contact-centric repr, hundreds of real trials |
| DexDiffuser (2024) | Partial point cloud | Allegro Hand | Diffusion + evaluator refinement, open source |
| UniDexGrasp++ (CVPR 2023) | Point cloud | Shadow Hand | RL-based grasp+lift, 85% success |

### Point Cloud → MuJoCo Pipeline

```
FUSE complete point cloud (N×3)
  → Open3D Poisson surface reconstruction → watertight mesh
  → CoACD convex decomposition (pip install coacd)
  → obj2mjcf (pip install obj2mjcf) → MJCF XML
  → MuJoCo simulation with LEAP Hand
```

### Hardware: LEAP Hand

$2K (vs $16K Allegro, $100K+ Shadow), 16 DOF, MuJoCo models ready in MuJoCo Playground, DexGraspNet 2.0 provides 427M grasp labels, sim-to-real demonstrated.

### The Main Innovation

It's the **connection between generative 3D and dexterous manipulation that nobody has made yet.**

Every grasp planner (DexGraspNet, AnyDexGrasp, DexDiffuser) takes point clouds as input. But they all use raw depth camera output — partial, noisy, single-view. They grasp based on what they can see, not what they know is there.

FUSE's insight: **use a generative 3D model (Hunyuan3D) to complete the shape before planning the grasp.** This means:

1. **The grasp planner sees the full object** — including the handle backside, the interior, the bottom — not just the front-facing surface. It can plan finger placements on geometry it can't directly observe.

2. **Contact physics reasoning improves** — a world model simulating a grasp on a complete mesh predicts contact dynamics more accurately than one working with a partial scan full of holes.

3. **Topology matters for dexterous grasping** — you can't plan a finger through a handle loop if the model thinks the handle is a solid bump. We showed only Hunyuan3D Full gets this right (through-hole handle, hollow interior). SR3D tried with TripoSR and got solid blobs.

The contribution is: **perception-quality complete 3D geometry fed into dexterous grasp planning**, enabled by a specific combination (open-vocab seg + DiT-based generative 3D + async cloud) that didn't exist before.

### Key Papers to Read

**VLAs:**
- RT-2 (CoRL 2023) — coined "VLA", 2D-only, shows what's missing without 3D
- OpenVLA (ICML 2024) — open-source 7B VLA, most practical to extend with 3D
- 3D-VLA (ICML 2024) — first paper arguing VLAs need 3D point cloud input, also a world model
- PointVLA (Mar 2025) — inject point clouds into existing VLAs without retraining

**World Models:**
- World Models (Ha & Schmidhuber, NeurIPS 2018) — the OG
- DreamerV3 (Nature 2025) — most mature, 150+ tasks, pixel-based
- DayDreamer (CoRL 2022) — first Dreamer on real robots
- FOCUS (2025) — per-object world model, validates object-centric 3D representations

**Dexterous Grasping:**
- DexGraspNet 2.0 (2024) — LEAP Hand, diffusion-based, 90.7% real success
- AnyDexGrasp (ICLR 2025) — hand-agnostic, contact-centric, minimal real trials
- DexDiffuser (2024) — diffusion + evaluator, open source

**Shape Completion for Grasping:**
- 3DSGrasp (2023) — transformer point cloud completion → GPD grasp, +30pp success rate
- PCF-Grasp (2025) — completion as feature for 6-DOF grasp, +24pp success rate (17.8% → 41.6%)

---

## The Gap in 3D-VLA and PointVLA

**Date:** 2026-03-13

### Neither Uses Shape Completion

Both 3D-VLA and PointVLA claim to bring 3D understanding to VLAs. But neither actually completes object geometry:

- **3D-VLA (ICML 2024):** Uses 3D scene tokens from multi-view reconstruction. The 3D representation is a scene-level feature, not per-object complete geometry. It's essentially a better spatial encoding of what the cameras already see — not geometry the cameras *can't* see.

- **PointVLA (Mar 2025):** Injects point clouds from depth sensors into frozen VLAs via a lightweight adapter. But the point clouds are raw depth — partial, single-view, with holes and occlusion. No completion step. The "3D" is just repackaged depth.

Both improve over pure 2D VLAs, but neither gives the policy access to **complete object geometry** — the backside of the mug, the interior of the bowl, the underside of the bottle. They still plan actions based on what they can see, not what they know is there.

### Neither Has a Simulation Layer

This is the bigger problem. Neither 3D-VLA nor PointVLA validates proposed actions in simulation before executing them. They go directly from perception → action:

```
3D-VLA:   multi-view images → 3D tokens → transformer → action
PointVLA: RGB + depth cloud → VLA + adapter → action
```

Without a physics simulation step, there's no way to check:
- Will the grasp actually be stable? (contact forces, friction cones)
- Will the object tip over during approach?
- Does the planned trajectory collide with other objects?
- Is the finger placement physically reachable given joint limits?

3D-VLA has a "world model" component (it predicts future 3D states), but it's a learned predictor — it hallucinates plausible futures, it doesn't simulate physics. For contact-rich dexterous manipulation, you need actual physics simulation to verify grasp stability.

### What FUSE's Three-Stage System Adds

```
Existing (3D-VLA/PointVLA):
  partial 3D → learned policy → action (hope it works)

FUSE's approach:
  partial 3D → foundation model completion → full geometry
  → physics sim (MuJoCo) → verify grasp stability → action (know it works)
```

The key additions:
1. **Foundation model shape completion** — Hunyuan3D Full provides complete geometry including topology (handles, holes, hollow interiors) that no depth sensor can observe directly
2. **Physics simulation verification** — CoACD + MuJoCo lets us test grasp stability before execution, rather than relying on a learned policy to implicitly learn physics

### Foundation Model vs Task-Trained Completion

Existing shape-completion-for-grasping work (3DSGrasp, PCF-Grasp) uses task-trained completors — small networks trained on specific object categories to fill in missing geometry. They work (+30pp and +24pp respectively) but:

- **Category-limited:** Trained on specific shapes. Novel objects = poor completion.
- **No semantic understanding:** They complete geometry but don't understand what a handle *is* or how a mug's interior relates to its function.
- **Low-fidelity topology:** Tend to produce smooth interpolations. A handle might get filled in as a solid bump rather than a through-hole.

Foundation model completion (Hunyuan3D Full) is different:
- **Zero-shot generalization:** Trained on massive 3D datasets, handles novel objects without retraining
- **Semantic understanding:** The DiT architecture with direct image conditioning understands that a mug handle should be a loop, not a bump
- **High-fidelity topology:** We demonstrated through-hole handles, hollow interiors — geometry that task-trained completors miss

The thesis: **foundation-model-quality shape completion will improve grasp planning more than task-trained completion, because it generalizes better and preserves the topological features that matter for dexterous manipulation.**

Nobody has tested this. 3DSGrasp and PCF-Grasp prove completion helps grasping. We're testing whether *better* completion (from a foundation model) helps *more* — and extends to objects the system has never seen before.

### The Generalization Argument

Task-trained completors learn shape priors for specific categories. A completor trained on mugs can fill in a mug's backside but may fail on a wine glass. A foundation model like Hunyuan3D has seen millions of 3D objects — it can complete a wine glass, a wrench, a toy dinosaur, all zero-shot.

This matters for real-world robotics where you can't retrain the completor every time you encounter a new object. The whole point of open-vocab detection (YOLOE) is handling arbitrary objects. The 3D completion should generalize just as broadly.

---

## Hunyuan3D v2.1 vs v2.0 Quick Notes

**Date:** 2026-03-13

Tested v2.1 (3B, MoE with 6 MoE layers, 8 experts, top-2 routing, DINOv2-Large 1024-dim, 4096 latent tokens) against v2.0 (1.1B, DINOv2-Giant 1536-dim, 3072 latent tokens).

- v2.1 is 15-20% slower (~31s vs ~27s on A100)
- Point cloud quality is roughly comparable for our use case
- v2.1 produces smaller meshes (~210-295K verts vs ~815K)
- **Decision:** Stay with v2.0 as baseline. The extra capacity doesn't help for single-object completion.

See `docs/experiments.md` Experiment 8 for full comparison data.

---

## Simulation + VLA: State of the Field

**Date:** 2026-03-13

### How Simulation Is Used With VLAs Today

Surveyed the intersection of physics simulation and VLAs. The work falls into four categories — none of which do what we're proposing.

#### 1. Sim for Training VLAs (most common)

| System | Year | What it does |
|--------|------|-------------|
| VLA-RFT | 2025 | Runs VLA-proposed actions in MuJoCo/Isaac Sim, uses success/failure as reward signal to fine-tune VLA via RL. Sim-in-the-loop during *training*, not inference. |
| DreamGen | 2025 | Trains a video world model in sim, generates synthetic rollouts to train VLA policies. Sim → world model → synthetic data → better VLA. |
| MultiGen | 2025 | Multimodal generation *within* simulation to create diverse training scenarios for real-world policies. |
| GR00T N1 / N1.6 (NVIDIA) | 2025 | Full sim-to-real pipeline for humanoid VLAs. Train in Isaac Sim, deploy to real robot. Classic sim-for-training at VLA scale. |

#### 2. Learned World Models Inside VLAs (not physics sim)

| System | Venue | What it does |
|--------|-------|-------------|
| 3D-VLA | ICML 2024 | Internal "world model" predicts future 3D states. Learned predictor, not physics — hallucinates plausible futures, no contact force verification. |
| WorldVLA | 2025 | Autoregressive action + world model in one architecture. Predicts next state as part of action generation. Same limitation: learned, not physics-based. |
| VLAW | 2026 | Iteratively co-trains VLA and world model so they improve each other. World model as mental rehearsal. |
| IRL-VLA | 2025 | "Reward world model" scores VLA actions during training. |

#### 3. Reasoning Before Acting (closest to verification)

| System | Venue | What it does |
|--------|-------|-------------|
| ThinkAct | NeurIPS 2025 | Visual chain-of-thought in learned latent space before outputting actions. Mental simulation, not physics. |
| CoT-VLA | CVPR 2025 | Visual chain-of-thought reasoning. Plans in image space before acting. |

#### 4. Sim for Evaluation

- **NVIDIA Isaac Lab-Arena** — Standardized sim environment for *evaluating* VLA policies, not verifying individual actions at inference time.

### The Gap: No Physics-Sim-in-the-Loop at Inference Time

**Nobody does physics simulation verification at inference time.** Every existing system either:
1. Uses sim for training/fine-tuning (VLA-RFT, GR00T)
2. Uses a learned world model as a soft proxy for sim (3D-VLA, WorldVLA, ThinkAct)
3. Uses sim for evaluation benchmarks (Isaac Lab-Arena)

No published system takes a VLA's proposed action, runs it through MuJoCo/Isaac to check contact forces and grasp stability, and then decides whether to execute.

The closest is **VLA-RFT**, which does sim verification but only during training to generate reward signals — not at test time. And none of them feed *complete* 3D geometry into the simulation — they all work with whatever the sensors observe.

### Why This Matters for FUSE

FUSE's three-stage system proposes exactly this missing piece:

```
VLA proposes action → build sim scene from FUSE's complete geometry
→ test action in MuJoCo → check contact forces, stability, collisions
→ execute if stable, re-plan if not
```

This requires two things no existing system has:
1. **Complete 3D geometry** to build an accurate sim scene (FUSE provides this via Hunyuan3D)
2. **Fast enough sim** to verify actions in real-time (MuJoCo runs at kHz — not the bottleneck)

The bottleneck is perception (getting complete geometry fast enough), which is exactly what Phase 10's decoder optimization addresses. If we get Hunyuan3D down to ~10s, the sim verification step is essentially free (<1ms in MuJoCo).

### Implications for the Thesis

The "simulate before you act" loop at inference time for dexterous manipulation with foundation-model-completed geometry is genuinely novel. It combines:
- Foundation model 3D completion (nobody else does this for VLAs)
- Physics verification at inference time (nobody else does this for VLAs)
- Complete geometry in the sim scene (nobody else has this)

This isn't just a systems contribution — it's a new paradigm for VLA execution that addresses a fundamental limitation of current approaches: they learn physics implicitly from data rather than verifying it explicitly through simulation.

---

## How Existing Systems Handle Partial Point Clouds

**Date:** 2026-03-16

### The Standard Downstream Pipeline

Almost every robotic grasping system follows the same pattern:

```
Depth Camera (RealSense, ZED, etc.)
  → Raw partial point cloud (single-view, noisy, holes)
  → Segmentation (project 2D masks to 3D)
  → Per-object partial cloud
  → Grasp planner directly on partial data
  → Execute (hope for the best)
```

### What Each System Does With Partial Data

| System | What happens to the partial cloud |
|--------|----------------------------------|
| GraspNet-1Billion | Partial cloud → 6-DOF grasp prediction directly. No completion. |
| Contact-GraspNet | Partial scene cloud → sample grasp candidates on visible surfaces only |
| AnyDexGrasp | Partial cloud → contact-centric grasp repr → hand-specific grasp. No completion. |
| DexGraspNet 2.0 | Partial cloud → depth restoration (clean up noise) → diffusion grasp model. Still partial. |
| OK-Robot | iPhone LiDAR scan → point cloud map → open-vocab query → grasp. No per-object completion. |
| RT-2 / OpenVLA | No point cloud — pure 2D pixels → action |
| 3D-VLA | Multi-view reconstruction → 3D tokens. Better than single-view but still only what cameras see. |
| PointVLA | Raw depth cloud injected into VLA. No processing beyond basic filtering. |

### How They Cope Without Complete Geometry

1. **Training on partial data** — models learn to infer reasonable grasps from partial observations
2. **Conservative grasp selection** — prefer top-down pinches on visible, stable surfaces
3. **Multiple attempts** — if the first grasp fails, re-observe and retry

### What Gets Lost

- Handle backsides (can't plan a wrap-around grasp you only see from the front)
- Object bottoms (can't plan a scoop grasp under a bowl)
- Interior geometry (can't distinguish hollow mug from solid cylinder)
- Thin structures (fork tines, pen caps — often missing from depth entirely)

### Only Two Exceptions (task-trained completion)

Only **3DSGrasp** (+30pp) and **PCF-Grasp** (+24pp) add a completion step before grasp planning, and both use small category-specific networks — not foundation models. They validate that completion helps but don't generalize to novel objects.

---

## FUSE's Foundation Model vs 3D-VLA's Foundation Model

**Date:** 2026-03-16

Both FUSE and 3D-VLA use foundation models, but for completely different purposes.

### 3D-VLA: Foundation Model as Decision-Maker

3D-VLA uses a language/vision foundation model (LLaMA-based LLM fine-tuned with 3D tokens). The FM is the **decision-maker** — it takes 3D scene representations + language instructions and outputs actions.

```
3D-VLA:  RGB images → 3D scene encoder → 3D tokens → LLM (foundation model) → action
         The FM decides WHAT TO DO
```

### FUSE: Foundation Model as Geometry Generator

FUSE uses a generative 3D foundation model (Hunyuan3D Full — DiT trained on millions of 3D objects). The FM is the **perception layer** — it takes a single RGB image and generates complete 3D geometry that didn't exist in the sensor data.

```
FUSE:    RGB image → Hunyuan3D (foundation model) → complete mesh/point cloud → downstream
         The FM creates geometry THAT WASN'T OBSERVED
```

### Comparison

| | 3D-VLA's FM | FUSE's FM |
|---|---|---|
| Type | LLM (language + vision + action) | Generative 3D (image → mesh) |
| Role | Decision-making (what action to take) | Perception (what the object looks like) |
| Input | 3D scene tokens + language | Single RGB crop |
| Output | Robot actions | Complete 3D geometry |
| What it adds | Reasoning about goals + actions | Geometry the camera can't see |
| 3D representation | Scene-level features (from what cameras observed) | Per-object complete shape (including occluded surfaces) |

### The Key Insight

3D-VLA's FM is smart about **what to do** but blind about **what's there**. Its 3D tokens are derived from camera observations — if the back of the mug isn't visible, the tokens don't encode it.

FUSE's FM doesn't decide actions but **hallucinates complete geometry** from a single view — handle backside, hollow interior, bottom surface — all from the model's learned prior over millions of 3D objects.

### They're Complementary

In principle, you could feed FUSE's completed geometry into 3D-VLA's scene encoder and get a system that both **sees complete objects** and **reasons about actions** — which neither does alone today.
