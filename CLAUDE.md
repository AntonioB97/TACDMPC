# CLAUDE.md - AI Assistant Guide for TACDMPC

**Last Updated**: 2025-11-14
**Repository**: TACDMPC (Transformer Actor-Critic with Differentiable MPC)
**Version**: 0.2.0

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Codebase Structure](#codebase-structure)
3. [Core Architecture](#core-architecture)
4. [Development Workflows](#development-workflows)
5. [Coding Conventions](#coding-conventions)
6. [Testing Practices](#testing-practices)
7. [Common Tasks](#common-tasks)
8. [Important Notes & Gotchas](#important-notes--gotchas)
9. [Key Files Reference](#key-files-reference)

---

## Project Overview

### What is TACDMPC?

TACDMPC is a PyTorch-based reinforcement learning library that combines:
- **Actor-Critic RL** (using PPO algorithm)
- **Differentiable Model Predictive Control (MPC)** as the actor's policy
- **Transformer-based Critic** for value estimation with history awareness

### Key Innovation

Instead of learning a direct observation→action mapping, the **actor learns to parameterize the cost function** of an MPC controller. The MPC then solves an optimal control problem using a known dynamics model, producing actions that are:
- Physics-aware (respects dynamics constraints)
- Interpretable (through cost function parameters)
- End-to-end trainable (via differentiable MPC layer)

### Academic Foundation

Based on papers:
- "Actor-Critic Model Predictive Control" (Romero et al., 2023)
- "Differentiable MPC for End-to-end Planning and Control" (Amos et al., 2019)

### Technology Stack

- **PyTorch >= 2.2**: Deep learning framework, autograd, vmap
- **HuggingFace Transformers >= 4.37**: BERT-based critics
- **Gym/Gymnasium**: Environment interfaces
- **Python >= 3.11**: Modern Python features

---

## Codebase Structure

```
TACDMPC/
├── ACMPC/                          # Actor-Critic implementation
│   ├── actor.py                    # ActorMPC class (policy network)
│   ├── critic_transformer.py       # Transformer-based value functions
│   ├── training_loop.py            # PPO training loop
│   ├── parallel_env.py             # Parallel environment manager
│   ├── Checkpoint_Manager.py       # Save/load checkpoints
│   └── __init__.py
│
├── DifferentialMPC/                # Differentiable MPC controller
│   ├── controller.py               # Main MPC solver (iLQR)
│   ├── cost.py                     # Quadratic cost functions
│   ├── utils.py                    # PNQP solver, batched operations
│   ├── AugmentedLagrangianManager.py  # Constraint handling
│   └── __init__.py
│
├── examples/                       # Demonstration scripts
│   ├── 01_demo_linear_AC.py       # Double integrator (WORKING)
│   ├── 02_demo_cartpole_AC.py     # CartPole stabilization (WORKING)
│   ├── 03_demo_unicycle_AC.py     # Trajectory tracking (INCOMPLETE)
│   ├── __main__.py                 # CLI entry point
│   └── checkpoints_*/              # Pre-trained models
│
├── tests/                          # Test suite (18 tests, OUTDATED)
│   ├── conftest.py
│   └── test_*.py
│
├── utils/                          # Utility modules
│   └── profiler.py                 # Performance profiling
│
├── DeprecatedActorCritic/          # Legacy code (do not use)
├── DepracatedKoopmanAutoEncoders/  # Experimental features (deprecated)
│
├── pyproject.toml                  # Package configuration
├── requirements.txt                # Dependencies
├── README.md                       # User documentation
└── LICENSE                         # Apache 2.0
```

### Package Organization

- **ACMPC**: High-level RL training logic (actor, critic, training loop)
- **DifferentialMPC**: Low-level MPC solver and optimization
- **examples**: Runnable demonstrations and benchmarks
- **tests**: Unit tests (currently outdated, needs update)
- **utils**: Cross-cutting utilities

---

## Core Architecture

### Data Flow

```
Environment Observation
         ↓
    ActorMPC (policy)
         ├→ cost_map_net: MLP(obs) → cost parameters (C, c)
         └→ DifferentiableMPCController: solves MPC with costs
                   ↓
            Action (first timestep of optimal trajectory)
                   ↓
            Environment.step()
                   ↓
         [state, action, reward, done]
                   ↓
    CriticTransformer (value function)
         └→ processes state history → value estimate
                   ↓
            GAE (advantage estimation)
                   ↓
            PPO Update (gradient flow through entire pipeline)
```

### Key Classes

#### 1. **ActorMPC** (`ACMPC/actor.py`)

**Purpose**: Policy network that parameterizes MPC cost functions

**Architecture**:
```python
cost_map_net: Sequential(
    Linear(obs_dim, 512),
    ReLU(),
    Linear(512, 512),
    ReLU(),
    Linear(512, horizon * (nx + nu) * 2)  # Outputs Q and p parameters
)
```

**Important Methods**:
- `forward(x, deterministic=False)`: Generate actions with log probabilities
- `evaluate_actions(x, actions)`: Evaluate log_prob and entropy for given actions
- `_update_cost_module(x)`: Update MPC cost function from network output
- `reset_warm_start()`: Clear MPC's internal warm-start buffer

**Key Attributes**:
- `mpc`: Instance of `DifferentiableMPCController`
- `log_std`: Learnable parameter for action noise (exploration)
- `nx`, `nu`: State and control dimensions
- `horizon`: MPC prediction horizon

**Notes**:
- Supports different `observation_dim` vs `nx` (state dimension)
- Outputs diagonal Q matrices (via `softplus` + small constant for stability)
- Action = MPC mean + Gaussian noise (during training)

#### 2. **CriticTransformer** (`ACMPC/critic_transformer.py`)

**Purpose**: Value function estimator using Transformer architecture

**Architecture**:
```python
state_embed: Linear(nx, d_model)
bert: BertModel (from transformers)
value_head: Linear(d_model, 1)
```

**Input**: Sequence of states `[batch, seq_len, nx]`
**Output**: Value estimates `[batch, seq_len, 1]`

**Why Transformer?**:
- Handles partial observability (POMDPs)
- Direct access to history via self-attention
- No fixed hidden state compression (unlike RNNs)
- Enables within-episode adaptation

**Variant: CriticDecisionTransformer**:
- Also processes actions and timesteps
- Inspired by Decision Transformer architecture
- Better for complex history dependencies

#### 3. **DifferentiableMPCController** (`DifferentialMPC/controller.py`)

**Purpose**: Solve MPC optimization with full backpropagation support

**Algorithm**: Iterative LQR (iLQR)
- **Backward pass**: Riccati recursion for feedback gains
- **Forward pass**: Line search over step sizes [1.0, 0.8, 0.5, 0.2, 0.1]
- **Linearization**: Batched dynamics linearization

**Custom Autograd Function**: `ILQRSolve`
- `forward()`: Solve MPC optimization
- `backward()`: Implicit differentiation of KKT conditions

**Gradient Methods** (`grad_method` parameter):
- `"analytic"`: User-provided Jacobian functions (fastest)
- `"auto_diff"`: PyTorch autograd (flexible, slower)
- `"finite_diff"`: Finite differences (debugging only)

**Constraints**:
- Box constraints on controls: `u_min <= u <= u_max`
- Uses PNQP (Projected Newton QP) solver

**Important Parameters**:
- `reg_eps`: Regularization for numerical stability (default: 1e-2)
- `max_iters`: Maximum iLQR iterations (default: 100)
- `grad_clip`: Gradient clipping threshold

#### 4. **GeneralQuadCost** (`DifferentialMPC/cost.py`)

**Purpose**: Quadratic cost function for MPC

**Cost Structure**:
```
Running cost: 0.5 * (z - z_ref)^T C (z - z_ref) + c^T (z - z_ref)
Terminal cost: 0.5 * (z - z_ref)^T C_final (z - z_ref) + c_final^T (z - z_ref)
where z = [x; u] (stacked state and control)
```

**Methods**:
- `objective(X, U)`: Compute total trajectory cost
- `quadraticize(X, U)`: Return first and second-order derivatives
- `set_reference(x_ref, u_ref)`: Update reference trajectory

#### 5. **Training Loop** (`ACMPC/training_loop.py`)

**Function**: `train_agent(actor, critic, env_manager, ...)`

**Training Phases**:

1. **Rollout Collection**:
   ```python
   for step in range(rollout_length):
       action, log_prob, predicted_X, predicted_U = actor(obs)
       next_obs, reward, done, truncated, info = env.step(action)
       # Store in buffer
   ```

2. **History Buffer Creation**:
   - For Transformer critics, create sequences of recent states/actions
   - Pad sequences to fixed length
   - Shape: `[batch, history_len, feature_dim]`

3. **Advantage Computation**:
   - Get value estimates from critic on full sequences
   - Compute GAE (Generalized Advantage Estimation)
   - Normalize advantages

4. **PPO Updates** (multiple epochs):
   ```python
   for epoch in range(ppo_epochs):
       for mini_batch in shuffle(data):
           # Re-evaluate actions
           new_log_prob, entropy = actor.evaluate_actions(states, old_actions)

           # Compute losses
           ratio = exp(new_log_prob - old_log_prob)
           policy_loss = -min(ratio * advantages,
                              clip(ratio, 1-eps, 1+eps) * advantages)
           value_loss = (returns - critic(states))^2
           entropy_loss = -entropy

           total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

           # Backprop and update
   ```

5. **Checkpointing**:
   - Save best model based on mean reward
   - Keep N most recent checkpoints
   - Save optimizer and scaler states

#### 6. **ParallelEnvManager** (`ACMPC/parallel_env.py`)

**Purpose**: Manage multiple Gym environments in parallel

**Usage**:
```python
env_manager = ParallelEnvManager(env_fn=lambda: MyEnv(), num_envs=24)
obs = env_manager.reset()  # [batch=24, obs_dim]
obs, rewards, dones, truncated, infos = env_manager.step(actions)
```

**Features**:
- Automatic reset on episode termination
- NumPy → PyTorch tensor conversion
- Batched operations for efficiency

#### 7. **CheckpointManager** (`ACMPC/Checkpoint_Manager.py`)

**Purpose**: Save and load training checkpoints

**Usage**:
```python
manager = CheckpointManager(checkpoint_dir="./checkpoints", max_recent=5)

# Save
manager.save_checkpoint(actor, critic, optimizer, step, mean_reward)

# Load
actor, critic, optimizer, scaler = manager.load_checkpoint(actor, critic, optimizer, mode="best")
```

**Features**:
- Tracks best model by reward
- Keeps N most recent checkpoints
- Resets actor's warm-start buffer before saving

---

## Development Workflows

### Setting Up Development Environment

1. **Clone repository**:
   ```bash
   git clone <repo_url>
   cd TACDMPC
   ```

2. **Create virtual environment** (Python 3.11+):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # .venv\Scripts\activate  # Windows
   ```

3. **Install in editable mode with dev dependencies**:
   ```bash
   pip install -e .[dev,examples]
   ```

4. **Verify installation**:
   ```bash
   python -m examples.01_demo_linear_AC
   ```

### Running Examples

**Via Python module**:
```bash
python -m examples.01_demo_linear_AC
python -m examples.02_demo_cartpole_AC
```

**Via CLI** (if installed):
```bash
ac-mpc-examples 01_demo_linear_AC
```

### Adding a New Environment/Task

1. **Create custom Gym environment**:
   ```python
   import gymnasium as gym

   class MyEnv(gym.Env):
       def __init__(self):
           self.observation_space = gym.spaces.Box(low=-inf, high=inf, shape=(obs_dim,))
           self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_dim,))

       def reset(self, seed=None, options=None):
           super().reset(seed=seed)
           obs = ...
           info = {}
           return obs, info

       def step(self, action):
           next_obs = ...
           reward = ...
           terminated = ...
           truncated = ...
           info = {}
           return next_obs, reward, terminated, truncated, info
   ```

2. **Define dynamics model for MPC**:
   ```python
   def f_dyn(x, u, t):
       """
       Args:
           x: [batch, nx] - current state
           u: [batch, nu] - control input
           t: scalar - timestep
       Returns:
           x_next: [batch, nx] - next state
       """
       # Implement discrete-time dynamics
       return x_next

   def f_dyn_jac(x, u, t):
       """
       Returns:
           A: [batch, nx, nx] - ∂f/∂x
           B: [batch, nx, nu] - ∂f/∂u
       """
       # Return analytic Jacobians (optional but faster)
       return A, B
   ```

3. **Create training script**:
   ```python
   from ACMPC import ActorMPC, CriticTransformer, train_agent, ParallelEnvManager
   from DifferentialMPC import DifferentiableMPCController

   # Setup
   device = "cuda" if torch.cuda.is_available() else "cpu"
   nx, nu, horizon = 4, 1, 10

   # Create actor
   actor = ActorMPC(
       nx=nx, nu=nu, horizon=horizon, dt=0.02,
       f_dyn=f_dyn, f_dyn_jac=f_dyn_jac,
       u_min=torch.tensor([-10.0]), u_max=torch.tensor([10.0]),
       observation_dim=obs_dim,  # if different from nx
       grad_method="analytic",  # or "auto_diff"
       device=device
   )

   # Create critic
   critic = CriticTransformer(
       state_dim=nx, d_model=128, nhead=4, num_layers=2,
       history_length=50, device=device
   )

   # Create parallel environments
   env_manager = ParallelEnvManager(
       env_fn=lambda: MyEnv(),
       num_envs=24
   )

   # Train
   train_agent(
       actor=actor, critic=critic, env_manager=env_manager,
       num_steps=500, rollout_length=2048,
       ppo_epochs=10, batch_size=256,
       gamma=0.99, gae_lambda=0.95,
       checkpoint_dir="./checkpoints_my_task"
   )
   ```

### Testing Changes

**Run all tests** (note: currently outdated):
```bash
pytest
```

**Run specific test**:
```bash
pytest tests/test_actor_net.py -v
```

**Run with coverage**:
```bash
pytest --cov=ACMPC --cov=DifferentialMPC --cov-report=html
```

### Code Quality Checks

**Format code** (Black):
```bash
black ACMPC/ DifferentialMPC/ examples/
```

**Lint code** (Ruff):
```bash
ruff check ACMPC/ DifferentialMPC/ examples/
```

### Git Workflow

**Current branch**: `claude/claude-md-mhyski3xjz1spdfv-01YJFawsbTkbXUdnNUqt383q`

**Typical workflow**:
```bash
# Make changes
git add .
git commit -m "Descriptive commit message"

# Push to designated branch
git push -u origin claude/claude-md-mhyski3xjz1spdfv-01YJFawsbTkbXUdnNUqt383q
```

**Recent commits focus on**:
- README updates
- Demo fixes (examples 02, 03)
- Gradient flow fixes
- AC optimization improvements
- Checkpoint improvements

---

## Coding Conventions

### Language & Style

- **Primary Language**: Python 3.11+
- **Comments**: Mix of English and Italian (many comments in Italian)
- **Docstrings**: Inconsistent; prefer adding clear docstrings for new code
- **Line Length**: 88 characters (Black/Ruff standard)
- **Type Hints**: Use extensively; preferred for all function signatures

### Naming Conventions

**Variables**:
- `nx`: State dimension
- `nu`: Control dimension
- `horizon` or `N`: MPC prediction horizon
- `dt`: Time step
- `x`, `u`: State and control vectors
- `X`, `U`: State and control trajectories (sequences)
- `C`, `c`: Cost function matrices/vectors (quadratic and linear terms)
- `f_dyn`: Dynamics function
- `f_dyn_jac`: Dynamics Jacobian function

**Classes**: PascalCase (`ActorMPC`, `CriticTransformer`)
**Functions**: snake_case (`train_agent`, `evaluate_actions`)
**Constants**: UPPER_SNAKE_CASE (`MAX_ITERS`, `DEFAULT_GAMMA`)

### PyTorch Conventions

**Device & dtype**:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32  # Prefer float32 for speed
```

**Batching**: Always use leading batch dimension
```python
x: [batch, nx]
U: [batch, horizon, nu]
```

**Gradients**: Use `torch.autograd.Function` for custom gradients (see `ILQRSolve`)

**Vectorization**: Prefer `torch.func.vmap` over loops
```python
# Good
values = torch.func.vmap(critic)(states)

# Avoid
values = torch.stack([critic(s) for s in states])
```

### MPC-Specific Conventions

**State-control stacking**:
```python
z = torch.cat([x, u], dim=-1)  # [batch, nx + nu]
```

**Trajectory indexing**:
- `X[:, t]`: State at timestep t
- `U[:, t]`: Control at timestep t
- `X[:, :-1]`: All states except terminal
- `X[:, -1]`: Terminal state

**Reference trajectories**:
- `x_ref`: Reference state trajectory `[batch, horizon+1, nx]`
- `u_ref`: Reference control trajectory `[batch, horizon, nu]`

### Error Handling

**Numerical Issues**:
- Add small constants to avoid division by zero: `+ 1e-8`
- Use `torch.clamp()` for bounds
- Regularization for matrix inversions: `A + reg_eps * I`

**Convergence**:
- Always set `max_iters` for iterative solvers
- Optionally detach unconverged solutions

**NaN/Inf Detection**:
```python
assert not torch.isnan(x).any(), "NaN detected in state"
assert torch.isfinite(loss), "Loss is inf/nan"
```

---

## Testing Practices

### Current State

⚠️ **IMPORTANT**: The test suite refers to an old version and is **not currently working**. Tests need to be updated as core logic is finalized.

### Test Structure

**Location**: `tests/`
**Framework**: pytest
**Configuration**: `pyproject.toml` → `[tool.pytest.ini_options]`

**Test Categories**:
1. **Actor tests** (`test_actor_*.py`): Action generation, gradients, warm-start
2. **Critic tests** (`test_critic_*.py`): Forward pass, value estimation
3. **MPC tests** (`test_forward.py`, `test_gradcheck.py`): MPC solver correctness
4. **Training tests** (`test_training_loop.py`): End-to-end training
5. **Integration tests** (`test_example_*.py`): Full examples

### Writing New Tests

**Template**:
```python
import pytest
import torch
from ACMPC import ActorMPC

@pytest.fixture
def actor(device):
    return ActorMPC(nx=2, nu=1, horizon=10, dt=0.1,
                    f_dyn=..., device=device)

def test_actor_forward_pass(actor, device):
    """Test that actor generates valid actions."""
    obs = torch.randn(16, 2, device=device)  # batch=16, obs_dim=2
    action, log_prob, pred_X, pred_U = actor(obs)

    assert action.shape == (16, 1), "Action shape mismatch"
    assert log_prob.shape == (16,), "Log prob shape mismatch"
    assert not torch.isnan(action).any(), "Action contains NaN"
    assert torch.isfinite(log_prob).all(), "Log prob not finite"
```

**Running tests**:
```bash
pytest tests/test_actor_net.py::test_actor_forward_pass -v
```

### Testing Best Practices

1. **Use fixtures** for common objects (actor, critic, env)
2. **Test shapes** before values
3. **Test edge cases**: empty batches, extreme values
4. **Test gradients**: Use `torch.autograd.gradcheck()` for custom functions
5. **Mock slow operations**: Use `unittest.mock` for long training runs

---

## Common Tasks

### Task 1: Debugging Convergence Issues

**Symptoms**: Agent not learning, rewards flat, NaN losses

**Checklist**:
1. ✅ Check reward scale (should be roughly -100 to +100)
2. ✅ Verify MPC converges: Check `converged` flag in controller
3. ✅ Inspect gradients: Add gradient clipping, check for NaN
4. ✅ Reduce learning rate (try 1e-4 or 1e-5)
5. ✅ Increase `reg_eps` in MPC (try 1e-1)
6. ✅ Check cost function parameterization: Are Q values positive?
7. ✅ Verify dynamics model: Does it match reality?
8. ✅ Check observation normalization: Use `VecNormalize` if needed

**Debugging commands**:
```python
# Print MPC solve statistics
print(f"Converged: {converged}, Final cost: {final_cost}")

# Check gradients
for name, param in actor.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm()}")

# Visualize cost function
q_params = actor.cost_map_net(obs)
print(f"Q values: {F.softplus(q_params[:, :nx])}")
```

### Task 2: Speeding Up Training

**Strategies**:
1. **Use analytic Jacobians**: Set `grad_method="analytic"` and provide `f_dyn_jac`
2. **Increase batch size**: More parallel environments (24-64)
3. **Mixed precision**: Enable AMP (`use_amp=True`)
4. **Reduce MPC horizon**: Shorter horizons = faster solve
5. **Fewer PPO epochs**: Try 5-8 instead of 10
6. **Profile code**: Use `utils.profiler.Profiler`

**Example profiling**:
```python
from utils.profiler import Profiler

with Profiler("training_step", device="cuda") as prof:
    loss.backward()
    optimizer.step()

print(prof.summary())  # Shows time, memory usage
```

### Task 3: Adding Custom Cost Functions

**Steps**:
1. Create new cost class inheriting from `nn.Module`
2. Implement `objective(X, U)` and `quadraticize(X, U)` methods
3. Update `ActorMPC._update_cost_module()` to set cost parameters

**Example**:
```python
class CustomCost(nn.Module):
    def __init__(self, nx, nu):
        super().__init__()
        self.nx, self.nu = nx, nu

    def objective(self, X, U):
        """Compute total cost for trajectory."""
        # X: [batch, horizon+1, nx]
        # U: [batch, horizon, nu]
        cost = ...  # Your custom cost
        return cost  # [batch]

    def quadraticize(self, X, U):
        """Return quadratic approximation at (X, U)."""
        # Compute second-order Taylor expansion
        l_x = ...  # [batch, horizon, nx]
        l_u = ...  # [batch, horizon, nu]
        l_xx = ... # [batch, horizon, nx, nx]
        l_uu = ... # [batch, horizon, nu, nu]
        l_ux = ... # [batch, horizon, nu, nx]
        return l_x, l_u, l_xx, l_uu, l_ux
```

### Task 4: Implementing a New Critic Architecture

**Steps**:
1. Create class inheriting from `nn.Module`
2. Implement `forward(states)` returning value estimates
3. Ensure output shape matches `[batch, seq_len, 1]` or `[batch, 1]`

**Template**:
```python
class MyCustomCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, states):
        """
        Args:
            states: [batch, seq_len, state_dim] or [batch, state_dim]
        Returns:
            values: [batch, seq_len, 1] or [batch, 1]
        """
        return self.net(states)
```

### Task 5: Tuning Hyperparameters

**Critical hyperparameters**:

**PPO**:
- `learning_rate`: 1e-4 to 3e-4 (default: 3e-4)
- `ppo_epochs`: 5-10 (default: 10)
- `batch_size`: 64-512 (default: 256)
- `clip_epsilon`: 0.1-0.3 (default: 0.2)
- `entropy_coef`: 0.001-0.01 (default: 0.01)
- `value_coef`: 0.5 (default: 0.5)
- `gamma`: 0.95-0.99 (default: 0.99)
- `gae_lambda`: 0.9-0.98 (default: 0.95)

**MPC**:
- `horizon`: 5-20 (trade-off: longer = better but slower)
- `reg_eps`: 1e-3 to 1e-1 (increase if numerical issues)
- `max_iters`: 50-200 (default: 100)
- `grad_clip`: 1.0-10.0 (default: 5.0)

**Actor**:
- `log_std` initial value: -1.0 to 1.0 (controls exploration)
- Network width: 256-1024 (default: 512)

**Critic**:
- `d_model`: 64-256 (default: 128)
- `nhead`: 4-8 (default: 4)
- `num_layers`: 2-4 (default: 2)
- `history_length`: 20-100 (default: 50)

**Tuning process**:
1. Start with defaults
2. If not converging: increase `reg_eps`, decrease learning rate
3. If converging slowly: increase batch size, learning rate
4. If overfitting: increase entropy coefficient, reduce PPO epochs
5. If exploration poor: increase `log_std` initial value

---

## Important Notes & Gotchas

### 1. Observation Dimension vs. State Dimension

**Key Distinction**:
- `observation_dim`: What the actor network sees (can include extra info like waypoints)
- `nx` (state dimension): Physical state used by MPC dynamics

**Example**:
```python
# Unicycle task
nx = 3  # [x, y, theta]
observation_dim = 5  # [x, y, theta, waypoint_x, waypoint_y]

actor = ActorMPC(nx=3, observation_dim=5, ...)
```

**Inside ActorMPC**:
```python
# Network input: full observation
params = self.cost_map_net(x)  # x: [batch, observation_dim]

# MPC uses only physical state
predicted_states, predicted_actions = self.mpc(x[:, :self.nx], U_init)
```

### 2. Control Bounds

**MUST be PyTorch tensors**, not floats:
```python
# ✅ Correct
u_min = torch.tensor([-10.0])
u_max = torch.tensor([10.0])

# ❌ Wrong
u_min = -10.0
u_max = 10.0
```

### 3. Warm-Starting Behavior

**Issue**: MPC maintains internal buffers for warm-starting. If not reset, gradients can flow incorrectly.

**Solution**: Reset before saving checkpoints:
```python
actor.reset_warm_start()  # Clear internal MPC buffers
checkpoint_manager.save_checkpoint(actor, ...)
```

### 4. Cost Function Output

**Actor outputs diagonal Q matrices**:
```python
q_params_raw = cost_map_net(obs)  # Raw network output
q_diag = F.softplus(q_params_raw) + 1e-2  # Ensure positive, avoid zero
C = torch.diag_embed(q_diag)  # Convert to diagonal matrices
```

**Why softplus?**: Ensures Q matrices are positive semi-definite (required for convexity).

### 5. Gradient Methods Trade-off

| Method | Speed | Flexibility | Accuracy |
|--------|-------|-------------|----------|
| `analytic` | ⚡⚡⚡ | ❌ (requires manual Jacobian) | ✅✅✅ |
| `auto_diff` | ⚡ | ✅✅✅ (works with any dynamics) | ✅✅ |
| `finite_diff` | 🐌 | ✅✅ | ✅ (but noisy) |

**Recommendation**: Use `analytic` if you can provide Jacobians, otherwise `auto_diff`.

### 6. History Buffer Creation

**For Transformer critics**, you need to create sequences:
```python
# In training_loop.py
state_history_buffer = create_history_buffer(
    states_flat, history_length=50, nx=nx
)
# Shape: [num_rollout_steps, history_length, nx]
```

**Padding**: Earlier timesteps are zero-padded (no history yet).

### 7. Mixed Precision Training

**Enable with caution**:
```python
train_agent(..., use_amp=True)  # Automatic Mixed Precision
```

**Potential issues**:
- MPC solver may be less stable with float16
- Gradients may underflow
- Use `GradScaler` (handled automatically in training loop)

### 8. Episode Truncation vs. Termination

**New Gym API** (gymnasium):
- `terminated`: True when episode ends naturally (goal reached, failure)
- `truncated`: True when episode times out (max steps reached)

**Handling in training**:
```python
# Don't bootstrap value if truly terminated
if not terminated:
    last_value = critic(last_state)
    returns[-1] += gamma * last_value
```

### 9. Checkpoint Loading

**Two modes**:
- `mode="best"`: Load best model by reward
- `mode="latest"`: Load most recent checkpoint

**Usage**:
```python
actor, critic, optimizer, scaler = checkpoint_manager.load_checkpoint(
    actor, critic, optimizer, mode="best"
)
```

### 10. Deprecated Code

**DO NOT USE**:
- `DeprecatedActorCritic/`: Old implementations, not compatible
- `DepracatedKoopmanAutoEncoders/`: Experimental, incomplete integration
- Old example scripts (without `_AC` suffix)

### 11. Italian Comments

Many comments are in Italian. Common terms:
- "obiettivo" = objective
- "passo" = step
- "aggiornamento" = update
- "rete" = network
- "costo" = cost
- "vincoli" = constraints
- "gradiente" = gradient

When adding code, prefer English for consistency with new contributions.

### 12. Tests Are Outdated

**From README**:
> "NOTE THE TEST REFER TO AN OLD VERSION NOT WORKING ANYMORE WILL BE UPDATED AS CORE LOGIC IS FINALIZED"

**Implication**: Do not rely on tests passing/failing. Validate changes manually with example scripts.

---

## Key Files Reference

### Must-Read Files

1. **`README.md`**: User documentation, architecture overview
2. **`ACMPC/actor.py`**: ActorMPC implementation (core policy)
3. **`ACMPC/critic_transformer.py`**: Critic architectures
4. **`ACMPC/training_loop.py`**: End-to-end PPO training
5. **`DifferentialMPC/controller.py`**: MPC solver + custom gradients
6. **`DifferentialMPC/cost.py`**: Cost function modules
7. **`examples/02_demo_cartpole_AC.py`**: Best working example

### Configuration Files

- **`pyproject.toml`**: Package metadata, dependencies, tool configs
- **`requirements.txt`**: Minimal runtime dependencies
- **`.gitignore`**: Ignored files/directories

### Utilities

- **`utils/profiler.py`**: Performance profiling context manager
- **`ACMPC/parallel_env.py`**: Parallel environment manager
- **`ACMPC/Checkpoint_Manager.py`**: Checkpoint save/load

### Entry Points

- **`examples/__main__.py`**: CLI entry (`ac-mpc-examples` command)
- **`examples/01_demo_linear_AC.py`**: Simplest working example
- **`examples/02_demo_cartpole_AC.py`**: Most complete example

---

## Quick Reference: Common Code Patterns

### Creating an Actor

```python
from ACMPC import ActorMPC
import torch

actor = ActorMPC(
    nx=4,                        # State dimension
    nu=1,                        # Control dimension
    horizon=10,                  # MPC horizon
    dt=0.02,                     # Time step
    f_dyn=my_dynamics_function,  # Dynamics model
    f_dyn_jac=my_jacobian_function,  # Optional, for speed
    u_min=torch.tensor([-10.0]), # Control lower bound
    u_max=torch.tensor([10.0]),  # Control upper bound
    observation_dim=6,           # If different from nx
    grad_method="analytic",      # or "auto_diff"
    device="cuda"
)
```

### Creating a Critic

```python
from ACMPC import CriticTransformer

critic = CriticTransformer(
    state_dim=4,
    d_model=128,
    nhead=4,
    num_layers=2,
    history_length=50,
    device="cuda"
)
```

### Training Loop

```python
from ACMPC import train_agent, ParallelEnvManager

env_manager = ParallelEnvManager(
    env_fn=lambda: MyEnv(),
    num_envs=24
)

train_agent(
    actor=actor,
    critic=critic,
    env_manager=env_manager,
    num_steps=500,
    rollout_length=2048,
    ppo_epochs=10,
    batch_size=256,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    checkpoint_dir="./checkpoints"
)
```

### Dynamics Function Template

```python
def f_dyn(x, u, t):
    """
    Discrete-time dynamics: x_{t+1} = f(x_t, u_t, t)

    Args:
        x: [batch, nx] - current state
        u: [batch, nu] - control input
        t: scalar - timestep
    Returns:
        x_next: [batch, nx] - next state
    """
    # Example: double integrator
    # state: [position, velocity]
    # control: [acceleration]
    dt = 0.02
    x_next = x.clone()
    x_next[:, 0] = x[:, 0] + dt * x[:, 1]  # position += dt * velocity
    x_next[:, 1] = x[:, 1] + dt * u[:, 0]  # velocity += dt * acceleration
    return x_next

def f_dyn_jac(x, u, t):
    """
    Jacobians of dynamics.

    Returns:
        A: [batch, nx, nx] - ∂f/∂x
        B: [batch, nx, nu] - ∂f/∂u
    """
    batch = x.shape[0]
    dt = 0.02

    A = torch.zeros(batch, 2, 2, device=x.device)
    A[:, 0, 0] = 1.0
    A[:, 0, 1] = dt
    A[:, 1, 1] = 1.0

    B = torch.zeros(batch, 2, 1, device=x.device)
    B[:, 1, 0] = dt

    return A, B
```

---

## Summary

TACDMPC is a sophisticated reinforcement learning framework that combines:
- **Structured policies** (MPC-based) for sample efficiency and safety
- **End-to-end learning** via differentiable optimization layers
- **History-aware value functions** using Transformer architectures

**Key strengths**:
- Interpretable policies (via learned cost functions)
- Respects known system dynamics
- Handles partial observability

**Key challenges**:
- Requires a dynamics model (analytic or learned)
- Numerically sensitive (requires tuning regularization)
- Computationally expensive (MPC solve in inner loop)

**Best for**:
- Robotics and control tasks
- Environments with known or learnable physics
- Applications requiring safety/constraint satisfaction

**Not ideal for**:
- High-dimensional action spaces (>10D controls)
- Discrete action spaces
- Pure model-free settings

---

**Last Updated**: 2025-11-14
**Maintained by**: TACDMPC Team
**For Questions**: See issues at repository homepage

---
