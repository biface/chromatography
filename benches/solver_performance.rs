//! Performance benchmarks for numerical solvers
//!
//! This benchmark compares Euler and RK4 solvers on identical problems
//! to measure their relative performance characteristics.
//!
//! # What We're Measuring
//!
//! 1. **Euler solver** (Forward Euler):
//!    - 1st order accuracy: O(dt)
//!    - 1 function evaluation per step
//!    - Fast but requires small dt for accuracy
//!
//! 2. **RK4 solver** (Runge-Kutta 4):
//!    - 4th order accuracy: O(dt⁴)
//!    - 4 function evaluations per step
//!    - Slower per step but more accurate
//!
//! # Expected Results
//!
//! **Performance ratio**: RK4 ≈ 4× slower than Euler
//! - Same problem, same accuracy requirements
//! - RK4 does 4 evaluations vs Euler's 1
//! - Linear scaling with problem size
//!
//! **Scaling with problem size**:
//! - Time ∝ points (spatial discretization)
//! - Time ∝ time_steps (temporal discretization)
//!
//! # Running Benchmarks
//!
//! ```bash
//! # Run all solver benchmarks
//! cargo bench --bench solver_performance
//!
//! # Run only Euler tests
//! cargo bench --bench solver_performance euler
//!
//! # Run only RK4 tests
//! cargo bench --bench solver_performance rk4
//!
//! # Direct comparison
//! cargo bench --bench solver_performance comparison
//! ```
//!
//! # Understanding Results
//!
//! Example output interpretation:
//!
//! ```text
//! solver_comparison/euler
//!   Time: [15.234 ms 15.456 ms 15.678 ms]
//!
//! solver_comparison/rk4
//!   Time: [61.123 ms 61.456 ms 61.789 ms]
//!
//! Ratio: 61.456 / 15.456 ≈ 3.98 ≈ 4.0× (expected!)
//! ```
//!
//! If ratio differs significantly from 4.0×:
//! - > 5×: Extra overhead in RK4 (cache misses, allocations)
//! - < 3×: Unexpected optimization (check black_box usage)

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, SamplingMode};
use std::hint::black_box;
use std::time::Duration;
use chrom_rs::solver::{Solver, SolverConfiguration, Scenario, DomainBoundaries};
use chrom_rs::solver::{EulerSolver, RK4Solver};
use chrom_rs::physics::{PhysicalModel, PhysicalState, PhysicalQuantity, PhysicalData};

// =================================================================================================
// Simple Model for Benchmarking
// =================================================================================================

/// Simple physical model for benchmarking purposes
///
/// Implements exponential decay: dy/dt = -k*y with k = 0.1
///
/// # Why This Model?
///
/// - **Simplicity**: Easy to compute, no complex physics
/// - **Predictable**: Known analytical solution y(t) = y₀ * exp(-0.1*t)
/// - **Scalability**: Works with any number of spatial points
/// - **Pure benchmark**: Isolates solver performance (not physics complexity)
///
/// # Mathematical Background
///
/// ```text
/// dy/dt = -k*y    where k = 0.1
/// ```
///
/// Solution: y(t) = y₀ * exp(-0.1*t)
///
/// This is a **stiff problem** for k > 1, but k=0.1 is mild,
/// making it suitable for testing basic solver performance.
struct SimpleModel {
    points: usize,
}

impl PhysicalModel for SimpleModel {
    fn points(&self) -> usize {
        self.points
    }

    fn compute_physics(&self, state: &PhysicalState) -> PhysicalState {
        let mut result = state.clone();
        if let Some(conc) = result.get_mut(PhysicalQuantity::Concentration) {
            conc.apply(|x| -0.1 * x);
        }
        result
    }

    fn setup_initial_state(&self) -> PhysicalState {
        PhysicalState::new(
            PhysicalQuantity::Concentration,
            PhysicalData::uniform_vector(self.points, 1.0),
        )
    }

    fn name(&self) -> &str {
        "Simple Model"
    }
}

// =================================================================================================
// Benchmark Functions
// =================================================================================================

/// Benchmark Euler solver with different problem sizes
///
/// Tests performance scaling with spatial discretization (number of points).
///
/// # Test Configuration
///
/// - **Points**: 10, 50, 100, 500 (spatial discretization)
/// - **Time steps**: 100 (fixed for fair comparison)
/// - **Total time**: 10.0 seconds
/// - **dt**: 0.1 seconds per step
///
/// # Expected Scaling
///
/// Time should scale linearly with points:
///
/// ```text
/// points=10:   baseline (e.g., 0.5 ms)
/// points=50:   ~5× slower (e.g., 2.5 ms)
/// points=100:  ~10× slower (e.g., 5.0 ms)
/// points=500:  ~50× slower (e.g., 25 ms)
/// ```
///
/// # Why These Sizes?
///
/// - **10 points**: Minimal problem (cache-friendly)
/// - **50 points**: Small but realistic
/// - **100 points**: Standard medium size
/// - **500 points**: Large problem (may exceed L1 cache)
///
/// If scaling is **not linear**, investigate:
/// - Cache effects (L1 → L2 → L3 → RAM)
/// - Memory allocation overhead
/// - SIMD vectorization thresholds
fn benchmark_euler_solver(c: &mut Criterion) {
    let mut group = c.benchmark_group("Forward Euler Solver");

    for points in [10, 50, 100, 500].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(points),
            points,
            |b, &points| {
                // Setup phase (NOT measured by criterion)
                // ==========================================
                
                // Create model with specified number of points
                let model = Box::new(SimpleModel { points });
                
                // Setup initial state: all points at concentration = 1.0
                let initial = model.setup_initial_state();
                
                // Create temporal domain (ODE problem)
                let boundaries = DomainBoundaries::temporal(initial);
                
                // Package model + boundaries into scenario
                let scenario = Scenario::new(model, boundaries);
                
                // Configure solver: 10 seconds, 100 steps → dt = 0.1s
                let config = SolverConfiguration::time_evolution(10.0, 100);
                
                // Create solver instance
                let solver = EulerSolver::new();

                // Measurement phase (THIS is what criterion measures)
                // ====================================================
                b.iter(|| {
                    // black_box prevents compiler from:
                    // 1. Caching the result across iterations
                    // 2. Eliminating "unused" computations
                    // 3. Inlining everything and optimizing away
                    solver.solve(black_box(&scenario), black_box(&config)).unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark RK4 solver with different problem sizes
///
/// Tests performance scaling with spatial discretization (number of points).
///
/// # Test Configuration
///
/// - **Points**: 10, 50, 100, 500 (spatial discretization)
/// - **Time steps**: 100 (fixed, same as Euler for comparison)
/// - **Total time**: 10.0 seconds
/// - **dt**: 0.1 seconds per step
///
/// # Expected Scaling
///
/// Time should scale linearly with points, but ~4× slower than Euler:
///
/// ```text
/// points=10:   ~4× Euler baseline (e.g., 2.0 ms)
/// points=50:   ~4× Euler at 50 (e.g., 10 ms)
/// points=100:  ~4× Euler at 100 (e.g., 20 ms)
/// points=500:  ~4× Euler at 500 (e.g., 100 ms)
/// ```
///
/// # RK4 Characteristics
///
/// - **4 function evaluations** per step vs Euler's 1
/// - **Higher order accuracy**: O(dt⁴) vs O(dt)
/// - **Trade-off**: 4× slower but can use larger dt for same accuracy
///
/// # Practical Implication
///
/// For same **accuracy**, RK4 may be faster overall:
/// - RK4 with dt=0.1: 4 evaluations per step
/// - Euler with dt=0.025 (for same accuracy): 4 steps × 1 evaluation = 4 evaluations
/// - But RK4 needs less memory (fewer stored states)
fn benchmark_rk4_solver(c: &mut Criterion) {
    let mut group = c.benchmark_group("Runge-Kutta (4 steps) Solver");

    for points in [10, 50, 100, 500].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(points),
            points,
            |b, &points| {
                let model = Box::new(SimpleModel { points });
                let initial = model.setup_initial_state();
                let boundaries = DomainBoundaries::temporal(initial);
                let scenario = Scenario::new(model, boundaries);
                let config = SolverConfiguration::time_evolution(10.0, 100);
                let solver = RK4Solver::new();

                b.iter(|| {
                    solver.solve(black_box(&scenario), black_box(&config)).unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Direct comparison between Euler and RK4 solvers across problem sizes
///
/// Tests both solvers on **multiple configurations** of spatial and temporal
/// discretization to understand their performance characteristics.
///
/// # Test Strategy
///
/// We test **key configurations** that represent different use cases:
///
/// 1. **Small/Fast** (50 pts, 100 steps): Quick prototyping
/// 2. **Medium/Standard** (100 pts, 1000 steps): Typical simulation
/// 3. **Large/Accurate** (200 pts, 5000 steps): Publication quality
/// 4. **XLarge/Extreme** (500 pts, 10000 steps): High-resolution research
///
/// For each configuration, we measure:
/// - Euler solver time
/// - RK4 solver time
/// - Ratio RK4/Euler (should be ≈ 4.0)
///
/// # What We're Learning
///
/// 1. **Does 4× ratio hold across problem sizes?**
///    - If yes: RK4 overhead is pure function evaluations
///    - If no: Cache effects or memory bandwidth bottleneck
///
/// 2. **Does scaling remain linear?**
///    - Time ∝ (points × time_steps) for both solvers?
///
/// 3. **Where's the crossover point?**
///    - When does problem become memory-bound vs compute-bound?
///
/// # Reading Results
///
/// Criterion will output results like:
///
/// ```text
/// solver_comparison/euler_50pts_100steps
///   Time: [1.234 ms 1.256 ms 1.278 ms]
///   Throughput: 3980 Kelem/s
///
/// solver_comparison/rk4_50pts_100steps
///   Time: [4.912 ms 4.956 ms 5.001 ms]
///   Throughput: 1010 Kelem/s
///   
/// Ratio: 4.956 / 1.256 = 3.95 ≈ 4.0 ✅
/// ```
///
/// # Performance Targets
///
/// Expected times (approximate, hardware-dependent):
///
/// | Config | Euler | RK4 | Ratio |
/// |--------|-------|-----|-------|
/// | 50×100 | 1-2 ms | 4-8 ms | 4.0 |
/// | 100×1000 | 15-20 ms | 60-80 ms | 4.0 |
/// | 200×5000 | 150-200 ms | 600-800 ms | 4.0 |
/// | 500×10000 | 1.5-2 s | 6-8 s | 4.0 |
///
/// # Troubleshooting
///
/// **If ratio ≠ 4.0 for large problems**:
/// - Check L3 cache size (maybe exceeding)
/// - Profile with `perf stat` to see cache misses
/// - Consider memory bandwidth limitations
///
/// **If times are much slower than expected**:
/// - Verify `cargo bench` runs in release mode (it should)
/// - Check CPU throttling: `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`
/// - Close background applications
fn benchmark_solver_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Solver Comparison");

    // Define test configurations: (points, time_steps, label)
    // These represent realistic use cases from fast prototyping to high-accuracy
    let configurations = vec![
        (50, 100, "small"),      // Fast prototyping
        (100, 1000, "medium"),   // Standard simulation
        (200, 5000, "large"),    // Publication quality
        // (500, 10000, "xlarge"),  // High-resolution research
    ];

    // Test each configuration with both solvers
    for (points, time_steps, label) in configurations {
        // Calculate total time based on problem size
        // We use dt ≈ 0.1s, so total_time = time_steps * 0.1
        let total_time = (time_steps as f64) * 0.1;
        
        // Calculate throughput metric: total operations
        // Operations = points × time_steps × evaluations_per_step
        // This allows Criterion to report "Melem/s" throughput
        let ops_euler = (points * time_steps) as u64;      // 1 eval per step
        let ops_rk4 = (points * time_steps * 4) as u64;    // 4 evals per step
        
        // Benchmark Euler solver
        {
            // Setup (not measured)
            let model = Box::new(SimpleModel { points });
            let initial = model.setup_initial_state();
            let boundaries = DomainBoundaries::temporal(initial);
            let scenario = Scenario::new(model, boundaries);
            let config = SolverConfiguration::time_evolution(total_time, time_steps);
            let solver = EulerSolver::new();
            
            // Set throughput for this specific test
            // Criterion will display results as "X Melem/s"
            group.throughput(criterion::Throughput::Elements(ops_euler));
            
            // Benchmark ID format: "euler_50pts_100steps"
            group.bench_function(
                format!("Forward Euler {} points & {} steps", points, time_steps),
                |b| {
                    b.iter(|| {
                        solver.solve(
                            black_box(&scenario),
                            black_box(&config)
                        ).unwrap()
                    });
                },
            );
        }
        
        // Benchmark RK4 solver (same problem)
        {
            // Setup (not measured)
            let model = Box::new(SimpleModel { points });
            let initial = model.setup_initial_state();
            let boundaries = DomainBoundaries::temporal(initial);
            let scenario = Scenario::new(model, boundaries);
            let config = SolverConfiguration::time_evolution(total_time, time_steps);
            let solver = RK4Solver::new();
            
            // RK4 does 4× more operations, so throughput is different
            group.throughput(criterion::Throughput::Elements(ops_rk4));
            
            group.bench_function(
                format!("Runge-Kutta 4 {} points & {} steps", points, time_steps),
                |b| {
                    b.iter(|| {
                        solver.solve(
                            black_box(&scenario),
                            black_box(&config)
                        ).unwrap()
                    });
                },
            );
        }
    }

    group.finish();
}

// =================================================================================================
// Criterion Configuration
// =================================================================================================

/// Optional: Exhaustive comparison across ALL combinations
///
/// This benchmarks **every combination** of points and time_steps.
/// ⚠️ WARNING: This generates MANY benchmarks and takes a long time!
///
/// # When to Use
///
/// - Detailed performance analysis
/// - Finding cache transition points
/// - Creating performance heatmaps
///
/// # To Enable
///
/// Uncomment this function in `criterion_group!` below:
///
/// ```rust
/// criterion_group!(
///     benches,
///     benchmark_euler_solver,
///     benchmark_rk4_solver,
///     benchmark_solver_comparison,
///     // benchmark_exhaustive_comparison,  // ← Uncomment this
/// );
/// ```
///
/// # Results Analysis
///
/// After running, you can create a heatmap of performance:
///
/// ```python
/// import pandas as pd
/// import seaborn as sns
///
/// # Extract times from criterion output
/// # Create matrix: rows=points, cols=time_steps
/// # Plot heatmap to see performance landscape
/// ```
#[allow(dead_code)]
fn benchmark_exhaustive_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Exhaustive Solver Comparison");

    // Configuration large treatments
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(50);
    group.warm_up_time(Duration::from_secs(5));

    // Progression

    group.significance_level(0.1);
    
    // Fine-grained test points
    let points_values = vec![50, 100, 200];
    let time_steps_values = vec![100, 500, 1000, 5000, 10000];
    
    println!("\n🔬 Running exhaustive comparison:");
    println!("   {} points configs × {} time_steps configs × 2 solvers", 
             points_values.len(), time_steps_values.len());
    println!("   = {} benchmarks total", 
             points_values.len() * time_steps_values.len() * 2);
    println!("   This will take several minutes...\n");
    
    // Test all combinations
    for &points in &points_values {
        for &time_steps in &time_steps_values {
            let total_time = (time_steps as f64) * 0.1;
            
            // Test Euler
            {
                let model = Box::new(SimpleModel { points });
                let initial = model.setup_initial_state();
                let boundaries = DomainBoundaries::temporal(initial);
                let scenario = Scenario::new(model, boundaries);
                let config = SolverConfiguration::time_evolution(total_time, time_steps);
                let solver = EulerSolver::new();
                
                group.throughput(criterion::Throughput::Elements(
                    (points * time_steps) as u64
                ));
                
                group.bench_with_input(
                    criterion::BenchmarkId::new(
                        "Forward Euler",
                        format!("{}x{}", points, time_steps)
                    ),
                    &(points, time_steps),
                    |b, _| {
                        b.iter(|| {
                            solver.solve(
                                black_box(&scenario),
                                black_box(&config)
                            ).unwrap()
                        });
                    },
                );
            }
            
            // Test RK4
            {
                let model = Box::new(SimpleModel { points });
                let initial = model.setup_initial_state();
                let boundaries = DomainBoundaries::temporal(initial);
                let scenario = Scenario::new(model, boundaries);
                let config = SolverConfiguration::time_evolution(total_time, time_steps);
                let solver = RK4Solver::new();
                
                group.throughput(criterion::Throughput::Elements(
                    (points * time_steps * 4) as u64
                ));
                
                group.bench_with_input(
                    criterion::BenchmarkId::new(
                        "Runge-Kutta",
                        format!("{}x{}", points, time_steps)
                    ),
                    &(points, time_steps),
                    |b, _| {
                        b.iter(|| {
                            solver.solve(
                                black_box(&scenario),
                                black_box(&config)
                            ).unwrap()
                        });
                    },
                );
            }
        }
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_euler_solver,
    benchmark_rk4_solver,
    benchmark_solver_comparison,
    // Uncomment below for exhaustive testing (takes much longer!)
    benchmark_exhaustive_comparison,
);
criterion_main!(benches);

// =================================================================================================
// Learning Notes & Debugging Guide
// =================================================================================================

/*
# Understanding Benchmark Results

## 1. Reading Criterion Output

When you run `cargo bench --bench solver_performance`, you'll see:

```
euler_solver/10         time:   [485.23 µs 487.45 µs 489.67 µs]
                        ^^^^^^   ^^^^^^^^  ^^^^^^^^  ^^^^^^^^
                        label    lower     mean      upper (95% confidence)

euler_solver/50         time:   [2.4123 ms 2.4567 ms 2.5012 ms]
                        change: [+395.2% +404.0% +412.8%] (p = 0.00 < 0.05)
                                ^^^^^^^^^^^^^^^^^^^^^^^^
                                Comparison with previous run

rk4_solver/10          time:   [1.9234 ms 1.9456 ms 1.9678 ms]
```

## 2. Verifying Linear Scaling

To check if scaling is O(n), compute ratios:

```
Euler times:
  10 points:   0.487 ms  → baseline
  50 points:   2.457 ms  → 2.457 / 0.487 = 5.04× ✅ (expected 5×)
  100 points:  4.912 ms  → 4.912 / 0.487 = 10.08× ✅ (expected 10×)
  500 points:  24.56 ms  → 24.56 / 0.487 = 50.4× ✅ (expected 50×)
```

If ratios don't match:
- Cache effects (look for jumps at L1→L2 boundary)
- Memory allocation overhead
- Non-linear algorithms accidentally introduced

## 2b. Analyzing Multi-Configuration Comparison Results

The `solver_comparison` benchmark tests multiple (points, time_steps) combinations.
Here's how to analyze the results:

### Step 1: Extract Times

```
solver_comparison/euler_50pts_100steps    time: [1.256 ms ...]
solver_comparison/euler_100pts_1000steps  time: [15.456 ms ...]
solver_comparison/euler_200pts_5000steps  time: [154.23 ms ...]
solver_comparison/euler_500pts_10000steps time: [1.5423 s ...]

solver_comparison/rk4_50pts_100steps      time: [4.956 ms ...]
solver_comparison/rk4_100pts_1000steps    time: [61.456 ms ...]
solver_comparison/rk4_200pts_5000steps    time: [616.89 ms ...]
solver_comparison/rk4_500pts_10000steps   time: [6.1689 s ...]
```

### Step 2: Verify 4× Ratio Across All Sizes

```
Config         | Euler    | RK4      | Ratio      | Status
---------------|----------|----------|------------|--------
50 × 100       | 1.256 ms | 4.956 ms | 3.95       | ✅ ≈ 4.0
100 × 1000     | 15.46 ms | 61.46 ms | 3.98       | ✅ ≈ 4.0
200 × 5000     | 154.2 ms | 616.9 ms | 4.00       | ✅ = 4.0
500 × 10000    | 1.542 s  | 6.169 s  | 4.00       | ✅ = 4.0
```

**What this tells you:**
- If ratio ≈ 4.0 everywhere: Perfect! RK4 overhead is pure function evals
- If ratio increases with size: Cache effects or memory bandwidth limit
- If ratio decreases with size: Unexpected compiler optimizations

### Step 3: Check Linear Scaling (Total Operations)

Calculate "cost per operation" for each configuration:

```
Total operations = points × time_steps

Config         | Ops       | Euler time | ns/op (Euler) | ns/op (RK4)
---------------|-----------|------------|---------------|-------------
50 × 100       | 5,000     | 1.256 ms   | 251.2 ns      | 991.2 ns
100 × 1000     | 100,000   | 15.46 ms   | 154.6 ns      | 614.6 ns
200 × 5000     | 1,000,000 | 154.2 ms   | 154.2 ns      | 616.9 ns
500 × 10000    | 5,000,000 | 1.542 s    | 308.4 ns      | 1233.8 ns
```

**Ideal behavior**: ns/op should be constant (or close)

**Reality check**:
- Small problems (5k ops): Higher overhead (251 ns/op) due to setup
- Medium problems (100k-1M ops): Optimal (154 ns/op) - fits in cache
- Large problems (5M ops): Higher again (308 ns/op) - cache misses

This is **normal** and expected!

### Step 4: Visualizing Results

Create a simple Python script to visualize:

```python
import matplotlib.pyplot as plt

# Data from benchmarks
points = [50, 100, 200, 500]
euler_times = [1.256, 15.46, 154.2, 1542]
rk4_times = [4.956, 61.46, 616.9, 6169]

plt.figure(figsize=(10, 6))
plt.loglog(points, euler_times, 'o-', label='Euler', linewidth=2)
plt.loglog(points, rk4_times, 's-', label='RK4', linewidth=2)
plt.xlabel('Number of spatial points')
plt.ylabel('Time (ms)')
plt.title('Solver Performance vs Problem Size')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.savefig('solver_scaling.png', dpi=300, bbox_inches='tight')
```

**Expected plot**: Two parallel lines (constant ratio)

### Step 5: Using Throughput Metrics

Criterion reports throughput automatically:

```
euler_50pts_100steps
  Throughput: 3.98 Melem/s

euler_100pts_1000steps  
  Throughput: 6.47 Melem/s  ← Higher is better!

rk4_50pts_100steps
  Throughput: 1.01 Melem/s  ← Accounts for 4× operations
```

**Note**: RK4 throughput includes 4× function evaluations in calculation

## 3. Verifying 4× Ratio (Euler vs RK4)

From solver_comparison results:

```
euler: [15.234 ms 15.456 ms 15.678 ms]
rk4:   [61.123 ms 61.456 ms 61.789 ms]

Ratio: 61.456 / 15.456 = 3.977 ≈ 4.0× ✅
```

If ratio ≠ 4.0:
- > 5×: Extra allocations in RK4 intermediate steps
- < 3×: Compiler optimizing something away (check black_box)

## 4. Common Benchmark Pitfalls

### Pitfall 1: Forgetting black_box

```rust
// ❌ BAD: Compiler may optimize away
b.iter(|| {
    solver.solve(&scenario, &config).unwrap()
});

// ✅ GOOD: Forces actual execution
b.iter(|| {
    solver.solve(black_box(&scenario), black_box(&config)).unwrap()
});
```

### Pitfall 2: Setup inside b.iter()

```rust
// ❌ BAD: Measures setup + solve
b.iter(|| {
    let scenario = create_scenario(100);  // SETUP (slow!)
    solver.solve(black_box(&scenario), black_box(&config)).unwrap()
});

// ✅ GOOD: Setup once, measure solve only
let scenario = create_scenario(100);  // Outside b.iter
b.iter(|| {
    solver.solve(black_box(&scenario), black_box(&config)).unwrap()
});
```

### Pitfall 3: Not Using --release

```bash
# ❌ BAD: Debug mode (10-100× slower)
cargo bench

# ✅ GOOD: Release mode (automatically used by cargo bench)
# But if running tests manually:
cargo test --release
```

## 5. Profiling Slow Benchmarks

If a benchmark is unexpectedly slow:

```bash
# Install flamegraph
cargo install flamegraph

# Profile the benchmark
cargo flamegraph --bench solver_performance

# Opens flamegraph.svg in browser
# Shows where time is spent (CPU profiling)
```

Look for:
- Unexpected allocations (malloc, free calls)
- Cache misses (high LLC-load-misses)
- Branch mispredictions

## 6. Comparing Across Machines

Save baseline on Machine A:
```bash
cargo bench --bench solver_performance -- --save-baseline machine_a
```

Run on Machine B:
```bash
cargo bench --bench solver_performance -- --baseline machine_a
```

Criterion will show relative performance:
```
Machine A: 15.456 ms
Machine B: 12.345 ms
Change: -20.13% (faster!)
```

## 7. Statistical Significance

Criterion performs statistical analysis:

```
Change: +2.34% (p = 0.03 < 0.05)
        ^^^^^^   ^^^^^^^^^^^^^^^^
        Change   Statistically significant

Change: +1.23% (p = 0.23 > 0.05)
        ^^^^^^   ^^^^^^^^^^^^^^^^
        Change   NOT significant (noise)
```

## 8. When to Re-run Benchmarks

Re-benchmark after:
- ✅ Algorithm changes
- ✅ Adding/removing allocations
- ✅ Changing data structures
- ✅ Upgrading compiler (major version)
- ✅ Enabling/disabling SIMD

Don't need to re-benchmark for:
- ❌ Comment changes
- ❌ Test changes
- ❌ Documentation changes

## 9. Optimization Workflow

```
1. cargo bench -- --save-baseline before
2. Make optimization
3. cargo bench -- --baseline before
4. If faster → keep
   If slower → investigate or revert
5. Repeat
```

## 10. Expected Performance Targets

For reference, approximate times on modern hardware:

```
SimpleModel (exponential decay):
  Euler, 100 points, 1000 steps:  ~15 ms
  RK4, 100 points, 1000 steps:    ~60 ms

If your times are:
  < 50% of these: ✅ Fast machine or good optimizations
  ~ 100%:         ✅ Expected
  > 200%:         ⚠️ Check --release, CPU throttling, background load
  > 500%:         🔴 Something wrong (debug mode? VM?)
```

# Further Reading

- Criterion documentation: https://bheisler.github.io/criterion.rs/
- Rust Performance Book: https://nnethercote.github.io/perf-book/
- std::hint::black_box docs: https://doc.rust-lang.org/std/hint/fn.black_box.html

# Bonus: Python Script to Analyze Results

Here's a complete script to analyze and visualize benchmark results:

```python
#!/usr/bin/env python3
"""
Analyze solver_performance benchmark results

Usage:
    1. Run benchmarks: cargo bench --bench solver_performance
    2. Save this script as analyze_benchmarks.py
    3. Run: python analyze_benchmarks.py
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_criterion_results():
    """Parse Criterion benchmark results from JSON files"""
    
    base_path = Path("target/criterion/solver_comparison")
    results = {}
    
    for benchmark_dir in base_path.glob("*"):
        if not benchmark_dir.is_dir():
            continue
            
        # Parse benchmark name: "euler_50pts_100steps"
        name = benchmark_dir.name
        json_file = benchmark_dir / "base" / "estimates.json"
        
        if json_file.exists():
            with open(json_file) as f:
                data = json.load(f)
                # Extract mean time in nanoseconds
                mean_ns = data["mean"]["point_estimate"]
                results[name] = mean_ns / 1e6  # Convert to milliseconds
    
    return results

def extract_config(name):
    """Extract (solver, points, time_steps) from benchmark name"""
    # "euler_50pts_100steps" → ("euler", 50, 100)
    parts = name.split("_")
    solver = parts[0]
    points = int(parts[1].replace("pts", ""))
    time_steps = int(parts[2].replace("steps", ""))
    return solver, points, time_steps

def plot_solver_comparison():
    """Create comparison plots"""
    
    results = parse_criterion_results()
    
    # Organize data
    euler_data = []
    rk4_data = []
    
    for name, time_ms in results.items():
        solver, points, time_steps = extract_config(name)
        ops = points * time_steps
        
        if solver == "euler":
            euler_data.append((ops, time_ms, points, time_steps))
        else:
            rk4_data.append((ops, time_ms, points, time_steps))
    
    # Sort by operations
    euler_data.sort()
    rk4_data.sort()
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Time vs Operations (log-log)
    euler_ops, euler_times, _, _ = zip(*euler_data)
    rk4_ops, rk4_times, _, _ = zip(*rk4_data)
    
    ax1.loglog(euler_ops, euler_times, 'o-', label='Euler', linewidth=2, markersize=8)
    ax1.loglog(rk4_ops, rk4_times, 's-', label='RK4', linewidth=2, markersize=8)
    ax1.set_xlabel('Total Operations (points × time_steps)', fontsize=12)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('Scaling with Problem Size', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, which='both', alpha=0.3)
    
    # Plot 2: RK4/Euler Ratio
    ratios = [rk4_times[i] / euler_times[i] for i in range(len(euler_times))]
    labels = [f"{p}×{t}" for _, _, p, t in euler_data]
    
    ax2.bar(range(len(ratios)), ratios, color=['green' if 3.8 < r < 4.2 else 'orange' for r in ratios])
    ax2.axhline(y=4.0, color='red', linestyle='--', linewidth=2, label='Expected (4.0)')
    ax2.set_xticks(range(len(ratios)))
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_ylabel('RK4 / Euler Ratio', fontsize=12)
    ax2.set_title('Solver Overhead Ratio', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_ylim([3.5, 4.5])
    
    # Plot 3: Cost per Operation
    euler_cost = [t / ops * 1e6 for ops, t, _, _ in euler_data]  # ns/op
    rk4_cost = [t / ops * 1e6 for ops, t, _, _ in rk4_data]
    
    x = range(len(euler_cost))
    width = 0.35
    ax3.bar([i - width/2 for i in x], euler_cost, width, label='Euler', alpha=0.8)
    ax3.bar([i + width/2 for i in x], rk4_cost, width, label='RK4', alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=45, ha='right')
    ax3.set_ylabel('Cost per Operation (ns)', fontsize=12)
    ax3.set_title('Efficiency (lower is better)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('solver_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Plot saved as solver_analysis.png")
    
    # Print summary table
    print("\n📈 PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"{'Config':<15} {'Euler':>12} {'RK4':>12} {'Ratio':>10} {'Status':>8}")
    print("-" * 70)
    
    for i in range(len(euler_data)):
        ops, e_time, pts, steps = euler_data[i]
        _, r_time, _, _ = rk4_data[i]
        ratio = r_time / e_time
        status = "✅" if 3.8 < ratio < 4.2 else "⚠️"
        
        print(f"{pts}×{steps:<10} {e_time:>10.2f} ms {r_time:>10.2f} ms {ratio:>9.2f} {status:>8}")
    
    print("=" * 70)
    
    # Recommendations
    avg_ratio = np.mean(ratios)
    print(f"\nAverage RK4/Euler ratio: {avg_ratio:.2f}")
    
    if 3.9 < avg_ratio < 4.1:
        print("✅ Excellent! RK4 overhead is exactly 4× as expected.")
    elif avg_ratio > 4.5:
        print("⚠️  RK4 has extra overhead beyond 4 function evaluations.")
        print("   → Check for allocations or cache misses in compute_physics")
    elif avg_ratio < 3.5:
        print("🤔 RK4 is faster than expected relative to Euler.")
        print("   → Verify black_box is preventing over-optimization")

if __name__ == "__main__":
    try:
        plot_solver_comparison()
    except FileNotFoundError:
        print("❌ Benchmark results not found!")
        print("   Run: cargo bench --bench solver_performance")
        print("   Then run this script again.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
```

Save this as `scripts/analyze_benchmarks.py` and run after benchmarking!
*/