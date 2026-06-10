//! Parser throughput baseline. Run with `cargo bench -p oxyz-core`.
//!
//! Input is generated in memory from a fixed seed: identical bytes every run
//! (comparable numbers), but varied within the file so the parser doesn't
//! benchmark against one frame repeated. Parsing from memory keeps disk I/O
//! out of the measurement.

use std::{fmt::Write as _, hint::black_box, io::Cursor};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use oxyz_core::{FrameIter, read_frames_parallel, scan_frames};

const SPECIES: &[&str] = &["H", "C", "N", "O", "Si"];

/// Deterministic xorshift64 PRNG; enough for data generation, no `rand`
/// dependency needed.
struct XorShift64(u64);

impl XorShift64 {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in [0, 1).
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next_u64() as usize) % (hi - lo)
    }
}

fn write_frame(text: &mut String, rng: &mut XorShift64, n_atoms: usize) {
    let a = 5.0 + 10.0 * rng.uniform();
    let energy = -10.0 * rng.uniform();

    writeln!(text, "{n_atoms}").unwrap();
    writeln!(
        text,
        "Lattice=\"{a:.6} 0.0 0.0 0.0 {a:.6} 0.0 0.0 0.0 {a:.6}\" \
         Properties=species:S:1:pos:R:3:forces:R:3 energy={energy:.6} pbc=\"T T T\"",
    )
    .unwrap();

    for _ in 0..n_atoms {
        let species = SPECIES[rng.range(0, SPECIES.len())];
        writeln!(
            text,
            "{species} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
            a * rng.uniform(),
            a * rng.uniform(),
            a * rng.uniform(),
            rng.uniform() - 0.5,
            rng.uniform() - 0.5,
            rng.uniform() - 0.5,
        )
        .unwrap();
    }
}

fn trajectory(n_frames: usize, atoms_lo: usize, atoms_hi: usize, seed: u64) -> String {
    let mut rng = XorShift64(seed);
    let mut text = String::new();

    for _ in 0..n_frames {
        let n_atoms = rng.range(atoms_lo, atoms_hi);
        write_frame(&mut text, &mut rng, n_atoms);
    }

    text
}

fn parse_all(text: &str) -> usize {
    FrameIter::new(Cursor::new(text.as_bytes()))
        .map(|frame| frame.unwrap().n_atoms)
        .sum()
}

fn scan_all(text: &str) -> usize {
    scan_frames(Cursor::new(text.as_bytes()))
        .unwrap()
        .total_atoms()
}

fn bench_parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse");

    // Many small frames: stresses per-frame overhead (header parsing,
    // metadata typing, column setup).
    let small = trajectory(2_000, 16, 64, 0x5EED);
    group.throughput(Throughput::Bytes(small.len() as u64));
    group.bench_function("many_small_frames", |b| {
        b.iter(|| black_box(parse_all(&small)))
    });

    // Few large frames: stresses the per-atom hot loop.
    let large = trajectory(4, 100_000, 100_001, 0x5EED2);
    group.throughput(Throughput::Bytes(large.len() as u64));
    group.bench_function("large_frames", |b| b.iter(|| black_box(parse_all(&large))));

    group.finish();
}

/// The structural scan must stay far above parse throughput to earn its
/// keep; this is the measurement gate for reaching for `memchr`.
fn bench_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("scan");

    let small = trajectory(2_000, 16, 64, 0x5EED);
    group.throughput(Throughput::Bytes(small.len() as u64));
    group.bench_function("many_small_frames", |b| {
        b.iter(|| black_box(scan_all(&small)))
    });

    let large = trajectory(4, 100_000, 100_001, 0x5EED2);
    group.throughput(Throughput::Bytes(large.len() as u64));
    group.bench_function("large_frames", |b| b.iter(|| black_box(scan_all(&large))));

    group.finish();
}

/// Thread-count scaling of the parallel bulk read (includes the scan).
/// This is the measurement gate for mmap and scan/parse pipelining: it
/// shows where per-worker buffered reads or the serial scan become the
/// bottleneck.
fn bench_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_read");

    let cases = [
        ("many_small_frames", trajectory(2_000, 16, 64, 0x5EED)),
        ("large_frames", trajectory(4, 100_000, 100_001, 0x5EED2)),
    ];

    for (name, text) in cases {
        let path = std::env::temp_dir().join(format!("oxyz_bench_{name}.extxyz"));
        std::fs::write(&path, &text).unwrap();
        group.throughput(Throughput::Bytes(text.len() as u64));

        for threads in [1, 2, 4, 8] {
            group.bench_function(BenchmarkId::new(name, threads), |b| {
                b.iter(|| black_box(read_frames_parallel(&path, Some(threads)).unwrap().len()))
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_parse, bench_scan, bench_parallel);
criterion_main!(benches);
