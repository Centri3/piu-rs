#![allow(arithmetic_overflow)]

use std::hash::{DefaultHasher, Hasher};

use mersenne_twister::MT19937;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const M_MIN: f64 = 0.0725487;
const M_MAX: f64 = 1000.0;
const M_CUT: f64 = 50.0;

const CHUNK_SIZE: i64 = 40;
const STARS_PER_CHUNK: usize = 10;
const GLOBAL_SEED: i64 = 12345;

#[allow(non_upper_case_globals)]
const _PyHASH_MODULUS: u64 = (1 << _PyHASH_BITS) - 1;
#[allow(non_upper_case_globals)]
const _PyHASH_BITS: i32 = 61;
#[allow(non_upper_case_globals)]
const _PyHASH_XX_PRIME_1: u64 = 11400714785074694791;
#[allow(non_upper_case_globals)]
const _PyHASH_XX_PRIME_2: u64 = 14029467366897019727;
#[allow(non_upper_case_globals)]
const _PyHASH_XX_PRIME_5: u64 = 2870177450012600261;

/// reimplementation of python's hash() *FOR TUPLES!*
///
/// only use for `(chunk_x, chunk_y, GLOBAL_SEED)`! (UNLESS WE NEED IT FOR OTHER STUFF OH NO)
///
/// https://blazejcyrzon.com/tuple_hash/
fn wtf_hash(global_seed: i64, chunk_x: i64, chunk_y: i64) -> i64 {
    let (global_seed, chunk_x, chunk_y) = (global_seed as u64, chunk_x as u64, chunk_y as u64);

    let mut acc = _PyHASH_XX_PRIME_5;

    // element 1
    acc += global_seed * _PyHASH_XX_PRIME_2;
    acc = acc.rotate_left(31);
    acc *= _PyHASH_XX_PRIME_1;

    // element 2
    acc += chunk_x * _PyHASH_XX_PRIME_2;
    acc = acc.rotate_left(31);
    acc *= _PyHASH_XX_PRIME_1;

    // element 3
    acc += chunk_y * _PyHASH_XX_PRIME_2;
    acc = acc.rotate_left(31);
    acc *= _PyHASH_XX_PRIME_1;

    acc += 3 ^ (_PyHASH_XX_PRIME_5 ^ 3527539u64);

    if acc == -1i64 as u64 {
        return 1546275796;
    }

    acc as i64
}

/// ugh
fn wtf_hashdouble(v: f64) -> i64 {
    let mut e: i32;
    let mut sign: i32;
    let mut m: f64;
    let mut x: u64;
    let mut y: u64;

    assert!(v.is_finite(), "fuck that");

    // if (!isfinite(v)) {
    //     if (isinf(v))
    //         return v > 0 ? _PyHASH_INF : -_PyHASH_INF;
    //     else
    //         return PyObject_GenericHash(inst);
    // }

    (m, e) = libm::frexp(v);

    sign = 1;
    if m < 0.0 {
        sign = -1;
        m = -m;
    }

    /* process 28 bits at a time;  this should work well both for binary
    and hexadecimal floating point. */
    x = 0;

    while m != 0.0 {
        x = ((x << 28) & _PyHASH_MODULUS) | x >> (_PyHASH_BITS - 28);
        m *= 268435456.0; /* 2**28 */
        e -= 28;
        y = m as u64; /* pull out integer part */
        m -= y as f64;
        x += y;
        if x >= _PyHASH_MODULUS {
            x -= _PyHASH_MODULUS;
        }
    }

    /* adjust for the exponent;  first reduce it modulo _PyHASH_BITS */
    e = if e >= 0 {
        e % _PyHASH_BITS
    } else {
        _PyHASH_BITS - 1 - ((-1 - e) % _PyHASH_BITS)
    };
    x = ((x << e) & _PyHASH_MODULUS) | x >> (_PyHASH_BITS - e);

    x = x * sign as u64;
    if x == -1i64 as u64 {
        x = -2i64 as u64;
    }

    x as i64
}

struct WtfRng(MT19937);

impl WtfRng {
    /// idc about using the right trait rn.
    fn wtf_float(&mut self) -> f64 {
        let a = self.0.gen::<u32>() >> 5;
        let b = self.0.gen::<u32>() >> 6;

        (a as f64 * 67108864.0 + b as f64) * (1.0 / 9007199254740992.0)
    }

    fn uniform(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.wtf_float()
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Star {
    mass: f64,
    age: f64,
    fe_h: f64,
    radius: f64,
    temperature: f64,
    color: f64,
    spectral: f64,
    axial_tilt: f64,
    rotation_period: f64,
    density: f64,
    oblateness: f64,
    class: &'static str,
    position: (f64, f64),
}

fn generate_mass(rng: &mut WtfRng) -> f64 {
    let alpha = 2.35;
    let mut u = rng.wtf_float();
    let exponent = 1.0 - alpha;
    loop {
        let m_power = M_MIN
            * (((M_MAX / M_MIN).powf(exponent) - u * ((M_MAX / M_MIN).powf(exponent) - 1.0))
                .powf(1.0 / exponent));
        if rng.wtf_float() < f64::exp(-m_power / M_CUT) {
            return m_power;
        }
        u = rng.wtf_float();
    }
}

fn compute_radius(mass: f64) -> f64 {
    if mass < 1.0 {
        mass.powf(0.8)
    } else {
        mass.powf(0.65)
    }
}

fn generate_additional_star_properties(
    rng: &mut WtfRng,
    mass: f64,
    radius: f64,
) -> (f64, f64, f64, f64) {
    let (mut axial_tilt, mut rotation_period, mut density, mut f) = (0.0, 0.0, 0.0, 0.0);
    loop {
        axial_tilt = rng.uniform(0.0, 180.0);
        if mass < 1.0 {
            rotation_period = rng.uniform(10.0, 100.0);
        } else if mass < 3.0 {
            rotation_period = rng.uniform(5.0, 30.0);
        } else {
            rotation_period = rng.uniform(0.5, 10.0);
        }
        density = if radius != 0.0 {
            mass / (radius.powi(3))
        } else {
            0.0
        };
        let rotation_period_s = rotation_period * 86400.0;
        let omega = 2.0 * std::f64::consts::PI / rotation_period_s;
        let r_si = radius * 6.957e8;
        let m_si = mass * 1.989e30;
        f = if m_si > 0.0 {
            (3.0 / 2.0) * (omega.powi(2) * (r_si.powi(3))) / (6.67430e-11 * m_si)
        } else {
            0.0
        };
        if f < 0.69420 {
            break;
        }
    }

    (axial_tilt, rotation_period, density, f)
}

fn generate_star(star_seed: i64, pos_x: f64, pos_y: f64) -> Star {
    let mut rng = WtfRng(SeedableRng::from_seed(star_seed as u64));

    let base_mass = generate_mass(&mut rng);
    let delta_mass = rng.uniform(-0.01, 0.01);
    let mass = base_mass * (1.0 + delta_mass);

    let base_radius = compute_radius(mass);
    let delta_radius = rng.uniform(-0.01, 0.01);
    let radius = base_radius * (1.0 + delta_radius);

    // age
    rng.wtf_float();
    // metallicity
    rng.wtf_float();

    // age = generate_age(mass, rng)
    // fe_h = generate_metallicity(age, rng)
    // temperature = compute_temperature(mass, radius)
    // luminosity = compute_luminosity(radius, temperature)
    // color = temperature_to_rgb(temperature)
    // spectral = assign_spectral_class(temperature)
    let (axial_tilt, rotation_period, density, oblateness) =
        generate_additional_star_properties(&mut rng, mass, radius);
    // star_class = assign_star_class(spectral)

    Star {
        mass,
        radius,
        axial_tilt,
        rotation_period,
        density,
        oblateness,
        position: (pos_x, pos_y),
        ..Default::default()
    }
}

fn generate_chunk(chunk_x: i64, chunk_y: i64) -> [Star; STARS_PER_CHUNK as usize] {
    let mut chunk = [Star::default(); 10];

    let chunk_seed = wtf_hash(GLOBAL_SEED, chunk_x, chunk_y);
    let mut rng = WtfRng(SeedableRng::from_seed(chunk_seed as u64));

    for i in 0..STARS_PER_CHUNK {
        let pos_x = (chunk_x * CHUNK_SIZE) as f64 + rng.uniform(0.0f64, CHUNK_SIZE as f64);
        let pos_y = (chunk_y * CHUNK_SIZE) as f64 + rng.uniform(0.0f64, CHUNK_SIZE as f64);

        // why. why why why why does this sometimes not equal the actual hash wtf python. the inputs are all identical. wtf wtf wtf wtf -a
        // you should really run this on every chunk for both and compare them vs just manually -b
        let star_seed = wtf_hash(chunk_seed, wtf_hashdouble(pos_x), wtf_hashdouble(pos_y));
        chunk[i] = generate_star(star_seed, pos_x, pos_y);
    }

    chunk
}

fn main() {
    (-99995..100000).into_par_iter().for_each(|i| {
        (-100000..100000).for_each(|j| {
            for star in generate_chunk(i, j) {
                if star.mass > 300.0 {
                    println!(
                        "found star of mass {}, x = {}, y = {}",
                        star.mass, star.position.0 as i128, star.position.1 as i128,
                    );
                }
            }
        })
    });
}
