mod frozen {
    use crate::electron_catalog::{
        EXPLICIT_ELECTRON_PROCESSES as ROWS, ElectronChannel as Ch, ElectronMassMeV as Mass,
        ElectronX as X, ExplicitElectronProcess as Proc, FourMomentumMeV as Four, NeutrinoY as Y,
        TemperatureMeV as Temp,
    };
    use crate::electron_hm::author_hm_w_dimensionless as hm;
    use crate::electron_phase_point::{
        PhysicalPointDensity as Point, PhysicalRadialCell as Cell, PhysicalSupportSlice as Slice,
        checked_nonnegative_hm as check_hm, physical_point_density as make_point,
        physical_support_slice as make_support,
    };
    use core::f64::consts::PI;
    const ME: f64 = 0.510_998_95;
    type Seed = [f64; 8];
    const S: Seed = [1.0, 1.0, ME, 2.0, 0.4, 0.23, 1.0, 1.0];
    const SPLIT: Seed = [1.2, 1.0, ME, 2.0, 0.4, 0.23, 1.0, 1.0];
    const COLD: Seed = [1.0, 1.0, 0.0, 2.0, 0.4, 0.23, 1.0, 1.0];
    const HOT: Seed = [2.0, 2.0, 0.0, 2.0, 0.4, 0.23, 1.0, 1.0];
    const WEIGHTED: Seed = [1.0, 1.0, ME, 2.0, 0.4, 0.23, 0.7, 1.3];
    const UNDER: Seed = [1.0, 1.0, 0.0, 1.0, 0.4, 0.23, 1.0, 1.0];
    const NF: [f64; 3] = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
    struct Oracle {
        support: [f64; 17],
        raw: [[f64; 4]; 4],
        tail: [f64; 7],
    }
    const T: fn(f64) -> Temp = |v| Temp::new(v).unwrap();
    const M: fn(f64) -> Mass = |v| Mass::new(v).unwrap();
    const YV: fn(f64) -> Y = |v| Y::new(v).unwrap();
    const XV: fn(f64) -> X = |v| X::new(v).unwrap();
    const FD: fn(f64) -> f64 = |v| 1.0 / (v.exp() + 1.0);
    const P2: fn(f64) -> f64 = |v| 0.5 * (3.0 * v * v - 1.0);
    const REL: fn(f64, f64) -> f64 = |a, b| (a - b).abs() / a.abs().max(b.abs()).max(1.0e-300);
    const NEAR: fn(f64, f64) = |a, b| assert!(REL(a, b) <= 1.0e-12);
    const FAR: fn(f64, f64, f64) = |a, b, floor| assert!(REL(a, b) > floor);
    const BITS: fn(f64, f64) = |a, b| assert!(a.to_bits() == b.to_bits());
    fn arrays<const N: usize>(a: [f64; N], b: [f64; N]) {
        for (left, right) in a.into_iter().zip(b) {
            NEAR(left, right);
        }
    }
    fn part<const N: usize>(values: &[f64], first: usize) -> [f64; N] {
        values[first..first + N].try_into().unwrap()
    }
    fn err<T>(value: Result<T, &'static str>) {
        assert!(value.is_err());
    }
    fn changed(index: usize, value: f64) -> Seed {
        let mut seed = S;
        seed[index] = value;
        seed
    }
    fn el(x2: f64, w2: f64, y3: f64, w3: f64) -> Result<Cell, &'static str> {
        Cell::elastic(XV(x2), w2, YV(y3), w3)
    }
    fn pa(y2: f64, w2: f64, x3: f64, w3: f64) -> Result<Cell, &'static str> {
        Cell::pair(YV(y2), w2, XV(x3), w3)
    }
    fn radial(row: Proc, s: Seed) -> Cell {
        match row.channel() {
            Ch::Pair => pa(1.8, s[6], 1.0, s[7]).unwrap(),
            _ => el(1.3, s[6], 1.1, s[7]).unwrap(),
        }
    }
    fn build(slot: usize, s: Seed, cell: Cell) -> Result<Option<Slice>, &'static str> {
        make_support(slot, T(s[0]), T(s[1]), M(s[2]), YV(s[3]), cell, s[4])
    }
    fn absent(slot: usize, cell: Cell) {
        assert!(build(slot, S, cell).unwrap().is_none());
    }
    fn production(slot: usize, s: Seed) -> (Slice, Point) {
        let slice = build(slot, s, radial(ROWS[slot], s)).unwrap().unwrap();
        let point = make_point(&slice, s[5]).unwrap();
        (slice, point)
    }
    fn four(raw: [[f64; 4]; 4]) -> [Four; 4] {
        raw.map(|v| Four::new(v[0], v[1], v[2], v[3]).unwrap())
    }
    fn w(row: Proc, momenta: [Four; 4], mass: f64) -> f64 {
        hm(row, momenta, M(mass))
    }
    fn oracle(row: Proc, s: Seed) -> Oracle {
        let [tg, tc, mass, y1, mu, u, w2, w3] = s;
        let p1 = y1 * tc;
        let (p2, p3, e2, e3, dp) = match row.channel() {
            Ch::Pair => (1.8 * tc, tg, 1.8 * tc, tg.hypot(mass), [tc * w2, tg * w3]),
            _ => (
                1.3 * tg,
                1.1 * tc,
                (1.3 * tg).hypot(mass),
                1.1 * tc,
                [tg * w2, tc * w3],
            ),
        };
        let e4 = p1 + e2 - e3;
        let p4 = ((e4 - mass) * (e4 + mass)).sqrt();
        let k = p1 * p1 + p2 * p2 + p3 * p3 - 2.0 * p1 * p3 * mu - p4 * p4;
        let q = 2.0 * p2 * (p1 - p3 * mu);
        let base = 4.0 * p2 * p2 * p3 * p3 * (1.0 - mu * mu);
        let poly = [-base - q * q, -2.0 * k * q, base - k * k];
        let [a, b, c] = poly;
        let roots = [-b / (2.0 * a), (b * b - 4.0 * a * c).sqrt() / (-2.0 * a)];
        let theta = (PI / 2.0) * (1.0 + u);
        let mu12 = roots[0] + roots[1] * theta.cos();
        let d = a * mu12 * mu12 + b * mu12 + c;
        let (s12, s13) = ((1.0 - mu12 * mu12).sqrt(), (1.0 - mu * mu).sqrt());
        let beta = ((k + q * mu12) / (2.0 * p2 * p3 * s12 * s13)).acos();
        let raw = [
            [p1, 0.0, 0.0, p1],
            [e2, p2 * s12, 0.0, p2 * mu12],
            [e3, p3 * s13 * beta.cos(), p3 * s13 * beta.sin(), p3 * mu],
            [
                e4,
                p2 * s12 - p3 * s13 * beta.cos(),
                -p3 * s13 * beta.sin(),
                p1 + p2 * mu12 - p3 * mu,
            ],
        ];
        let phase = 2.0 / (2.0 * PI).powi(4) / (2.0 * p1) * p2.powi(2) * dp[0] / (2.0 * e2)
            * p3.powi(2)
            * dp[1]
            / (2.0 * e3)
            * (roots[1] * (PI / 2.0) * theta.sin()).abs()
            / (2.0 * p2 * p3 * s12 * s13 * beta.sin().abs());
        let momenta = four(raw);
        let hm = w(row, momenta, mass);
        let ff = match row.channel() {
            Ch::Pair => [FD(e3 / tg), FD(e4 / tg)],
            _ => [FD(e2 / tg), FD(e4 / tg)],
        };
        Oracle {
            support: [
                p1, e2, e3, e4, p1, p2, p3, p4, dp[0], dp[1], ff[0], ff[1], poly[0], poly[1],
                poly[2], roots[0], roots[1],
            ],
            raw,
            tail: [theta, mu12, beta, d, raw[3][3] / p4, hm, phase * hm],
        }
    }
    fn compare_physical(row: Proc, slice: Slice, point: Point, o: Oracle) {
        assert_eq!(slice.process, row);
        assert_eq!(slice.electron_mass, M(ME));
        assert_eq!(slice.mu13, S[4]);
        arrays(slice.energies_mev, part(&o.support, 0));
        arrays(slice.momentum_magnitudes_mev, part(&o.support, 4));
        arrays(slice.radial_differentials_mev, part(&o.support, 8));
        arrays(slice.fixed_fermions, part(&o.support, 10));
        arrays(slice.support_polynomial_mev4, part(&o.support, 12));
        arrays(slice.support_center_radius, part(&o.support, 15));
        assert_eq!(point.four_momenta, four(o.raw));
        NEAR(point.theta, o.tail[0]);
        NEAR(point.mu12, o.tail[1]);
        NEAR(point.mu13, S[4]);
        NEAR(point.beta, o.tail[2]);
        NEAR(point.support_d_mev4, o.tail[3]);
        arrays(point.mu_1i, [1.0, o.tail[1], S[4], o.tail[4]]);
        let masses = match row.channel() {
            Ch::Pair => [0.0, 0.0, ME, ME],
            _ => [0.0, ME, 0.0, ME],
        };
        for i in 0..4 {
            let leg = point.raw_momenta_mev[i];
            arrays(leg, o.raw[i]);
            let spatial = leg[1] * leg[1] + leg[2] * leg[2] + leg[3] * leg[3];
            NEAR(spatial.sqrt(), o.support[4 + i]);
            assert!(
                (leg[0] * leg[0] - spatial - masses[i] * masses[i]).abs()
                    <= 1.0e-12 * leg[0] * leg[0]
            );
            let legs = point.raw_momenta_mev.map(|v| v[i]);
            let scale: f64 = legs.map(f64::abs).into_iter().sum();
            assert!((legs[0] + legs[1] - legs[2] - legs[3]).abs() <= 1.0e-12 * scale);
        }
    }
    fn occupancy(row: Proc, fixed: [f64; 2]) -> [f64; 4] {
        match row.channel() {
            Ch::Pair => [FD(2.0), FD(1.8), fixed[0], fixed[1]],
            _ => [FD(2.0), fixed[0], FD(1.1), fixed[1]],
        }
    }
    fn balance([f1, f2, f3, f4]: [f64; 4]) -> [f64; 3] {
        let gain = (1.0 - f1) * (1.0 - f2) * f3 * f4;
        let loss = f1 * f2 * (1.0 - f3) * (1.0 - f4);
        let residual = (gain - loss).abs() / gain.max(loss).max(1.0e-300);
        [gain, loss, residual]
    }
    fn rates(slot: usize, s: Seed) -> [f64; 2] {
        let p = production(slot, s).1;
        [p.scalar_density_mev.value(), p.p2_densities_mev[1].value()]
    }
    #[test]
    fn support_and_body_point_match_direct_shell_oracle() {
        for (slot, row) in ROWS.into_iter().enumerate() {
            let (slice, point) = production(slot, S);
            compare_physical(row, slice, point, oracle(row, S));
        }
    }
    #[test]
    fn measure_and_p2_densities_match_direct_delta_oracle() {
        for (slot, row) in ROWS.into_iter().enumerate() {
            let actual = production(slot, S).1;
            let expected = oracle(row, S);
            let scalar = actual.scalar_density_mev.value();
            NEAR(actual.hm_dimensionless, expected.tail[5]);
            NEAR(scalar, expected.tail[6]);
            assert!(scalar.is_finite() && scalar > 0.0);
            BITS(actual.p2_densities_mev[0].value(), scalar);
            let (coupled, wrong) = match row.channel() {
                Ch::Pair => (expected.tail[1], S[4]),
                _ => (S[4], expected.tail[1]),
            };
            let value = actual.p2_densities_mev[1].value();
            NEAR(value, expected.tail[6] * P2(coupled));
            assert!(value.is_finite() && value != 0.0);
            FAR(value, expected.tail[6] * P2(wrong), 1.0e-4);
        }
        let seed = oracle(ROWS[0], S);
        let actual = rates(0, S)[0];
        let legacy = (1.0 / (4.0 * PI.powi(3))) / (2.0 / (2.0 * PI).powi(4));
        for factor in [0.5, 2.0, legacy, 1.0 / seed.tail[3].sqrt()] {
            FAR(actual, seed.tail[6] * factor, 0.25);
        }
        for slot in 0..18 {
            let (low, high) = (rates(slot, COLD), rates(slot, HOT));
            high.into_iter().zip(low).for_each(|(hi, lo)| {
                let ratio = hi / lo;
                NEAR(ratio, 32.0);
                FAR(ratio, 16.0, 0.1);
                FAR(ratio, 64.0, 0.1);
            });
        }
        for slot in [0, 2] {
            let expected = oracle(ROWS[slot], WEIGHTED);
            let (slice, point) = production(slot, WEIGHTED);
            arrays(slice.radial_differentials_mev, part(&expected.support, 8));
            NEAR(point.scalar_density_mev.value(), expected.tail[6]);
        }
    }
    #[test]
    fn pair_order_and_fixed_occupations_are_discriminating() {
        let (mut split_elastic, mut split_pair) = (false, false);
        for (slot, row) in ROWS.into_iter().enumerate() {
            let equal = production(slot, S).0;
            let split = production(slot, SPLIT).0;
            let split_expected = oracle(row, SPLIT);
            arrays(equal.fixed_fermions, part(&oracle(row, S).support, 10));
            arrays(split.fixed_fermions, part(&split_expected.support, 10));
            arrays(
                split.radial_differentials_mev,
                part(&split_expected.support, 8),
            );
            let db = balance(occupancy(row, equal.fixed_fermions));
            assert!(db[0] > 0.0 && db[1] > 0.0 && db[2] <= 1.0e-12);
            let residual = balance(occupancy(row, split.fixed_fermions))[2];
            match row.channel() {
                Ch::Pair => split_pair |= residual > 1.0e-8,
                _ => split_elastic |= residual > 1.0e-8,
            }
        }
        assert!(split_elastic && split_pair);
        for [nu, anti, wrong] in [[2, 5, 8], [8, 11, 2], [14, 17, 2]] {
            let (p, a) = (production(nu, S).1, production(anti, S).1);
            assert_eq!(p.raw_momenta_mev, a.raw_momenta_mev);
            assert_eq!(ROWS[anti].input(), ROWS[nu].target());
            let independent = four(oracle(ROWS[nu], S).raw);
            let mut swapped = independent;
            swapped.swap(2, 3);
            let expected = w(ROWS[nu], independent, ME);
            NEAR(expected, w(ROWS[anti], swapped, ME));
            NEAR(p.hm_dimensionless, expected);
            NEAR(a.hm_dimensionless, w(ROWS[anti], independent, ME));
            FAR(expected, w(ROWS[anti], independent, ME), 1.0e-4);
            FAR(expected, w(ROWS[wrong], independent, ME), 1.0e-4);
        }
        for bad in [
            build(18, S, radial(ROWS[0], S)),
            build(0, S, radial(ROWS[2], S)),
            build(2, S, radial(ROWS[0], S)),
            build(0, changed(3, 0.0), radial(ROWS[0], S)),
        ] {
            err(bad);
        }
        absent(0, el(1.3, 1.0, 2.8, 1.0).unwrap());
        absent(2, pa(1.8, 1.0, 4.0, 1.0).unwrap());
        for bad in [-1.0, 1.0].into_iter().chain(NF) {
            err(build(0, changed(4, bad), radial(ROWS[0], S)));
            err(make_point(&production(0, S).0, bad));
        }
        for bad in [
            build(2, UNDER, pa(1.0e-200, 1.0, 1.0e-200, 1.0).unwrap()),
            build(0, changed(0, f64::MAX), el(2.0, 1.0, 1.0, 1.0).unwrap()),
            build(2, changed(1, f64::MAX), pa(2.0, 1.0, 1.0, 1.0).unwrap()),
        ] {
            err(bad);
        }
        for bad in [0.0, -1.0].into_iter().chain(NF) {
            err(el(1.0, bad, 1.0, 1.0));
            err(el(1.0, 1.0, 1.0, bad));
            err(pa(1.0, bad, 1.0, 1.0));
            err(pa(1.0, 1.0, 1.0, bad));
        }
        err(el(0.0, 1.0, 1.0, 1.0));
        err(el(1.0, 1.0, 0.0, 1.0));
        err(pa(0.0, 1.0, 1.0, 1.0));
        err(pa(1.0, 1.0, 0.0, 1.0));
        for good in [-0.0, 0.0, 1.25, f64::MAX] {
            BITS(check_hm(good).unwrap(), good);
        }
        for bad in [-f64::MIN_POSITIVE].into_iter().chain(NF) {
            err(check_hm(bad));
        }
    }
}
