mod frozen {
    use crate::electron_catalog::*;
    use crate::electron_catalog::{ElectronMassMeV as Mass, TemperatureMeV as Temp};
    use crate::electron_hm::{
        author_hm_w_dimensionless as physical_w, author_hm_w_over_gf2_mev4 as reduced_w,
        dimensionless_hm_event_factor_mev as event_factor,
    };
    use ElectronChannel::{ElectronMinusElastic as Em, ElectronPlusElastic as Ep, Pair};
    use ExplicitNeutrino::{Nue, Nuebar, Numu, Numubar, Nutau, Nutaubar};
    use NeutrinoBank::{Nue as BNue, Nuebar as BNuebar, Nux};

    const ME: f64 = 0.510_998_950_0;
    const STATES: [ExplicitNeutrino; 6] = [Nue, Nuebar, Numu, Numubar, Nutau, Nutaubar];
    const CHANNELS: [ElectronChannel; 3] = [Em, Ep, Pair];
    type Event = [[f64; 4]; 4];

    fn close(a: f64, b: f64, tolerance: f64) {
        assert!((a - b).abs() <= tolerance * a.abs().max(b.abs()).max(1.0e-300));
    }
    fn different(a: f64, b: f64) {
        assert!((a - b).abs() > 1.0e-4 * a.abs().max(b.abs()));
    }
    fn momenta(event: Event) -> [FourMomentumMeV; 4] {
        event.map(|p| FourMomentumMeV::new(p[0], p[1], p[2], p[3]).unwrap())
    }
    fn swap(mut event: Event, a: usize, b: usize) -> Event {
        event.swap(a, b);
        event
    }
    fn elastic() -> Event {
        let (energy, cosine): (f64, f64) = (2.3, 0.27);
        let outgoing = energy / (1.0 + energy * (1.0 - cosine) / ME);
        let sine = (1.0 - cosine * cosine).sqrt();
        let p3 = [outgoing, outgoing * sine, 0.0, outgoing * cosine];
        let p4 = [ME + energy - p3[0], -p3[1], 0.0, energy - p3[3]];
        [[energy, 0.0, 0.0, energy], [ME, 0.0, 0.0, 0.0], p3, p4]
    }
    fn pair(mass: f64, energy: f64, cosine: f64) -> Event {
        let momentum = (energy * energy - mass * mass).sqrt();
        let sine = (1.0 - cosine * cosine).sqrt();
        let p3 = [energy, momentum * sine, 0.0, momentum * cosine];
        let p4 = [energy, -p3[1], 0.0, -p3[3]];
        let p1 = [energy, 0.0, 0.0, energy];
        let p2 = [energy, 0.0, 0.0, -energy];
        [p1, p2, p3, p4]
    }
    fn assert_physical(event: Event, masses: [f64; 4]) {
        for axis in [0, 1, 2, 3] {
            let residual = event[0][axis] + event[1][axis] - event[2][axis] - event[3][axis];
            assert!(residual.abs() <= 1.0e-13);
        }
        for (p, mass) in event.into_iter().zip(masses) {
            let shell = p[0] * p[0] - p[1] * p[1] - p[2] * p[2] - p[3] * p[3];
            assert!((shell - mass * mass).abs() <= 1.0e-13);
        }
    }
    fn proc(target: ExplicitNeutrino, channel: ElectronChannel) -> ExplicitElectronProcess {
        EXPLICIT_ELECTRON_PROCESSES
            .into_iter()
            .find(|row| row.target() == target && row.channel() == channel)
            .unwrap()
    }

    fn independent(row: ExplicitElectronProcess) -> f64 {
        let right = 2.0 * 0.231_22;
        let left = match row.target().bank() {
            Nux => -1.0 + right,
            _ => 1.0 + right,
        };
        if row.channel() == Pair {
            let energy: f64 = 1.7;
            let cosine: f64 = 0.37;
            let momentum = (energy * energy - ME * ME).sqrt();
            let (plus, minus) = (energy + momentum * cosine, energy - momentum * cosine);
            let terms = left * left * plus * plus
                + right * right * minus * minus
                + 2.0 * left * right * ME * ME;
            return 32.0 * energy * energy * terms;
        }
        let (left, right) = if row.target().is_antineutrino() {
            (right, left)
        } else {
            (left, right)
        };
        let (energy, cosine): (f64, f64) = (2.3, 0.27);
        let outgoing = energy / (1.0 + energy * (1.0 - cosine) / ME);
        let (leading, recoil) = if row.channel() == Em {
            (left, right)
        } else {
            (right, left)
        };
        let terms = leading * leading * energy * energy + recoil * recoil * outgoing * outgoing
            - left * right * ME * (energy - outgoing);
        32.0 * ME * ME * terms
    }
    fn event(row: ExplicitElectronProcess) -> Event {
        match (row.channel(), row.target().is_antineutrino()) {
            (Pair, true) => swap(pair(ME, 1.7, 0.37), 2, 3),
            (Pair, false) => pair(ME, 1.7, 0.37),
            _ => elastic(),
        }
    }
    fn hm(row: ExplicitElectronProcess, event: Event) -> f64 {
        reduced_w(row, momenta(event), Mass::new(ME).unwrap())
    }
    fn value(row: ExplicitElectronProcess) -> f64 {
        event_factor(hm(row, event(row)), Temp::new(1.7).unwrap()).value()
    }
    fn compare(left: ExplicitElectronProcess, a: Event, right: ExplicitElectronProcess, b: Event) {
        close(hm(left, a), hm(right, b), 1.0e-12);
    }
    fn response(target: ExplicitNeutrino, inputs: [f64; 6]) -> f64 {
        CHANNELS
            .into_iter()
            .map(|channel| {
                let row = proc(target, channel);
                let input = STATES
                    .iter()
                    .position(|state| *state == row.input())
                    .unwrap();
                inputs[input] * value(row)
            })
            .sum()
    }
    fn mandelstam(event: Event) -> [f64; 3] {
        let square = |a: usize, b: usize, sign: f64| {
            let q = [0, 1, 2, 3].map(|i| event[a][i] + sign * event[b][i]);
            q[0] * q[0] - q[1] * q[1] - q[2] * q[2] - q[3] * q[3]
        };
        [square(0, 1, 1.0), square(0, 2, -1.0), square(0, 3, -1.0)]
    }

    #[test]
    fn massless_scale_bridge_exposes_double_node_scaling() {
        let row = proc(Nue, Pair);
        let mass = Mass::new(0.0).unwrap();
        let (y, x) = (NeutrinoY::new(1.4).unwrap(), ElectronX::new(1.4).unwrap());
        let (t1, t2) = (Temp::new(1.0).unwrap(), Temp::new(2.0).unwrap());
        let e1 = y.momentum(t1).value();
        let e2 = y.momentum(t2).value();
        close(e1, x.momentum(t1).value(), 1.0e-15);
        close(e2, x.momentum(t2).value(), 1.0e-15);
        let unit = pair(0.0, e1, 0.31);
        let scaled = pair(0.0, e2, 0.31);
        let w1 = reduced_w(row, momenta(unit), mass);
        let w2 = reduced_w(row, momenta(scaled), mass);
        close(w2 / w1, 2.0_f64.powi(4), 1.0e-15);
        let r1 = event_factor(w1, t1).value();
        let r2 = event_factor(w2, t2).value();
        close(r2 / r1, 2.0_f64.powi(5), 1.0e-15);
        let physical_path = t2.value() * physical_w(row, momenta(scaled), mass);
        close(r2, physical_path, 1.0e-15);
        let wrong_w = reduced_w(row, momenta(pair(0.0, e2 / t2.value(), 0.31)), mass);
        let double_divided = event_factor(wrong_w, t2).value();
        close(double_divided / r1, 2.0, 1.0e-15);
        different(double_divided, r2);
    }

    #[test]
    fn physical_hm_events_match_independent_reductions() {
        assert_physical(elastic(), [0.0, ME, 0.0, ME]);
        assert_physical(pair(ME, 1.7, 0.37), [0.0, 0.0, ME, ME]);
        for row in EXPLICIT_ELECTRON_PROCESSES {
            let momenta = momenta(event(row));
            let actual = reduced_w(row, momenta, Mass::new(ME).unwrap());
            close(actual, independent(row), 1.0e-12);
        }
    }

    #[test]
    fn cp_crossing_and_wrong_permutations_are_discriminating() {
        let (elastic, pair) = (elastic(), pair(ME, 1.7, 0.37));
        for target in [Nue, Numu, Nutau] {
            let anti = target.conjugate();
            compare(proc(target, Em), elastic, proc(anti, Ep), elastic);
            compare(proc(target, Ep), elastic, proc(anti, Em), elastic);
            compare(proc(target, Pair), pair, proc(anti, Pair), swap(pair, 2, 3));
        }
        let expected = hm(proc(Nue, Pair), pair);
        let crossed = [pair[0], pair[3].map(|x| -x), pair[1].map(|x| -x), pair[2]];
        let [s, t, u] = mandelstam(pair);
        close(s + t + u, 2.0 * ME * ME, 1.0e-13);
        for (actual, expected) in mandelstam(crossed).into_iter().zip([u, s, t]) {
            close(actual, expected, 1.0e-13);
        }
        close(expected, hm(proc(Nue, Em), crossed), 1.0e-12);
        different(expected, hm(proc(Nue, Pair), swap(pair, 0, 1)));
        different(expected, hm(proc(Nue, Pair), swap(pair, 2, 3)));
    }

    #[test]
    fn full_sector_fold_uses_four_only_in_readout() {
        let unweighted = STATES.map(|state| response(state, [1.0; 6]));
        for heavy in &unweighted[3..] {
            close(*heavy, unweighted[2], 1.0e-13);
        }
        let sentinel = [2.0, 3.0, 5.0];
        let lifted = lift_three_to_six(sentinel);
        let explicit = STATES.map(|state| response(state, lifted));
        let projected = project_six_to_three(explicit);
        let mut folded = [0.0; 3];
        let groups: [&[ExplicitNeutrino]; 3] = [&STATES[0..1], &STATES[1..2], &STATES[2..6]];
        for (bank, rows) in FOLDED_ELECTRON_CHANNELS.chunks_exact(3).enumerate() {
            for row in rows {
                let average = groups[bank]
                    .iter()
                    .map(|&state| value(proc(state, row.channel())))
                    .sum::<f64>()
                    / groups[bank].len() as f64;
                let input = match row.input() {
                    BNue => sentinel[0],
                    BNuebar => sentinel[1],
                    Nux => sentinel[2],
                };
                folded[bank] += input * average;
            }
        }
        for bank in 0..3 {
            close(projected[bank], folded[bank], 1.0e-13);
        }
        let catalogue_sum: f64 = EXPLICIT_ELECTRON_PROCESSES.into_iter().map(value).sum();
        close(catalogue_sum, unweighted.iter().sum(), 1.0e-13);
        close(explicit.iter().sum(), conserved_readout(projected), 1.0e-13);
    }
}
