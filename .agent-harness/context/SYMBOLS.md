# Symbol and Interface Table

| Symbol / interface | Definition | Domain / type | Units / dimensions | Sign / branch convention | Source of truth |
|---|---|---|---|---|---|
| `a` | FLRW scale factor | positive scalar | dimensionless | `a(N=0)=1` | `isotropic_boltzmann.rs` |
| `N` | `ln(a)`, independent ODE variable | real scalar | dimensionless | increases during expansion | `ode.rs`, `isotropic_boltzmann.rs` |
| `p` | physical neutrino momentum magnitude | nonnegative | MeV | future-directed energy `E=p` for massless neutrinos | F-10 spec |
| `q` | comoving momentum `a p` | nonnegative | MeV | constant under collisionless redshifting | `ComovingMomentumGrid` |
| `y` | `q/T_ref` | positive node | dimensionless | selected map `y=-3 ln(1-t)`, `t in (0,1)` | `quadrature.rs` |
| `T_ref` | initial/reference comoving temperature | positive scalar | MeV | endpoint test fixes 10 MeV | `IsotropicBoltzmannFlrwSystem` |
| `T_cm` | `T_ref exp(-N)` | positive scalar | MeV | collision scale, decreases with expansion | `isotropic_boltzmann.rs` |
| `T_gamma` | electromagnetic bath temperature | positive ODE state | MeV | terminal crossing direction is negative | `isotropic_boltzmann.rs` |
| `f_E,f_X` | folded electron and heavy neutrino-pair occupations | vector entries in `(0,1)` | dimensionless | zero lepton asymmetry; heavy mu/tau shapes degenerate | `isotropic_boltzmann.rs` |
| `u` | logit `ln[f/(1-f)]` | finite ODE coordinate | dimensionless | logistic inverse; no clipping | `isotropic_boltzmann.rs` |
| `C[f]` | classical diagonal collision action | grid vector | inverse time in natural units | gain minus loss; equilibrium null | `electron_spectral.rs`, `neutrino_self_spectral.rs` |
| `H` | flat-FLRW Hubble rate | positive scalar | MeV, converted to s^-1 | positive expanding branch | `flrw.rs` |
| `G_F` | Fermi constant | positive scalar | MeV^-2 | repository constant; never refit | `electron_hm.rs` |
| `K_s` | `(p_1.p_2)(p_3.p_4)` | nonnegative on physical event | MeV^4 | metric/product convention inherited from HM event algebra | F-10 spec, `neutrino_self_spectral.rs` |
| `K_t` | `(p_1.p_4)(p_2.p_3)` | nonnegative on physical event | MeV^4 | directed crossed kernel | F-10 spec |
| `eta` | time-orientation/symmetry factor in global four-leg form | positive scalar | dimensionless | same-flavour elastic uses 1/2 | F-10 catalogue |
| `Q_E,Q_X` | electron/heavy collision energy moments | signed scalar | MeV^5 | positive heats that neutrino block; EM receives equal opposite electron debit | `electron_spectral.rs`, `isotropic_boltzmann.rs` |
| `N_eff` | conventional energy-density readout | positive scalar | dimensionless | regression diagnostic only in current same-code endpoint | endpoint test |
| `x_F` | FortEPiaNO independent variable `m_e/T_cm` | positive scalar | dimensionless | starts at `0.051099895` and increases | FortEPiaNO 1.4.0 adapter contract |
| `z_F` | FortEPiaNO photon ratio `T_gamma/T_cm` | positive scalar | dimensionless | event uses `m_e z_F/x_F=0.005 MeV` on the decreasing branch | FortEPiaNO 1.4.0 adapter contract |
| `IsotropicBoltzmannFlrwSystem` | crate-private coupled RHS/Jacobian and physical-state consumer | Rust type | n/a | no public dispatch authority | `isotropic_boltzmann.rs` |

Any CAS axis may introduce internal names, but its result must map them back to this table and the shared CAS contract.
