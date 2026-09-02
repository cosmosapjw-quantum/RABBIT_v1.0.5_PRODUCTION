# D-081R0 SciSpace literature receipt

Status: `LITERATURE_SUPPORTED_CONTEXT_ONLY`

The search asked for peer-reviewed work supporting independently validated analytic Jacobians in compiled stiff integration for momentum-dependent primordial-neutrino decoupling, while separating semantic validation from performance measurement and preserving full collision physics.

- Froustey, Pitrou and Volpe, *Neutrino decoupling including flavour oscillations and primordial nucleosynthesis*, JCAP 12 (2020) 015, DOI `10.1088/1475-7516/2020/12/015`: closest methodological precedent for direct differential-system Jacobian construction in momentum-dependent neutrino decoupling. It motivates the Rust analytic-Jacobian route but does not validate RABBIT's discretisation.
- Hannestad, Hansen, Tram and Wong, *Active-sterile neutrino oscillations in the early Universe with full collision terms*, JCAP 08 (2015) 019, DOI `10.1088/1475-7516/2015/08/019`: supports retaining full collision physics and explicit Pauli/momentum-redistribution checks when comparing numerical implementations.
- Bennett et al., *Towards a precision calculation of N_eff in the Standard Model II*, JCAP 04 (2021) 073, DOI `10.1088/1475-7516/2021/04/073`: supports treating momentum discretisation and numerical convergence as separate uncertainty components.
- Blaschke and Cirigliano, *Neutrino Quantum Kinetic Equations: The Collision Term*, arXiv:1605.09383: broader collision-formalism context; not direct authority for the present classical diagonal no-QKE comparator.

The attempted SciSpace table-column augmentation returned `not_found`; no persistent SciSpace table update is claimed.
