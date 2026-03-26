# Literature and tool familiarization
## PLATO Mission Goals
> The PLATO mission (PLAnetary Transits and Oscillations of stars) is designed to detect and characterise a large number of exoplanetary systems, including terrestrial exoplanets orbiting bright solar-type stars in their habitable zone [1].

The mission adresses three overall scientific questions:

* How do planets and planetary systems form and evolve?
* Is our Solar System special or are there other systems like ours?
* Are there potentially habitable planets?

To achieve these goals, PLATO relies on the photometric transit method: as a planet crosses the face of its host star along our line of sight, it causes a small, periodic dimming of the observed stellar flux. PLATO will monitor these brightness variations continuously, producing high-precision light curves from which both planetary and stellar parameters can be extracted:

- Stellar
	- Evolutionary ages
	- Magnetic variability
	- Activity
- Planetary
	- Radii
	- Masses
	- Densities
	- Ages

A critical and distinguishing feature of PLATO is its systematic use of asteroseismology — the analysis of stellar oscillation frequencies — to precisely characterise the host stars of detected planets. The frequencies of these oscillations, which can be trapped acoustic waves, internal gravity waves, or a mixture of the two, depend on the radially varying density and internal sound speed of the star. By fitting observed oscillation spectra to stellar models, one can infer a star's mass, radius, and crucially its age — a quantity that is otherwise very difficult to determine. PLATO will be the first mission to make systematic use of asteroseismology to characterise planet host stars, allowing us to link planetary and stellar evolution [2]. 

## Why is simulation necessary

Simulated light curves are essential for developing and validating the procedures used to determine stellar and planetary parameters before real PLATO data becomes available. They enable _hare-and-hounds exercises_: one team generates a set of artificial observations based on a known physical model, and a separate team attempts to recover those underlying parameters blindly [3] . This approach allows pipeline developers to benchmark their methods under controlled conditions, identify systematic biases, and assess detection limits.

Beyond pipeline testing, the preparation of PLATO's science objectives will require the massive generation of representative simulated light curves to support these exercises at scale. Crucially, simulations must be realistic — they need to reproduce not just the stellar signal, but the full noise environment a real observation would contain, including both stellar variability and instrumental effects.
### PSLS

PSLS is a Python tool that simulates solar-like oscillators representative of PLATO targets. It includes planetary transits, stochastically-excited oscillations, granulation and activity components, as well as instrumental systematic errors and random noise representative of PLATO. It was developed specifically to support the scientific preparation of the mission [3].

The simulator handles two main classes of physical signal:

- **Stellar signals:** solar-like p-mode oscillations (modelled either via the Universal Pattern of Mosser et al. 2011, or from theoretical frequencies computed with the ADIPLS pulsation code), convective granulation, and magnetic activity (modelled either as a stochastic background or as a spot component following Dorren 1987).
- **Planetary transits:** implemented following Mandel & Agol (2002) equations.

On the instrumental side, the instrumental noise level is quantified by carrying out realistic simulations at CCD pixel level using the Plato Image Simulator (PIS). The simulated light curves are then corrected for instrumental effects using the instrument point spread functions reconstructed on the basis of a microscanning technique to be operated during in-flight calibration. 

Key findings from the original PSLS paper establish the operating regime of these instrumental systematics: instrumental systematic errors dominate the signal only at frequencies below ~20 μHz, with the level of systematic errors found to mainly depend on stellar magnitude and on the detector charge transfer inefficiency. The simulator's realism was validated by comparing its predictions against Kepler observations of three typical targets, finding good qualitative agreement.

The most recent version (v1.9, September 2025) further extends PSLS with the inclusion of flares and improved mode splitting treatment, making it an actively maintained and growing tool for mission preparation.

## What is Stellar Activity and Why it Contaminates Transit Detection

Stellar activity refers to the ensemble of physical processes that cause a star's brightness to vary intrinsically over time. These arise from the interplay of magnetic fields, convection, and rotation at the stellar surface, and they manifest across a wide range of timescales and amplitudes. The main contributors relevant to PLATO photometry are:

- **Starspots and faculae:** dark and bright magnetic surface features that rotate in and out of view with the stellar rotation period, producing quasi-periodic flux modulations. These rotationally modulated signals both increase the fractional depth of exoplanet transits and increase the variability of the background against which transits are detected [4].
- **Granulation:** convective cells at the stellar surface producing stochastic brightness fluctuations. Granulation operates on timescales from ~20 minutes up to several days, with amplitudes reaching a few hundred ppm [5]. This is particularly problematic for small planets: an Earth-sized planet transiting a Sun-like star produces a transit depth of only ~80 ppm, directly comparable to granulation amplitudes.
- **Oscillations:** solar-like p-mode oscillations with periods of a few minutes and amplitudes of ~10 ppm in solar-type stars, which add stochastic noise to the light curve at high frequencies.
- **Magnetic activity cycles:** long-term brightness trends driven by activity cycles analogous to the solar 11-year cycle.

From an observational standpoint, stellar activity poses a critical challenge to exoplanet science, as it inhibits both the detection of planets and the precise measurement of their parameters [2]. The key problem is one of signal confusion: activity-driven brightness changes can mimic, mask, or distort a transit signature. Rotationally modulated variability is the dominant low-frequency noise source and can be partially removed by detrending, but granulation and oscillations on transit timescales are harder to separate from the transit shape itself. Correctly accounting for all of these components — and understanding how they interact — is one of the central challenges PSLS is designed to address.

### Sources

[1] H. Rauer _et al._, “The PLATO mission,” _Exp Astron_, vol. 59, no. 3, p. 26, Jun. 2025, doi: [10.1007/s10686-025-09985-9](https://doi.org/10.1007/s10686-025-09985-9).

[2] G. Bruno and M. Deleuil, “Stellar activity and transits,” Apr. 13, 2021, _arXiv_: arXiv:2104.06173. doi: [10.48550/arXiv.2104.06173](https://doi.org/10.48550/arXiv.2104.06173).

[3] R. Samadi _et al._, “The PLATO Solar-like Light-curve Simulator: A tool to generate realistic stellar light-curves with instrumental effects representative of the PLATO mission,” _A&A_, vol. 624, p. A117, Apr. 2019, doi: [10.1051/0004-6361/201834822](https://doi.org/10.1051/0004-6361/201834822).

[4] A. C. Cameron, “The Impact of Stellar Activity on the Detection and Characterization of Exoplanets,” in _Handbook of Exoplanets_, Springer, Cham, 2018, pp. 1791–1799. doi: [10.1007/978-3-319-55333-7_23](https://doi.org/10.1007/978-3-319-55333-7_23).

[5] S. C. C. Barros _et al._, “Improving transit characterisation with Gaussian process modelling of stellar variability,” _A&A_, vol. 634, p. A75, Feb. 2020, doi: [10.1051/0004-6361/201936086](https://doi.org/10.1051/0004-6361/201936086).

# Setup of PSLS

Install the source code from https://pypi.org/project/psls/#files.

After fixing a compatibility issue in the `FortranIO.py` file I was able to run the example included in the package with the following comand:

```
psls.py -P -V psls.yaml
```

I obtained the following plots: 

![psd](project/PSLS/0012069449_fig1.png)
![LC](project/PSLS/0012069449_fig5.png)
