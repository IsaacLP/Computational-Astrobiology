# Literature and tool familiarization

## PLATO Mission Goals
> The PLATO mission (PLAnetary Transits and Oscillations of stars) is designed to detect and characterise a large number of exoplanetary systems, including terrestrial exoplanets orbiting bright solar-type stars in their habitable zone (Rauer et al 2025)

Overall objectives:
* How do planets and planetary systems form and evolve?
* Is our Solar System special or are there other systems like ours?
* Are there potentially habitable planets?

It will study planets this via the _photometric transit method_. The _light curve_ data will also enable to study the stars. 

cameras -> high-precision photometric light curves -> stellar modeling and characterisation -> planetary parameters 

Key results:
- Stellar
	- Evolutionary ages
	- Magnetic variability
	- Activity
- Planetary
	- Radii
	- Masses
	- Densities
	- Ages

## Why is simulation necessary

Simulated light-curves are needed to develop and test the procedures for determining stellar parameters. They are used to conduct hare-and-hounds exercises: one team produces a set of artificial observations and another tries to infer the physical model/properties behind these observations.

### PSLS

The Plato Stellar Light-curve Simulator3 (PSLS) aims at simulating stochastically-excited oscillations together with planetary transits, stellar signal (granulation, activity) and instrumental sources of noise that are representative of the PLATO cameras.

## What is Stellar Activity

Stellar activity is the predominant source of noise in the signal at low frequencies

###  Why it contaminates transit detection

The presence of stellar activity is critical for the detection of planetary transits.

### Sources

1. Rauer, H. et al. The PLATO mission. Exp Astron 59, 26 (2025).
2. Samadi, R. et al. The PLATO Solar-like Light-curve Simulator: A tool to generate realistic stellar light-curves with instrumental effects representative of the PLATO mission. A&A 624, A117 (2019).

# Setup of PSLS

After installing and performing some fixes I ran the example using:

```
psls.py -P -V psls.yaml
```

I obtained the plots included in the repository. 