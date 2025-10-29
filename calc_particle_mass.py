import numpy as np
import gitHelp as gh
import os
import math
import unicodedata

cdir = os.path.split(__file__)[0]

# This script generates a particle mass table (in kg) for all particles in the PDG catalog
# except quarks and nuclei/ions. Neutrinos and photon are set to 0 mass.
# Particle/antiparticle share the same value via |PDG|.
#
# Requires: pip install particle

# ---- constants / units ----
# 1 MeV/c^2 = 1.78266192e-30 kg
MEV_TO_KG = 1.78266192e-30

# ---- PDG access ----
try:
    from particle import Particle
except ImportError as e:
    raise SystemExit(
        "This script requires 'particle' (scikit-hep). Install with:\n"
        "    pip install particle"
    ) from e


def is_quark(p):
    """Exclude quarks: PDGIDs ±1..±6."""
    return abs(p.pdgid) in {1, 2, 3, 4, 5, 6}


def is_nucleus_or_ion(p):
    """Exclude nuclei/ions (10LZZZAAAI-encoded; usually pdgid >= 1e9)."""
    if abs(p.pdgid) >= 1_000_000_000:
        return True


def forced_mass_mev(pdgid):
    """CRPropa convention: neutrinos and photon massless."""
    if abs(pdgid) in {12, 14, 16, 22}:
        return 0.0
    p = Particle.from_pdgid(pdgid)
    m = p.mass  # MeV
    if m is None or (isinstance(m, float) and math.isnan(m)):
        raise ValueError("no mass")
    return float(m)


# Build dictionary keyed by |PDG| -> representative PDGID (>0 when possible)
absid_to_pdgid = {}
for p in Particle.findall():
    if is_quark(p):
        continue
    print(p.pdgid)
    if is_nucleus_or_ion(p):
        continue

    aid = abs(p.pdgid)
    if aid not in absid_to_pdgid:
        try:
            Particle.from_pdgid(aid)  # check positive exists
            absid_to_pdgid[aid] = aid
        except Exception:
            absid_to_pdgid[aid] = p.pdgid

# Prepare rows: (pdgid, mass_kg, name, mass_mev)
rows = []
for aid in sorted(absid_to_pdgid.keys()):
    pdgid = absid_to_pdgid[aid]
    try:
        m_mev = forced_mass_mev(pdgid)
    except ValueError:
        # Skip entries without numeric mass if not massless-by-convention
        continue
    m_kg = m_mev * MEV_TO_KG
    name = Particle.from_pdgid(pdgid).name or f"PDG{pdgid}"
    rows.append((pdgid, m_kg, name, m_mev))

# output folder
folder = "data"
if not os.path.exists(folder):
    os.makedirs(folder)

# Write to file
fout = open(os.path.join(folder, "particle_masses.txt"), "w")

fout.write("# PDG masses [kg]; particle/antiparticle share the same value via |PDG|\n")
try:
    git_hash = gh.get_git_revision_hash()
    fout.write("# Produced with crpropa-data version: " + git_hash + "\n")
except:
    pass

# Payload lines: "<PDGID> <mass_kg>    # <name> (<mass_MeV> MeV/c^2)"
for pdgid, m_kg, name, m_mev in rows:
    m_kg_str = f"{m_kg:.10e}" if m_kg != 0.0 else "0.0"
    fout.write(f"{pdgid:>6d} {m_kg_str:<18}    # {name} ({m_mev:.6g} MeV/c^2)\n")

fout.close()
