"""Resolve corpus topic labels against MeSH. Lookup only -- no model, no guessing.

MeSH (NLM Medical Subject Headings, public domain) is the vocabulary that indexes
PubMed. Using it replaces two things I had been approximating with hand-written
regexes, and replaces them with reference data:

  * DOMAIN comes from the tree number rather than a keyword table. "Septic Shock"
    sits at C01.757.800, and C01 is Infections -- no lexicon line required, and no
    argument about whether it is infectious or cardiovascular.
  * HIERARCHY comes from the tree number too. C19.246.300 is a child of C19.246,
    which is a child of C19. That is a real taxonomy, not a suffix-matching trick
    that put "Statin Therapy" and "Oxygen Therapy" in the same family.

A concept keeps a stable identifier (D003924) that survives re-processing, so URLs
and saved links stay valid even when the pipeline is re-run with better models.

Matching is deliberately conservative: exact normalised match against headings and
entry synonyms, then a light singular/plural fold. Anything that does not match is
reported as unmatched rather than approximated -- a wrong MeSH code is worse than
none, because it silently misfiles a concept under a specialty it does not belong to.
"""
import re
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MESH_BIN = BASE / "data" / "ref" / "mesh.bin"

# MeSH tree top/second level -> the clinical domains this site groups by.
# Longest prefix wins, so C19 beats C.
TREE_DOMAIN = [
    ("C01", "Infectious"), ("C02", "Infectious"), ("C03", "Infectious"),
    ("C04", "Hematology/Oncology"), ("C15", "Hematology/Oncology"),
    ("C05", "Musculoskeletal"), ("C26", "Musculoskeletal"),
    ("C06", "Gastrointestinal"),
    ("C07", "Dental/ENT"), ("C09", "Dental/ENT"),
    ("C08", "Pulmonary"),
    ("C10", "Neurology"),
    ("C11", "Ophthalmology"),
    ("C12", "Renal/Urologic"),
    # Current MeSH folded the old C13 into C12: C12.050 is Female Urogenital
    # Diseases AND Pregnancy Complications, so pregnancy sits at C12.050.703.
    # Longest prefix wins, so these override the generic C12 above.
    ("C12.050", "Gynecology"),
    ("C12.050.703", "Obstetrics"),
    ("C13", "Obstetrics"),
    ("C14", "Cardiovascular"),
    ("C16", "Pediatrics"),
    ("C17", "Dermatology"),
    ("C18", "Endocrine"), ("C19", "Endocrine"),
    ("C20", "Immunology"),
    ("C23", "Signs/Symptoms"),
    ("C25", "Toxicology"),
    ("D", "Pharmacology"),
    ("E01", "Diagnostics"), ("E02", "Procedures"), ("E04", "Procedures"),
    ("E05", "Diagnostics"), ("E06", "Dental/ENT"), ("E07", "Procedures"),
    ("E03", "Procedures"),
    ("F01", "Psychiatry"), ("F02", "Psychiatry"), ("F03", "Psychiatry"),
    ("F04", "Psychiatry"),
    ("G", "Physiology"),
    ("N", "Health Care"),
    ("I", "Health Care"), ("K", "Health Care"), ("L", "Health Care"),
    ("A", "Anatomy"), ("B", "Organisms"), ("H", "Health Care"),
    ("J", "Health Care"), ("M", "Health Care"), ("V", "Health Care"),
    ("Z", "Health Care"),
]


def norm(s: str) -> str:
    s = re.sub(r"\(.*?\)", " ", s or "")               # drop parentheticals
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def singular(s: str) -> str:
    if s.endswith("ies"):
        return s[:-3] + "y"
    if s.endswith("ses") or s.endswith("xes"):
        return s[:-2]
    if s.endswith("s") and not s.endswith("ss"):
        return s[:-1]
    return s


class Mesh:
    def __init__(self, path: Path = MESH_BIN):
        self.by_term = {}        # normalised term -> UI
        self.heading = {}        # UI -> preferred heading
        self.trees = {}          # UI -> [tree numbers]
        self.by_tree = {}        # tree number -> UI
        self._load(path)

    def _load(self, path: Path) -> None:
        mh = ui = None
        mns, terms = [], []
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line == "*NEWRECORD":
                    self._commit(mh, ui, mns, terms)
                    mh = ui = None
                    mns, terms = [], []
                elif line.startswith("MH = "):
                    mh = line[5:].strip()
                elif line.startswith("UI = "):
                    ui = line[5:].strip()
                elif line.startswith("MN = "):
                    mns.append(line[5:].strip())
                elif line.startswith("ENTRY = ") or line.startswith("PRINT ENTRY = "):
                    # "Term|T109|T195|..." -- only the leading term is the synonym
                    terms.append(line.split("=", 1)[1].split("|")[0].strip())
        self._commit(mh, ui, mns, terms)

    def _commit(self, mh, ui, mns, terms) -> None:
        if not (mh and ui):
            return
        self.heading[ui] = mh
        self.trees[ui] = mns
        for t in mns:
            self.by_tree.setdefault(t, ui)
        for t in [mh] + terms:
            n = norm(t)
            if n:
                self.by_term.setdefault(n, ui)
                self.by_term.setdefault(singular(n), ui)
        # "Diabetes Mellitus, Type 2" also reachable as "type 2 diabetes mellitus"
        if "," in mh:
            head, _, rest = mh.partition(",")
            inv = norm(f"{rest} {head}")
            if inv:
                self.by_term.setdefault(inv, ui)
                self.by_term.setdefault(singular(inv), ui)

    # Clinical shorthand that MeSH files under a formal name. Every entry here is
    # a term clinicians actually say that failed to resolve; the value is the MeSH
    # preferred heading. Kept small and explicit rather than fuzzy-matching, which
    # would misfile concepts silently.
    ALIASES = {
        "beta blockers": "Adrenergic beta-Antagonists",
        "beta blocker": "Adrenergic beta-Antagonists",
        "ace inhibitors": "Angiotensin-Converting Enzyme Inhibitors",
        "arbs": "Angiotensin Receptor Antagonists",
        "nonsteroidal anti inflammatory drugs": "Anti-Inflammatory Agents, Non-Steroidal",
        "nsaids": "Anti-Inflammatory Agents, Non-Steroidal",
        "ppis": "Proton Pump Inhibitors",
        "ssris": "Selective Serotonin Reuptake Inhibitors",
        "statins": "Hydroxymethylglutaryl-CoA Reductase Inhibitors",
        "hemoglobin a1c": "Glycated Hemoglobin",
        "a1c": "Glycated Hemoglobin",
        "lipid panel": "Lipids",
        "point of care ultrasound": "Point-of-Care Systems",
        "lung ultrasound": "Ultrasonography",
        "transvaginal ultrasound": "Ultrasonography",
        "ultrasound": "Ultrasonography",
        "digital rectal exam": "Digital Rectal Examination",
        "bipap": "Noninvasive Ventilation",
        "cpap": "Continuous Positive Airway Pressure",
        "copd exacerbation": "Pulmonary Disease, Chronic Obstructive",
        "somatic symptom disorder": "Medically Unexplained Symptoms",
        "glp 1 agonists": "Glucagon-Like Peptide-1 Receptor Agonists",
        "sglt2 inhibitors": "Sodium-Glucose Transporter 2 Inhibitors",
        "shared decision making": "Decision Making, Shared",
        "sleep hygiene": "Sleep Hygiene",
    }

    # ------------------------------------------------------------------
    def lookup(self, label: str) -> str:
        """UI for a label, or None. Conservative on purpose."""
        n = norm(label)
        alias = self.ALIASES.get(n) or self.ALIASES.get(singular(n))
        if alias:
            ui = self.by_term.get(norm(alias))
            if ui:
                return ui
        for cand in (n, singular(n)):
            if cand in self.by_term:
                return self.by_term[cand]
        # drop a leading qualifier: "acute pancreatitis" -> "pancreatitis"
        words = n.split()
        for i in range(1, min(3, len(words))):
            tail = " ".join(words[i:])
            if tail in self.by_term:
                return self.by_term[tail]
            if singular(tail) in self.by_term:
                return self.by_term[singular(tail)]
        return None

    def domain(self, ui: str) -> str:
        best, blen = "General", -1
        for t in self.trees.get(ui, []):
            for pref, dom in TREE_DOMAIN:
                if t.startswith(pref) and len(pref) > blen:
                    best, blen = dom, len(pref)
        return best

    def parent(self, ui: str) -> str:
        """Nearest ancestor that is itself a descriptor, via tree numbers."""
        for t in sorted(self.trees.get(ui, []), key=len):
            parts = t.split(".")
            while len(parts) > 1:
                parts = parts[:-1]
                p = self.by_tree.get(".".join(parts))
                if p and p != ui:
                    return p
        return None

    def depth(self, ui: str) -> int:
        ts = self.trees.get(ui) or []
        return min((t.count(".") for t in ts), default=0)
