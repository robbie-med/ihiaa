"""Build a topic hierarchy: domain -> root concept -> variants. Rules only, no model.

The flat topic list is unusable: 4,465 labels for 256 lectures, 76% of which appear in
exactly one episode. They are not duplicates -- "Gestational Hypertension",
"Hypertensive Emergency" and "Resistant Hypertension" are genuinely distinct concepts
-- they are just over-specific, with no structure connecting them.

Two deterministic passes give that structure:

  1. PARENT LINKING by head-noun suffix. English medical labels put the head last, so
     a label ending with another known label is a specialisation of it:
         "Gestational Hypertension"  -> "Hypertension"
         "Diabetic Foot Ulcer"       -> "Foot Ulcer" -> "Ulcer"
     Chains are resolved to a root. No semantics required, so it always agrees with
     itself.

  2. DOMAIN ASSIGNMENT by lexicon. Each root maps to one clinical domain via an
     explicit keyword table below. First match wins, and the table is ordered so that
     specific patterns are tested before generic ones. Unmatched roots land in
     "General" rather than being guessed at.

Output data/taxonomy.json feeds the Topics view and the concept map, where domain
becomes colour and the parent links become branches.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SITE = BASE / "site" / "data.json"
OUT = BASE / "data" / "taxonomy.json"

# Ordered: the first pattern that matches wins, so put specific before generic.
DOMAINS = [
    ("Obstetrics",     r"pregnan|obstetric|prenatal|antepartum|postpartum|labor|"
                       r"delivery|fetal|fetus|miscarriage|preeclampsia|eclampsia|"
                       r"gestational|breastfeed|lactation|contracepti|abortion|"
                       r"cesarean|amnio|placenta|neonat|newborn|magnesium sulfate|tocolytic|oxytocin"),
    ("Gynecology",     r"gyneco|cervi|vulva|vagin|uter|ovar|menstrua|menopaus|"
                       r"pap smear|colposcop|endometri|pelvic|fibroid|pcos|dysmenorrh|menorrhagi|dyspareun|infertil"),
    ("Cardiovascular", r"cardi|heart|hypertens|blood pressure|lipid|cholesterol|"
                       r"statin|atrial|arrhythm|angina|infarct|coronary|"
                       r"aneurysm|vascular|varicose|venous|dvt|embol|stroke|"
                       r"peripheral arterial|pad\b|pvd\b|edema|syncope"),
    ("Endocrine",      r"diabet|thyroid|insulin|metformin|glucose|glycem|a1c|"
                       r"adrenal|pituitar|testosterone|estrogen|obes|bariatric|"
                       r"osteoporos|calcium|vitamin d|metabolic syndrome"),
    ("Infectious",     r"infect|antibiotic|sepsis|abscess|cellulit|hiv|hepatitis|"
                       r"tubercul|malaria|parasit|viral|virus|bacteri|fungal|"
                       r"vaccin|immuniz|influenza|covid|pneumon|uti\b|std|sti\b|"
                       r"syphilis|gonorrh|chlamyd|tropical|dengue|typhoid|"
                       r"osteomyelitis|meningitis|septic"),
    ("Psychiatry",     r"depress|anxiet|anxious|bipolar|psychia|psychos|schizo|"
                       r"suicid|adhd|autism|ptsd|trauma-informed|substance|"
                       r"opioid|alcohol|addict|withdrawal|benzodiaz|buprenorphine|"
                       r"naltrexone|methadone|smoking|tobacco|nicotine|insomnia|"
                       r"mental health|grief|burnout|panic|phobia|ocd\b|psychotherap|sleep hygiene|cbt\b|counseling for"),
    ("Musculoskeletal", r"fracture|arthrit|joint|knee|shoulder|hip|back pain|"
                       r"spine|tendon|ligament|sprain|muscle|myalgia|gout|"
                       r"osteoarthr|rheumat|lupus|fibromyalgia|sciatica|"
                       r"rotator cuff|carpal|plantar|podiatr|foot|ankle|orthop"),
    ("Neurology",      r"neuro|headache|migraine|seizure|epilep|dementia|"
                       r"alzheim|parkinson|neuropath|tremor|vertigo|dizz|"
                       r"multiple sclerosis|guillain|myasthen|concussion|"
                       r"cognitive|delirium"),
    ("Gastrointestinal", r"gastro|bowel|colon|liver|hepatic|pancrea|biliar|"
                       r"gallbladder|esophag|stomach|ulcer|reflux|gerd|ibs\b|"
                       r"crohn|colitis|celiac|diarrhea|constipat|hemorrhoid|"
                       r"cirrhosis|ascites|paracentesis|abdominal"),
    ("Pulmonary",      r"pulmon|lung|asthma|copd|respirat|cough|dyspnea|"
                       r"bronch|apnea|smoking cessation|spirometr|oxygen|"
                       r"tuberculosis screening"),
    ("Renal",          r"renal|kidney|nephro|dialysis|creatinin|gfr|electrolyte|"
                       r"sodium|potassium|hyponatr|hyperkal|urin|bladder|prostat|"
                       r"nephrolith|stone"),
    ("Dermatology",    r"derm|skin|rash|eczema|psoriasis|acne|melanoma|"
                       r"cellulitis|wound|ulcer care|burn|lesion|mole|"
                       r"fungal skin|tinea|onychomyc|pressure injur"),
    ("Hematology/Oncology", r"cancer|oncol|tumor|malignan|chemother|leukemia|"
                       r"lymphoma|anemia|hemoglobin|bleeding disorder|"
                       r"coagul|anticoagul|warfarin|platelet|transfus|"
                       r"screening colonoscopy|mammogra"),
    ("Pediatrics",     r"pediatr|child|infant|adolescen|teen|school-age|"
                       r"growth chart|developmental|immunization schedule|"
                       r"failure to thrive|bronchiolitis|croup"),
    ("Geriatrics",     r"geriatr|elderly|older adult|falls|frailty|polypharm|"
                       r"advance directive|hospice|palliat|end of life|"
                       r"long-term care|caregiver"),
    ("Preventive",     r"screening|prevent|wellness|counsel|health maintenance|"
                       r"risk factor|lifestyle|diet|nutrition|exercise|"
                       r"uspstf|guideline"),
    ("Procedures",     r"procedure|biopsy|suture|laceration|injection|"
                       r"aspiration|incision|ultrasound|pocus|casting|splint|"
                       r"cryotherapy|excision|drainage|catheter|intubation|lidocaine|anesthe|x-?ray|radiograph|imaging|chest film"),
    ("Global Health",  r"global|mission|international|refugee|low-resource|"
                       r"tropical medicine|humanitarian|austere|missionary"),
    ("Practice",       r"billing|coding|documentation|ethic|advocacy|"
                       r"malpractice|informed consent|quality improvement|"
                       r"board review|residency|teaching|leadership|"
                       r"burnout prevention|charting|shared decision|social determinant|differential diagnosis|clinical reasoning|evidence-based|communication|health literacy|interpreter"),
]
DOMAIN_RE = [(name, re.compile(pat, re.I)) for name, pat in DOMAINS]

# Drugs were the single biggest hole: of the topics that appear in 3+ lectures and
# failed to classify, almost all were medications (Aspirin, Doxycycline, Heparin,
# GLP-1 Agonists...). Conditions were covered, drug names were not. Two rules fix it
# -- named agents/classes first, then drug-name morphology, which generalises to
# agents nobody listed.
DRUG_NAMES = [
    ("Infectious",     r"amoxicillin|penicillin|doxycycline|azithromycin|"
                       r"ceftriaxone|cephalexin|metronidazole|nitrofurantoin|"
                       r"vancomycin|clindamycin|acyclovir|fluconazole|"
                       r"bactrim|trimethoprim|sulfamethoxazole|rifampin|isoniazid"),
    ("Cardiovascular", r"aspirin|lisinopril|losartan|amlodipine|metoprolol|"
                       r"atorvastatin|simvastatin|furosemide|hydrochlorothiazide|"
                       r"clopidogrel|heparin|warfarin|apixaban|rivaroxaban|"
                       r"nitroglycerin|ace inhibitor|arb\b|beta.?blocker|spironolactone|"
                       r"statin|diuretic|calcium channel|anticoagul|antiplatelet|"
                       r"epinephrine|norepinephrine|vasopressor|digoxin"),
    ("Endocrine",      r"metformin|insulin|glipizide|semaglutide|liraglutide|"
                       r"empagliflozin|glp-?1|sglt2|levothyroxine|"
                       r"methimazole|alendronate|prednisone|steroid|"
                       r"corticosteroid|dexamethasone|hydrocortisone"),
    ("Psychiatry",     r"sertraline|fluoxetine|escitalopram|citalopram|"
                       r"bupropion|venlafaxine|duloxetine|mirtazapine|trazodone|"
                       r"lithium|quetiapine|risperidone|aripiprazole|lorazepam|"
                       r"alprazolam|clonazepam|diazepam|buprenorphine|methadone|"
                       r"naltrexone|naloxone|varenicline|ssri|snri|"
                       r"antidepressant|antipsychotic|benzodiazepine|stimulant"),
    ("Musculoskeletal", r"ibuprofen|naproxen|nsaid|acetaminophen|colchicine|"
                       r"allopurinol|febuxostat|cyclobenzaprine|methotrexate"),
    ("Neurology",      r"sumatriptan|rizatriptan|triptan|topiramate|gabapentin|"
                       r"pregabalin|levetiracetam|lamotrigine|amitriptyline|"
                       r"donepezil|carbidopa|levodopa|cgrp"),
    ("Gastrointestinal", r"omeprazole|pantoprazole|famotidine|ondansetron|"
                       r"ppi\b|proton pump|laxative|polyethylene glycol|"
                       r"loperamide|mesalamine"),
    ("Pulmonary",      r"albuterol|fluticasone|montelukast|tiotropium|"
                       r"budesonide|inhaled cortico|bronchodilator|nebuliz"),
    ("Dermatology",    r"triamcinolone|clotrimazole|terbinafine|mupirocin|"
                       r"isotretinoin|hydrocortisone cream"),
]
DRUG_RE = [(n, re.compile(p, re.I)) for n, p in DRUG_NAMES]

# Morphology fallback, for agents not named above.
DRUG_MORPH = [
    ("Infectious",     r"\w+(cillin|mycin|micin|floxacin|cycline|azole(?!pam))\b"),
    ("Cardiovascular", r"\w+(statin|pril|sartan|olol|dipine|parin|xaban)\b"),
    ("Endocrine",      r"\w+(glutide|gliflozin|gliptin|formin)\b"),
    ("Psychiatry",     r"\w+(azepam|azolam|codone|morphone|oxetine|opram)\b"),
    ("Neurology",      r"\w+(triptan|gabalin|racetam)\b"),
    ("Gastrointestinal", r"\w+(prazole|tidine|setron)\b"),
]
DRUG_MORPH_RE = [(n, re.compile(p, re.I)) for n, p in DRUG_MORPH]


def domain_of(label: str) -> str:
    for name, rx in DOMAIN_RE:
        if rx.search(label):
            return name
    for name, rx in DRUG_RE:
        if rx.search(label):
            return name
    for name, rx in DRUG_MORPH_RE:
        if rx.search(label):
            return name
    return "General"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def main() -> int:
    if not SITE.exists():
        print("run build_site.py first")
        return 1
    data = json.loads(SITE.read_text())
    topics = data["topics"]
    by_norm = {norm(t["label"]): t for t in topics}

    # Generic head nouns that must never become a parent: rolling up on them puts
    # "Statin Therapy" and "Oxygen Therapy" in one family, which is meaningless.
    GENERIC_HEADS = {
        "therapy", "screening", "referral", "management", "treatment", "care",
        "pain", "test", "testing", "evaluation", "assessment", "diagnosis",
        "prevention", "risk", "risk factors", "education", "counseling",
        "monitoring", "dosing", "use", "options", "considerations", "approach",
        "workup", "exam", "examination", "history", "criteria", "guidelines",
    }

    # ---- pass 1: parent by head-noun suffix -------------------------------
    parent = {}
    for t in topics:
        n = norm(t["label"])
        words = n.split()
        # try progressively shorter suffixes: "diabetic foot ulcer" -> "foot ulcer" -> "ulcer"
        for i in range(1, len(words)):
            cand = " ".join(words[i:])
            if cand in by_norm and cand != n and cand not in GENERIC_HEADS:
                parent[n] = cand
                break

    def root_of(n: str) -> str:
        seen = set()
        while n in parent and n not in seen:
            seen.add(n)
            n = parent[n]
        return n

    # ---- pass 2: MeSH resolution ------------------------------------------
    # Domain and hierarchy now come from MeSH tree numbers, which is reference data
    # rather than regexes I wrote. The keyword table survives only as a fallback for
    # the ~10% of concepts MeSH does not match (clinical shorthand, local coinages).
    from mesh import Mesh
    M = Mesh()
    nodes, domain_count = {}, Counter()
    matched = 0
    for t in topics:
        n = norm(t["label"])
        ui = M.lookup(t["label"])
        if ui:
            matched += 1
            dom = M.domain(ui)
            pui = M.parent(ui)
            mesh_parent = M.heading.get(pui) if pui else None
        else:
            dom, ui, mesh_parent = domain_of(t["label"]), None, None
            if dom == "General" and root_of(n) in by_norm:
                dom = domain_of(by_norm[root_of(n)]["label"])
        nodes[n] = {"label": t["label"], "slug": t["slug"], "root": root_of(n),
                    "parent": parent.get(n), "domain": dom,
                    "mesh_ui": ui, "mesh_parent": mesh_parent,
                    "episodes": t["episodes"], "n": len(t["episodes"])}
        domain_count[dom] += 1
    print(f"  MeSH matched : {matched}/{len(topics)} ({100*matched/len(topics):.0f}%)")

    # roll episode counts up to roots so parents reflect their whole subtree
    subtree = defaultdict(set)
    for n, v in nodes.items():
        subtree[v["root"]].update(v["episodes"])
    for n, v in nodes.items():
        v["subtree_episodes"] = len(subtree[v["root"]])

    roots = sorted({v["root"] for v in nodes.values()})
    OUT.write_text(json.dumps(
        {"nodes": nodes, "roots": roots,
         "domains": sorted(domain_count.items(), key=lambda kv: -kv[1])},
        indent=2, ensure_ascii=False))

    linked = sum(1 for v in nodes.values() if v["parent"])
    print(f"  topics        : {len(nodes)}")
    print(f"  with a parent : {linked} ({100*linked/len(nodes):.0f}%)")
    print(f"  distinct roots: {len(roots)}")
    print("\n  domains:")
    for d, c in sorted(domain_count.items(), key=lambda kv: -kv[1]):
        print(f"    {d:22s} {c:5d}")
    gen = 100 * domain_count['General'] / len(nodes)
    print(f"\n  unclassified (General): {gen:.0f}%")

    # biggest families, as a sanity check on the suffix linking
    fam = Counter(v["root"] for v in nodes.values())
    print("\n  largest concept families:")
    for r, c in fam.most_common(8):
        kids = [v["label"] for v in nodes.values() if v["root"] == r and v["parent"]][:3]
        print(f"    {by_norm.get(r,{}).get('label', r)[:34]:36s} {c:3d}  e.g. {kids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
