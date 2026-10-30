from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Speaker:
    name: str
    gender: str
    role: str
    aliases: tuple[str, ...]
    verbs: tuple[str, ...]
    concerns: tuple[str, ...]


@dataclass(frozen=True)
class StorySpec:
    directory: Path
    slug: str
    title: str
    target_word_count: int
    difficulty: str
    setting: str
    central_object: str
    stakes: str
    places: tuple[str, ...]
    motifs: tuple[str, ...]
    chapter_titles: tuple[str, ...]
    speakers: tuple[Speaker, ...]
    unknown_speakers: dict[str, str]
    notes: tuple[str, ...]


SPECS = (
    StorySpec(
        directory=Path("data/story_2"),
        slug="ciphered_tide",
        title="The Ciphered Tide",
        target_word_count=20000,
        difficulty="direct maritime espionage with denser speaker traffic than story_1",
        setting="the fogbound free harbor of Caul Reach",
        central_object="a tide chart whose printed shoals hide a naval cipher",
        stakes="a blockade fleet will fire on refugee ships if the false channel is accepted",
        places=(
            "the customs chainhouse",
            "the rusted buoy yard",
            "Pier Twelve beneath the whale-bone crane",
            "the flooded chart room",
            "the salvage bell",
            "the lantern barge",
            "the quarantine lock",
        ),
        motifs=(
            "green tide marks",
            "salt on official paper",
            "a gull call repeated at the wrong interval",
            "lamps blinking in groups of five",
            "ink that darkened when warmed",
            "ropes tied in courtroom knots",
        ),
        chapter_titles=(
            "Fog on the customs chain",
            "The chart that rose with the tide",
            "Signals from a ship declared sunk",
            "The channel under the quarantine lock",
            "A ledger written in brine",
            "The buoy that answered twice",
            "Night pilotage",
            "The harbor chooses a witness",
        ),
        speakers=(
            Speaker(
                "Lena Marr",
                "female",
                "harbor pilot who hides intelligence in tide charts",
                ("Lena", "Pilot Marr", "the pilot", "Marr"),
                ("said", "warned", "answered", "cut in"),
                ("the true depth", "the refugee convoy", "the channel marks", "Rook's old code"),
            ),
            Speaker(
                "Jon Vale",
                "male",
                "customs inspector who speaks carefully and asks leading questions",
                ("Jon", "Inspector Vale", "the inspector", "Vale"),
                ("asked", "said", "observed", "pressed"),
                ("seized manifests", "legal custody", "the false warrant", "which witness lied first"),
            ),
            Speaker(
                "Priya Sen",
                "female",
                "radio operator who interrupts with concise technical warnings",
                ("Priya", "Sen", "the operator", "Radio Sen"),
                ("interrupted", "reported", "said", "snapped"),
                ("carrier drift", "jammed channels", "lamp sequences", "the signal lag"),
            ),
            Speaker(
                "Oren Pike",
                "male",
                "salvage diver with blunt practical phrasing",
                ("Oren", "Pike", "the diver", "Salvage Pike"),
                ("said", "muttered", "grunted", "replied"),
                ("air hoses", "locked hatches", "dead ballast", "the weight of wet rope"),
            ),
            Speaker(
                "Marta Quill",
                "female",
                "quartermaster who uses dry humor",
                ("Marta", "Quill", "the quartermaster", "Quartermaster Quill"),
                ("said", "remarked", "answered", "drawled"),
                ("missing stores", "honest thieves", "bad coffee", "the ship's impossible accounts"),
            ),
            Speaker(
                "Elias Rook",
                "male",
                "exiled code clerk who evades direct answers",
                ("Elias", "Rook", "the clerk", "the exile"),
                ("said", "deflected", "answered", "whispered"),
                ("old keys", "burned registries", "names he no longer used", "the cipher's missing tide"),
            ),
        ),
        unknown_speakers={
            "narrator": "third-person narrator",
            "harbor loudspeaker": "automated port announcements",
        },
        notes=(
            "Story 2 is harder than story_1 through six recurring human speakers, denser dialogue, and mild attribution distractors.",
            "Speaker cues remain mostly direct, but narration often names several characters before the row speaker talks.",
            "Unknown rows use automated harbor announcements and signal notices.",
        ),
    ),
    StorySpec(
        directory=Path("data/story_3"),
        slug="ashen_lantern",
        title="The Ashen Lantern",
        target_word_count=20000,
        difficulty="alias-heavy mountain monastery mystery",
        setting="Irval Monastery above the black ravine",
        central_object="an ashen lantern that burns without wick or oil",
        stakes="the wrong confession will seal the monastery before the avalanche season",
        places=(
            "the west stair",
            "the scriptorium",
            "the bell tower",
            "the herb infirmary",
            "the kitchen court",
            "the pilgrim dormitory",
            "the frozen reliquary",
        ),
        motifs=(
            "gray flame",
            "snow inside a locked sill",
            "a bell toll counted by different hands",
            "gold leaf stuck to a prayer",
            "footprints ending at a wall",
            "ash in a cup of melted ice",
        ),
        chapter_titles=(
            "Snow at the west stair",
            "The Lamp Hand is named",
            "Brother Ash counts symptoms",
            "A courier drops a borrowed name",
            "Bell rope, pilgrim mask",
            "The recipe for a rumor",
            "Judgment before dawn",
            "The lantern refuses its owner",
        ),
        speakers=(
            Speaker(
                "Nima Darel",
                "female",
                "apprentice illuminator, also called the Lamp Hand",
                ("Nima", "Lamp Hand", "the illuminator", "Darel"),
                ("said", "insisted", "asked", "broke in"),
                ("the burning relic", "the stolen gold leaf", "which prayer was copied twice", "Amaya's mask"),
            ),
            Speaker(
                "Tovan Aric",
                "male",
                "physician-monk, also called Brother Ash",
                ("Tovan", "Brother Ash", "the physician", "Aric"),
                ("said", "replied", "diagnosed", "murmured"),
                ("symptoms", "ash stains", "the prior's fever", "whether guilt has a pulse"),
            ),
            Speaker(
                "Ysolde Fen",
                "female",
                "bell keeper who speaks in clipped observations",
                ("Ysolde", "the bell keeper", "Fen", "keeper"),
                ("said", "reported", "cut in", "answered"),
                ("wrong hours", "rope burns", "the tower wind", "the bell's second voice"),
            ),
            Speaker(
                "Garron Vey",
                "male",
                "courier who hides panic behind jokes",
                ("Garron", "Vey", "the courier", "road boy"),
                ("said", "joked", "protested", "asked"),
                ("closed roads", "borrowed seals", "the mule he abandoned", "letters he never read"),
            ),
            Speaker(
                "Savitri Morn",
                "female",
                "visiting magistrate who asks formal questions",
                ("Savitri", "Magistrate Morn", "the magistrate", "Morn"),
                ("said", "asked", "recorded", "ruled"),
                ("lawful custody", "confession order", "the name on the writ", "the difference between fear and evidence"),
            ),
            Speaker(
                "Kell Rusk",
                "male",
                "monastery cook who repeats rumors as if they were recipes",
                ("Kell", "Rusk", "the cook", "Cook Rusk"),
                ("said", "recited", "offered", "whispered"),
                ("rumors", "stew pots", "who stole salt", "the recipe for silence"),
            ),
            Speaker(
                "Amaya Sol",
                "female",
                "masked pilgrim using the aliases Mara-of-the-Stair and Liora Venn",
                ("Amaya", "Mara-of-the-Stair", "Liora Venn", "the masked pilgrim"),
                ("said", "answered", "admitted", "translated"),
                ("borrowed names", "old pilgrim vows", "the locked reliquary", "why her mask has no mouth"),
            ),
        ),
        unknown_speakers={
            "narrator": "third-person narrator",
            "monastery bell": "ritual bell inscriptions and tolling messages",
        },
        notes=(
            "Story 3 is harder than story_2 through aliases, titles, interruptions, reported speech, and group scenes.",
            "Human speakers recur under names, offices, and relational address terms.",
            "Unknown rows use bell inscriptions and tolling messages.",
        ),
    ),
    StorySpec(
        directory=Path("data/story_4"),
        slug="mirrored_depositions",
        title="The Mirrored Depositions",
        target_word_count=20000,
        difficulty="nested legal archive with quoted testimony and document frames",
        setting="the mirrored court archive beneath Lethen Hall",
        central_object="a deposition book that reflects different testimony depending on who reads it",
        stakes="a wrongful sentence will become permanent when the court automaton seals the record",
        places=(
            "the lower evidence vault",
            "Courtroom Six",
            "the mirror-maker's gallery",
            "the clerk's marginal desk",
            "the archive lift",
            "the sealed witness alcove",
            "the judge's cold antechamber",
        ),
        motifs=(
            "silver dust on black gloves",
            "two signatures occupying one line",
            "a mirror turned to the wall",
            "ink reversed in candlelight",
            "pages that clicked like teeth",
            "a docket number scratched into glass",
        ),
        chapter_titles=(
            "Exhibit A enters backward",
            "Marginal correction in another hand",
            "The witness quotes a letter",
            "A judge hears the mirror answer",
            "Bailiff's inventory of impossible items",
            "The defendant remembers fragments",
            "Archivist against clerk",
            "Final seal before reversal",
        ),
        speakers=(
            Speaker(
                "Cass Mirel",
                "female",
                "junior advocate annotating a trial record",
                ("Cass", "Advocate Mirel", "the junior advocate", "Mirel"),
                ("wrote", "said", "objected", "annotated"),
                ("the order of testimony", "who authored the note", "whether a mirror can be sworn", "Dalia's letter"),
            ),
            Speaker(
                "Bram Orison",
                "male",
                "defendant who answers with fragments",
                ("Bram", "Orison", "the defendant", "Prisoner Orison"),
                ("said", "answered", "fragmented", "recalled"),
                ("a door without hinges", "the face in the glass", "his missing hour", "a name spoken behind him"),
            ),
            Speaker(
                "Helia Voss",
                "female",
                "senior judge with ceremonial phrasing",
                ("Helia", "Judge Voss", "the senior judge", "Voss"),
                ("said", "ruled", "intoned", "ordered"),
                ("court sequence", "formal silence", "the dignity of evidence", "which version may stand"),
            ),
            Speaker(
                "Ren Sable",
                "male",
                "clerk who writes marginal corrections",
                ("Ren", "Sable", "the clerk", "Clerk Sable"),
                ("wrote", "said", "corrected", "noted"),
                ("page order", "witness initials", "erasures", "the automaton's timestamp"),
            ),
            Speaker(
                "Dalia Kerr",
                "female",
                "witness whose letters are quoted inside testimony",
                ("Dalia", "Kerr", "the witness", "Mistress Kerr"),
                ("said", "wrote", "testified", "repeated"),
                ("letters", "the mirror shipment", "a stranger's reflection", "why she signed twice"),
            ),
            Speaker(
                "Mikol Thane",
                "male",
                "bailiff who remembers logistical details",
                ("Mikol", "Thane", "the bailiff", "Bailiff Thane"),
                ("said", "listed", "reported", "answered"),
                ("keys", "chairs", "sealed doors", "who crossed the threshold"),
            ),
            Speaker(
                "Iva Pell",
                "female",
                "archivist who disputes document order",
                ("Iva", "Archivist Pell", "the archivist", "Pell"),
                ("said", "disputed", "cataloged", "replied"),
                ("provenance", "stacks", "false sequence", "how documents lie by arrangement"),
            ),
            Speaker(
                "Sorren Aft",
                "male",
                "mirror-maker expert witness",
                ("Sorren", "Aft", "the mirror-maker", "Master Aft"),
                ("said", "explained", "admitted", "measured"),
                ("silver backing", "warped glass", "reflected speech", "what a mirror refuses"),
            ),
        ),
        unknown_speakers={
            "narrator": "third-person frame narrator",
            "court automaton": "institutional transcript voice",
        },
        notes=(
            "Story 4 is harder than story_3 through nested documents, quoted testimony, letters, marginal notes, and court transcript frames.",
            "The row speaker is often different from the person discussed or quoted inside the paragraph.",
            "Unknown rows use institutional court automaton excerpts.",
        ),
    ),
    StorySpec(
        directory=Path("data/story_5"),
        slug="redacted_palace",
        title="The Redacted Palace",
        target_word_count=20000,
        difficulty="unreliable palace inquest with redactions, mimicry, and unknown voices",
        setting="the Redacted Palace during a ceremonial lockdown",
        central_object="a royal inquest record rewritten while witnesses are still speaking",
        stakes="the palace will choose a prince, a corpse, and a culprit by editing the same hour three ways",
        places=(
            "the Hall of Petitions",
            "the blue censor's office",
            "the singer's screened balcony",
            "the locked etiquette school",
            "the physician's tiled chamber",
            "the night garden vents",
            "the ambassador's proof room",
        ),
        motifs=(
            "[REDACTED]",
            "a signet worn by the wrong hand",
            "voices traveling through pipes",
            "a song copied in another throat",
            "black cordial on white marble",
            "titles changing at each bell",
        ),
        chapter_titles=(
            "The first inquest",
            "Titles temporarily removed",
            "A singer borrows the captain's voice",
            "The double refuses his own portrait",
            "Medical correction after censorship",
            "The garden vents repeat a toast",
            "Proof of a living corpse",
            "The palace edits the hour",
        ),
        speakers=(
            Speaker(
                "Aria Venn",
                "female",
                "palace historian who revises her own account",
                ("Aria", "Historian Venn", "the historian", "Venn"),
                ("said", "revised", "dictated", "crossed out"),
                ("official memory", "footnotes", "which hour was struck", "obedience with evidence"),
            ),
            Speaker(
                "Malek Dorn",
                "male",
                "captain of keys who hides orders inside etiquette",
                ("Malek", "Captain Dorn", "the captain of keys", "Dorn"),
                ("said", "instructed", "bowed", "answered"),
                ("lock order", "court manners", "servant routes", "the west locks never opening"),
            ),
            Speaker(
                "Selene Io",
                "female",
                "singer who can mimic other voices",
                ("Selene", "Io", "the singer", "the mimic"),
                ("said", "sang", "imitated", "answered in another voice"),
                ("borrowed voices", "balcony echoes", "who heard the prince", "the song that changed witnesses"),
            ),
            Speaker(
                "Tarek Moul",
                "male",
                "poison taster with literal phrasing",
                ("Tarek", "Moul", "the taster", "Poison Moul"),
                ("said", "reported", "tasted", "answered"),
                ("cordials", "dose logic", "which bitterness chose whom", "the etiquette of murder"),
            ),
            Speaker(
                "Vesper Linn",
                "female",
                "redaction officer also called the Blue Censor",
                ("Vesper", "Blue Censor", "Officer Linn", "Linn"),
                ("said", "redacted", "declared", "corrected"),
                ("blue pencil", "forbidden titles", "administrative panic", "how much imagination is treason"),
            ),
            Speaker(
                "Nolan Crest",
                "male",
                "prince's double using several titles",
                ("Nolan", "Crest", "the double", "Lantern Guest"),
                ("said", "claimed", "denied", "answered"),
                ("borrowed rank", "the prince's portrait", "which hand wore the signet", "being alive in the wrong office"),
            ),
            Speaker(
                "Rhea Sarn",
                "female",
                "physician who contradicts official records",
                ("Rhea", "Doctor Sarn", "the physician", "Sarn"),
                ("said", "corrected", "diagnosed", "contradicted"),
                ("pulse records", "cooling skin", "medical lies", "why the body moved after midnight"),
            ),
            Speaker(
                "Omar Jett",
                "male",
                "night gardener who overhears from vents",
                ("Omar", "Jett", "the gardener", "Night Jett"),
                ("said", "whispered", "reported", "recalled"),
                ("vents", "garden keys", "voices in irrigation pipes", "what roses heard"),
            ),
            Speaker(
                "Livia Quen",
                "female",
                "mathematician ambassador who speaks in proofs",
                ("Livia", "Ambassador Quen", "the mathematician", "Quen"),
                ("said", "proved", "calculated", "answered"),
                ("contradictory premises", "three possible princes", "proof by locked door", "the sum of missing witnesses"),
            ),
        ),
        unknown_speakers={
            "narrator": "unreliable third-person narrator",
            "palace intercom": "institutional announcements",
            "unsigned marginalia": "unattributed notes in the palace record",
        },
        notes=(
            "Story 5 is the hardest fixture through redactions, unreliable narration, voice mimicry, secret titles, unknown voices, and frequent speaker switches.",
            "Many rows quote or imitate a voice different from the row speaker.",
            "Unknown rows include institutional announcements, unsigned marginalia, and intercepted whispers copied into the palace record.",
        ),
    ),
)


def main() -> int:
    for spec in SPECS:
        build_fixture(spec)
    return 0


def build_fixture(spec: StorySpec) -> None:
    spec.directory.mkdir(parents=True, exist_ok=True)
    story_path = spec.directory / f"{spec.slug}_20000_word_story.txt"
    jsonl_path = spec.directory / f"{spec.slug}_speaker_attribution_ground_truth.jsonl"
    metadata_path = spec.directory / f"{spec.slug}_benchmark_metadata.json"

    rows = load_existing_rows(jsonl_path, spec.slug)
    if not rows:
        rows.extend(opening_rows(spec))

    generated_index = 0
    while word_count(join_story_text(rows)) < spec.target_word_count:
        generated_index += 1
        row_number = len(rows) + 1
        rows.append(make_row(spec, row_number, generated_index))

    normalize_ids(rows, spec.slug)
    story_text = join_story_text(rows)
    metadata = make_metadata(spec, rows, story_text)

    story_path.write_text(story_text + "\n", encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def load_existing_rows(path: Path, slug: str) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "speaker": str(row["speaker"]),
                "expected_gender": str(row["expected_gender"]),
                "text": str(row["text"]).strip(),
            }
        )
    for index, row in enumerate(rows, start=1):
        expected = f"{slug}_{index:04d}"
        if row["sample_id"] != expected:
            raise ValueError(f"{path}:{index}: expected {expected}, got {row['sample_id']}")
    return rows


def opening_rows(spec: StorySpec) -> list[dict[str, str]]:
    first, second, third = spec.speakers[0], spec.speakers[1], spec.speakers[2]
    return [
        row(
            "",
            "narrator",
            "unknown",
            (
                f"Chapter 1: {spec.chapter_titles[0]}. In {spec.setting}, the first account began with "
                f"{spec.central_object}. {first.name}, {second.name}, and {third.name} each noticed a different "
                f"part of the same warning, which made every later statement sound like an answer to the wrong "
                f"question. The problem was not simply danger; the problem was that {spec.stakes}."
            ),
        ),
        row(
            "",
            first.name,
            first.gender,
            (
                f"\"Call me {first.aliases[1]} if titles help you keep courage,\" {first.name} {first.verbs[0]}. "
                f"\"I am looking at {spec.motifs[0]}, not at the face everyone keeps turning toward. "
                f"{second.aliases[1]} may ask who touched the evidence, and {third.aliases[1]} may ask when the "
                f"warning arrived, but I want to know why the room behaved as if it had already heard us confess.\""
            ),
        ),
        row(
            "",
            second.name,
            second.gender,
            (
                f"\"Then answer in order,\" {second.name} {second.verbs[0]}, though {third.name} was already "
                f"checking {spec.motifs[1]} and {first.name} had moved toward {spec.places[0]}. \"Who had custody "
                f"of {spec.central_object}, who benefited from losing it, and who is pretending those are the same "
                f"question because separate answers would inconvenience them?\""
            ),
        ),
    ]


def make_row(spec: StorySpec, row_number: int, generated_index: int) -> dict[str, str]:
    chapter = ((row_number - 1) // 28) % len(spec.chapter_titles)
    beat = spec.chapter_titles[chapter]
    place = pick(spec.places, row_number)
    motif = pick(spec.motifs, row_number + generated_index)

    if row_number % 17 == 0:
        return make_unknown_row(spec, row_number, beat, place, motif)
    if row_number % 5 == 0:
        return make_narrator_row(spec, row_number, generated_index, beat, place, motif)
    return make_speaker_row(spec, row_number, generated_index, beat, place, motif)


def make_speaker_row(
    spec: StorySpec,
    row_number: int,
    generated_index: int,
    beat: str,
    place: str,
    motif: str,
) -> dict[str, str]:
    speaker = pick(spec.speakers, row_number + generated_index)
    other_a = pick_other(spec.speakers, speaker.name, row_number + 1)
    other_b = pick_other(spec.speakers, speaker.name, row_number + 3)
    other_c = pick_other(spec.speakers, speaker.name, row_number + 5)
    alias = pick(speaker.aliases, row_number)
    verb = pick(speaker.verbs, generated_index)
    concern = pick(speaker.concerns, row_number + generated_index)

    if spec.slug == "ciphered_tide":
        text = (
            f"\"{other_a.aliases[0]} is watching {other_b.aliases[1]}, and {other_c.aliases[0]} is watching the "
            f"door, so I will watch {motif},\" {speaker.name} {verb} at {place}. \"That is the only witness in "
            f"this harbor that has not learned to flatter uniforms. If {concern} changes with the tide, then the "
            f"safe channel was never safe; it was an invitation written for ships that trusted official ink.\" "
            f"The remark sounded direct until {other_a.name} repeated it, because then everyone heard the warning "
            f"inside the joke and the accusation inside the warning."
        )
    elif spec.slug == "ashen_lantern":
        text = (
            f"\"Do not write that as {alias}'s confession,\" {speaker.name} {verb}, while {other_a.aliases[1]} "
            f"examined {motif} and {other_b.aliases[0]} repeated what {other_c.aliases[1]} had supposedly heard "
            f"near {place}. \"I said {concern} mattered; I did not say it belonged to me. In this monastery a title "
            f"can be borrowed faster than a candle, and a borrowed title casts the wrong shadow when the lantern "
            f"turns gray.\" The interruption forced Savitri's pen to stop halfway through a name."
        )
    elif spec.slug == "mirrored_depositions":
        text = (
            f"Deposition continuation, {beat}: \"The quoted sentence is not mine simply because it appears in my "
            f"paragraph,\" {speaker.name} {verb}, addressing {other_a.aliases[1]} while {other_b.aliases[0]}'s "
            f"letter lay open beside {motif}. \"Record that {concern} was introduced after {other_c.aliases[1]} "
            f"asked about {place}. If the mirror repeats a witness, the court must still decide whether it repeated "
            f"memory, fear, or the clerk who stood behind the witness with a corrected page.\" In the margin Ren "
            f"Sable drew two arrows pointing at different speakers."
        )
    else:
        text = (
            f"\"If this line is later attributed to {other_a.aliases[1]}, restore my name and redact the adjective,\" "
            f"{speaker.name} {verb} in {place}, after {other_b.aliases[0]} had quoted {other_c.aliases[1]} in a voice "
            f"that sounded almost correct. \"I know the palace prefers {motif} to testimony, but {concern} cannot be "
            f"solved by calling the corpse a guest or the guest a draft. [REDACTED] is not an answer; it is a door "
            f"with a clerk posted in front of it.\" The narrator later moved this paragraph twice."
        )

    return row("", speaker.name, speaker.gender, text)


def make_narrator_row(
    spec: StorySpec,
    row_number: int,
    generated_index: int,
    beat: str,
    place: str,
    motif: str,
) -> dict[str, str]:
    speaker_a = pick(spec.speakers, row_number)
    speaker_b = pick_other(spec.speakers, speaker_a.name, row_number + 2)
    speaker_c = pick_other(spec.speakers, speaker_a.name, row_number + 4)
    chapter_line = f"Chapter {row_number // 28 + 1}: {beat}. " if row_number % 28 == 0 else ""
    if spec.slug in {"mirrored_depositions", "redacted_palace"}:
        frame = (
            f"{chapter_line}The record at {place} named {speaker_a.name}, quoted {speaker_b.name}, and accused "
            f"{speaker_c.name}, but none of those operations identified the paragraph's speaker. {motif.capitalize()} "
            f"sat between the lines like a second witness. By then the inquiry had learned that order could be a "
            f"weapon: put a denial after a confession and it sounded weak; put the same denial before the confession "
            f"and it sounded like prophecy. That is why the next voice mattered more than the next fact."
        )
    else:
        frame = (
            f"{chapter_line}At {place}, {speaker_a.name}, {speaker_b.name}, and {speaker_c.name} reached the same "
            f"evidence from three different angles. {motif.capitalize()} made the room feel staged, though no one "
            f"could agree who had arranged it. The danger in the moment was attribution: a warning overheard by "
            f"one person, repeated by a second, and denied by a third could become three different truths before "
            f"anyone had lied on purpose."
        )
    return row("", "narrator", "unknown", frame)


def make_unknown_row(
    spec: StorySpec,
    row_number: int,
    beat: str,
    place: str,
    motif: str,
) -> dict[str, str]:
    unknown_names = [name for name in spec.unknown_speakers if name != "narrator"]
    speaker = pick(unknown_names, row_number)
    if spec.slug == "ciphered_tide":
        text = (
            f"Harbor notice {row_number:04d}: all pilots are reminded that {place} is closed by order of the tide "
            f"office. Lamps showing {motif} are to be treated as maintenance signals unless they repeat after the "
            f"third bell. Do not answer unofficial code groups. Do not ask why a maintenance signal knows your name."
        )
    elif spec.slug == "ashen_lantern":
        text = (
            f"Bell inscription, misheard during {beat}: one toll for witness, two for keeper, three for the name "
            f"borrowed at {place}. When {motif} appears on the bronze lip, the hearer must count the silence after "
            f"the sound, not the sound itself. A title spoken in fear returns without its owner."
        )
    elif spec.slug == "mirrored_depositions":
        text = (
            f"Court automaton entry {row_number:04d}: transcript segment opened at {place}; speaker field contested; "
            f"quoted material contains {motif}. The machine recognizes oath, denial, exhibit, and interruption, but "
            f"does not recognize remorse. Marginal voices are preserved in sequence, although sequence is under appeal."
        )
    else:
        text = (
            f"Unsigned palace notice {row_number:04d}: the phrase {motif} is provisionally [REDACTED] unless sung, "
            f"whispered through vents near {place}, or attributed to a person whose title changed after midnight. "
            f"Any witness hearing their own voice from an empty room must report the room, not the voice."
        )
    return row("", speaker, "unknown", text)


def make_metadata(spec: StorySpec, rows: list[dict[str, str]], story_text: str) -> dict[str, object]:
    labels: dict[str, dict[str, str]] = {
        speaker.name: {"expected_gender": speaker.gender, "role": speaker.role}
        for speaker in spec.speakers
    }
    for name, role_text in spec.unknown_speakers.items():
        labels[name] = {"expected_gender": "unknown", "role": role_text}

    for row_data in rows:
        labels.setdefault(
            row_data["speaker"],
            {"expected_gender": row_data["expected_gender"], "role": "speaker preserved from generated fixture"},
        )

    return {
        "title": spec.title,
        "target_word_count": spec.target_word_count,
        "actual_word_count": word_count(story_text),
        "sample_count": len(rows),
        "labels": labels,
        "notes": [
            *spec.notes,
            f"Generated as a full fixture for the progressive difficulty ladder: {spec.difficulty}.",
            "Story text is derived from the ordered JSONL text values separated by blank lines.",
        ],
    }


def row(sample_id: str, speaker: str, expected_gender: str, text: str) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "speaker": speaker,
        "expected_gender": expected_gender,
        "text": " ".join(text.split()),
    }


def normalize_ids(rows: list[dict[str, str]], slug: str) -> None:
    for index, row_data in enumerate(rows, start=1):
        row_data["sample_id"] = f"{slug}_{index:04d}"


def join_story_text(rows: list[dict[str, str]]) -> str:
    return "\n\n".join(row_data["text"].strip() for row_data in rows)


def word_count(text: str) -> int:
    return len(text.split())


def pick(items, index: int):
    return items[index % len(items)]


def pick_other(speakers: tuple[Speaker, ...], current_name: str, index: int) -> Speaker:
    candidates = [speaker for speaker in speakers if speaker.name != current_name]
    return candidates[index % len(candidates)]


if __name__ == "__main__":
    raise SystemExit(main())
