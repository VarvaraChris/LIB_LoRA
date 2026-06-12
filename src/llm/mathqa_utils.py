import re


_MATHQA_OPTION_RE = re.compile(
    r"([a-eA-E])\s*\)\s*(.*?)(?=\s*,\s*[a-eA-E]\s*\)|$)",
    re.DOTALL,
)


def _clean_text(text):
    return str(text).replace("\n", " ").replace("\\n", " ").strip()


def parse_mathqa_choices(options):
    parsed = []
    for label, text in _MATHQA_OPTION_RE.findall(str(options)):
        parsed.append((label.upper(), _clean_text(text)))
    return parsed


def get_mathqa_choice_texts(options):
    return [text for _, text in parse_mathqa_choices(options)]


def get_mathqa_correct_choice(options, correct_label):
    correct_label = str(correct_label).strip().upper()
    for label, text in parse_mathqa_choices(options):
        if label == correct_label:
            return text
    return None


def build_mathqa_prompt(problem, options):
    lines = [f"Question: {_clean_text(problem)}", "Answer Choices:"]
    for label, text in parse_mathqa_choices(options):
        lines.append(f"({label}) {text}")
    lines.append("Answer:")
    return "\n".join(lines)