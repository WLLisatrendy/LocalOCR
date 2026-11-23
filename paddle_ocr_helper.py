"""PaddleOCR (via RapidOCR/ONNX Runtime) helper for LocalOCR.

Protocol: the app writes one image path per line to stdin; the helper answers
one JSON line per request: {"lines": [{"text": str, "box": [x0, y0, x1, y1]}]}
with pixel coordinates, origin top-left. On failure: {"error": str}.
The helper prints {"ready": true} once the models are loaded.
"""
import json
import logging
import statistics
import sys

# Gap (in line heights) above the normal Chinese character gap that marks a
# space between Chinese and English/digits. 0.1 inserted no false spaces on
# unspaced test text and restored most spaces on spaced text.
SPACE_GAP = 0.1


def is_han(c):
    return 0x3400 <= ord(c) <= 0x9FFF


def is_latin(c):
    return c.isascii() and c.isalnum()


def respace(text, words, height):
    """PaddleOCR drops spaces between Chinese and English; restore visible ones."""
    chars = []
    for word, x0, x1 in words:
        word = word.replace(" ", "")
        for k, c in enumerate(word):
            chars.append((c, x0 + (x1 - x0) * k / len(word), x0 + (x1 - x0) * (k + 1) / len(word)))
    if "".join(c for c, _, _ in chars) != text.replace(" ", ""):
        return text
    gaps = [chars[i][1] - chars[i - 1][2] for i in range(1, len(chars))
            if is_han(chars[i][0]) and is_han(chars[i - 1][0])]
    base = statistics.median(gaps) if gaps else 0
    out, j = "", 0
    for c in text:
        if c == " ":
            out += c
            continue
        if j > 0 and out and out[-1] != " ":
            a, b = chars[j - 1], chars[j]
            boundary = (is_han(a[0]) and is_latin(b[0])) or (is_latin(a[0]) and is_han(b[0]))
            if boundary and b[1] - a[2] > base + SPACE_GAP * height:
                out += " "
        out += c
        j += 1
    return out


def recognize(engine, path):
    result = engine(path, return_word_box=True)
    if result.boxes is None:
        return []
    lines = []
    for box, text, words in zip(result.boxes, result.txts, result.word_results):
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        height = max(ys) - min(ys)
        word_boxes = [(w[0], float(w[2][0][0]), float(w[2][1][0])) for w in words]
        lines.append({"text": respace(text, word_boxes, height),
                      "box": [min(xs), min(ys), max(xs), max(ys)]})
    return lines


def main():
    logging.disable(logging.CRITICAL)
    from rapidocr import RapidOCR
    # Screenshots are upright; the direction classifier flipped indented code
    # lines upside down in testing, so it stays off.
    engine = RapidOCR(params={"Global.use_cls": False, "Global.log_level": "critical"})
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        path = line.rstrip("\n")
        if not path:
            continue
        try:
            reply = {"lines": recognize(engine, path)}
        except Exception as error:  # report and keep serving
            reply = {"error": str(error)}
        print(json.dumps(reply, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
