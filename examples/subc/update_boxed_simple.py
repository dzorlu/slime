#!/usr/bin/env python3
import argparse
import json
import sys

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def main():
    parser = argparse.ArgumentParser(description="Append boxed-answer instruction to each prompt (simple).")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL path")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL path")
    args = parser.parse_args()

    total = 0
    changed = 0

    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                fout.write(line + "\n")
                continue

            prompt = rec.get("prompt")
            if isinstance(prompt, list) and prompt:
                # find last user message
                idx = None
                for i in range(len(prompt) - 1, -1, -1):
                    if isinstance(prompt[i], dict) and prompt[i].get("role") == "user":
                        idx = i
                        break
                if idx is None:
                    idx = 0
                msg = prompt[idx]
                content = msg.get("content")
                if isinstance(content, str):
                    if INSTRUCTION not in content:
                        msg["content"] = content.rstrip() + "\n\n" + INSTRUCTION
                        prompt[idx] = msg
                        rec["prompt"] = prompt
                        changed += 1
                else:
                    # leave non-string content unchanged (simple script)
                    pass

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Processed {total}, updated {changed}", file=sys.stderr)


if __name__ == "__main__":
    main()


