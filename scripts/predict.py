import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maintenance_priority import load_model, score_work_order


SAMPLE_RECORD = {
    "created_month": 1,
    "is_winter": 1,
    "complaint_type": "HEAT/HOT WATER",
    "descriptor": "ENTIRE BUILDING",
    "has_descriptor": 1,
    "borough": "BRONX",
    "location_type": "Residential Building",
    "open_data_channel_type": "PHONE",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Score a maintenance work order.")
    parser.add_argument("--input-json", help="Single work-order record as a JSON object.")
    parser.add_argument("--model", default=str(ROOT / "artifacts" / "maintenance_priority_model.json"))
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    record = json.loads(args.input_json) if args.input_json else SAMPLE_RECORD
    payload = load_model(args.model)
    result = score_work_order(record, model_payload=payload, threshold=args.threshold)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
