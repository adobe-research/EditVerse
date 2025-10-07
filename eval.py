# This file is under Adobe Research License. Copyright 2025 Adobe Inc.

import argparse
from automatic_evaluation import EvaluatorWrapper

def main(args):
    print("Initializing Evaluator...")
    evaluator = EvaluatorWrapper(
        metrics=args.metrics,
        test_json_path=args.test_json_path,
        gpt_api_key=args.gpt_api_key
    )

    print(f"Starting evaluation for results in: {args.generate_results_dir}")
    print(f"Metrics to be used: {args.metrics}")
    
    evaluator.evaluate(
        generate_results_dir=args.generate_results_dir,
        output_csv=args.output_csv
    )
    
    print(f"✅ Evaluation complete. Results have been saved to {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run automatic video evaluation with specified metrics and paths.")

    parser.add_argument(
        "--metrics",
        type=str,
        default="all", # "clip_temporal_consistency,dino_temporal_consistency,frame_text_alignment,video_text_alignment,pick_score_video_quality,editing_vlm_evaluation"
        help="Comma-separated string of metrics to use for evaluation."
    )
    parser.add_argument(
        "--test_json_path",
        type=str,
        default="EditVerseBench/EditVerseBench/test.json",
        help="Path to the test JSON file."
    )
    parser.add_argument(
        "--generate_results_dir",
        type=str,
        default="EditVerseBench/EditVerse_Comparison_Results/EditVerse",
        help="Directory containing the generated video results."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="EditVerse_eval.csv",
        help="Name for the output CSV file."
    )
    parser.add_argument(
        "--gpt_api_key",
        type=str,
        required=True,
        help="Your GPT API key. It is required for evaluations."
    )

    args = parser.parse_args()
    
    main(args)
