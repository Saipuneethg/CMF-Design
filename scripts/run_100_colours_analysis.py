import os
import sys
import argparse
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.batch_analyzer import BatchAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Spectral Metameric Fiber Analyzer — 100 Colour Batch Analysis")
    parser.add_argument("--image", default="Dataset/image.png", help="Path to source RGB image")
    parser.add_argument("--colours", type=int, default=100, help="Target number of unique in-gamut colours")
    parser.add_argument("--samples", type=int, default=500, help="Requested metamer samples per colour")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible random seed")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel worker threads")
    parser.add_argument("--output-dir", default="random_colours_ambiguity_results", help="Base output directory")
    args = parser.parse_args()

    print("=" * 80)
    print("SPECTRAL METAMERIC FIBER ANALYZER — 100 COLOUR AMBIGUITY PIPELINE")
    print("=" * 80)
    print(f"Source Image:         {args.image}")
    print(f"Target Colours:       {args.colours}")
    print(f"Samples per Colour:   {args.samples}")
    print(f"Random Seed:          {args.seed}")
    print(f"Worker Threads:       {args.workers}")
    print(f"Base Output Folder:   {args.output_dir}")
    print("=" * 80)

    analyzer = BatchAnalyzer(artifacts_dir='artifacts', base_output_dir=args.output_dir)

    def on_progress(status):
        pct = status.get('percent_complete', 0.0)
        action = status.get('current_action', '')
        print(f"[{pct:5.1f}%] {action}")

    t0 = time.time()
    result = analyzer.run_analysis(
        source_image_path=args.image,
        target_colours=args.colours,
        n_samples=args.samples,
        seed=args.seed,
        max_workers=args.workers,
        progress_cb=on_progress
    )
    t1 = time.time()

    print("=" * 80)
    print(f"PIPELINE COMPLETE in {t1 - t0:.1f} seconds!")
    print(f"Run ID:            {result['run_id']}")
    print(f"Directory:         {result['run_dir']}")
    print(f"Total Completed:   {result['total_completed']} colours")
    print(f"ZIP Archive:       {result['zip_path']}")
    print("=" * 80)

if __name__ == '__main__':
    main()
