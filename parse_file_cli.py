from pathlib import Path
import argparse


def analyze_dir(input_path: Path, output_path: Path):
    from main import AnimeVideoAnalyzer

    analyzer = AnimeVideoAnalyzer()

    for video_path in input_path.glob("*.mp4"):
        outpath = output_path / video_path.stem
        analyzer.process_video(video_path, "lines-output", outpath)


def analyze(input_path, output_path):
    from main import AnimeVideoAnalyzer

    analyzer = AnimeVideoAnalyzer()
    analyzer.process_video(input_path, "lines-output", output_path)


def main():
    parser = argparse.ArgumentParser(description="Analyze anime video files.")
    parser.add_argument("-i","--input", type=str, help="Path to the input file or directory", dest="input")
    parser.add_argument("-o","--output", type=str, help="Path to the output directory", dest="output")

    args = parser.parse_args()
    if not args.input:
        print("use -i or --input to specify the input path")
        return
    if not args.output:
        print("use -o or --output to specify the output path")
        return

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: The input path '{input_path}' does not exist.")
        return

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        analyze_dir(input_path, output_path)
    else:
        analyze(input_path, output_path)
    print("Analysis completed.")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
