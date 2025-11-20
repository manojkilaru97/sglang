import argparse

from sglang.cli.generate import generate
from sglang.cli.serve import serve


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch the SGLang server.",
        add_help=False,  # Defer help to the specific parser
    )
    def _serve_entry(args, extra_argv):
        from sglang.cli.serve import serve as _serve
        return _serve(args, extra_argv)
    serve_parser.set_defaults(func=_serve_entry)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Run inference on a multimodal model.",
        add_help=False,  # Defer help to the specific parser
    )
    def _generate_entry(args, extra_argv):
        from sglang.cli.generate import generate as _generate
        return _generate(args, extra_argv)
    generate_parser.set_defaults(func=_generate_entry)

    args, extra_argv = parser.parse_known_args()
    args.func(args, extra_argv)
