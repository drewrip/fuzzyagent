import sys
import argparse
import importlib
import importlib.machinery
import types
import inspect
import atheris

sig_types = []

@atheris.instrument_func
def consume_type(fdp, t: type):
    if t is float:
        return fdp.ConsumeFloat()
    else:
        return fdp.ConsumeInt(4)


@atheris.instrument_func
def get_input(data: bytes):
    """Create an input of the right type from the data"""
    fdp = atheris.FuzzedDataProvider(data)
    dl = [consume_type(fdp, t) for t in sig_types]
    return tuple(dl)

def main() -> int:

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--function-name")
    parser.add_argument("-r", "--runs", default=10000)
    parser.add_argument("-m", "--max-len", default=1024)
    parser.add_argument("-c", "--corpus")
    parser.add_argument("targets", nargs="*")
    args = parser.parse_args()

    if len(args.targets) != 1:
        print("[FUZZY_WRAPPER ERROR] you must provide the path to one target")
        sys.exit(1)

    loader = importlib.machinery.SourceFileLoader("target_module", args.targets[0])
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    all_func_names = [name for name in dir(mod) if name[:2] != "__"]
    if len(all_func_names) < 1:
        print("[FUZZY_WRAPPER ERROR] the target script doesn't contain any function definitions")
        sys.exit(1)
    target_func = getattr(mod, all_func_names[0]) if args.function_name is None else getattr(mod, args.function_name)
    atheris.instrument_func(target_func)

    sig = inspect.signature(target_func)
    global sig_types
    sig_types = [param[1].annotation if param[1].annotation is not inspect.Parameter.empty else int for param in sig.parameters.items()]

    @atheris.instrument_func
    def TestOneInput(data: bytes) -> None:
        inputs = get_input(data)
        print(inputs)
        target_func(*inputs)

    fuzzer_args = [f"-atheris_runs={args.runs}", f"-max_len={args.max_len}"]
    if args.corpus is not None:
        fuzzer_args.append(args.corpus)

    atheris.Setup(fuzzer_args, TestOneInput)
    atheris.Fuzz()

    return 0

if __name__ == "__main__":
    sys.exit(main())
