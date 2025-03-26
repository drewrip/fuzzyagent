import sys
import os
import argparse
import importlib
import importlib.machinery
import types
import inspect
import subprocess
import hashlib
from typing import List, Optional, Tuple
from dataclasses import dataclass
import z3
from langchain_community.llms import VLLMOpenAI
from langchain_core.prompts import PromptTemplate
import duckdb


RE_CODE_POINTS = r'# (\[[a-zA-Z0-9\-\_]+\])'
RE_IDENTIFIER = r'([^\d\W]\w*)'
RE_JSON_BLOCK = r'```(json)?([^(```)]*)```'

def type_to_sql(t: type) -> str:
    if t is float:
        return "DOUBLE"
    else:
        return "INTEGER"

def insert_guesses(con, arity, guesses: List[Tuple]):
    template = ", ".join(["?" for _ in range(arity)])
    try:
        con.executemany(f"INSERT INTO guesses VALUES ({template})", guesses)
    except:
        pass

def save_guesses(guesses: List[Tuple], corpus_dir="/tmp/corpus"):
    try:
        for guess in guesses:
            h = hashlib.sha256(bytes(guess)).hexdigest()
            with open(os.path.join(corpus_dir, h), "wb") as f:
                f.write(bytes(guess))
    except:
        pass

def topn(con, v, n=2):
    results = []
    try:
        results = con.execute(f"SELECT guesses.* FROM guesses, (SELECT {v}, COUNT({v}) as occurences FROM guesses GROUP BY {v}) counts WHERE guesses.{v} = counts.{v} ORDER BY guesses.{v} DESC LIMIT {n}").fetchall()
    except:
        pass
    return results

@dataclass
class CmdResult:
    rc: int
    stdout: Optional[str]
    stderr: Optional[str]

def run(cmd: str, ignore_output=False, out_file=sys.stdout, err_file=sys.stderr) -> CmdResult:
    ran = subprocess.run(cmd, shell=True, stdout=(subprocess.DEVNULL if ignore_output else subprocess.PIPE), stderr=(subprocess.DEVNULL if ignore_output else subprocess.PIPE))
    result = CmdResult(
        ran.returncode,
        None if ignore_output else ran.stdout.decode("utf-8"),
        None if ignore_output else ran.stderr.decode("utf-8"),
    )
    if not ignore_output:
        out_file.write(str(result.stdout))
        err_file.write(str(result.stderr))
    return result

def fuzz(target, function_name, runs=10000, corpus_dir="/tmp/corpus", max_len=128):
    fuzz_cmd = "python3 fuzz.py "
    fuzz_cmd += f"--function-name={function_name} "
    fuzz_cmd += f"--runs={runs} "
    fuzz_cmd += f"--max-len={max_len} "
    fuzz_cmd += f"--corpus={corpus_dir} "
    fuzz_cmd += f"{target}"

    res = run(fuzz_cmd)

    return res

def try_guess():
    pass

def try_examples():
    pass

def try_solver():
    pass

def try_eval(target_func, inputs: List[Tuple]):
    crashes = []
    for inp in inputs:
        try:
            target_func(*inp)
        except:
            crashes.append(inp)
    return crashes

def fuzzyagent() -> int:

    api_key = os.environ.get("FUZZYAGENT_API_KEY", "")
    api_endpoint = os.environ.get("FUZZYAGENT_ENDPOINT", "")
    api_model_name = os.environ.get("FUZZYAGENT_MODEL_NAME", "")

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--function-name")
    parser.add_argument("-u", "--fuzz-only", action="store_true")
    parser.add_argument("-r", "--runs", type=int, default=1)
    parser.add_argument("-l", "--fuzzer-loops", type=int, default=10000)
    parser.add_argument("-m", "--max-len", type=int, default=128)
    parser.add_argument("--prompts", type=str, default="prompts/")
    parser.add_argument("targets", nargs="*")
    args = parser.parse_args()

    if len(args.targets) != 1:
        print("[FUZZYAGENT ERROR] you must provide the path to one target")
        sys.exit(1)

    target_file = args.targets[0]

    loader = importlib.machinery.SourceFileLoader("target_module", args.targets[0])
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    all_func_names = [name for name in dir(mod) if name[:2] != "__"]
    if len(all_func_names) < 1:
        print("[FUZZYAGENT ERROR] the target script doesn't contain any function definitions")
        sys.exit(1)
    target_func_name = all_func_names[0] if args.function_name is None else args.function_name
    target_func = getattr(mod, target_func_name)

    sig = inspect.signature(target_func)
    params = [param for param in sig.parameters.keys()]
    arity = len(params)
    global sig_types
    sig_types = [param[1].annotation if param[1].annotation is not inspect.Parameter.empty else int for param in sig.parameters.items()]


    if args.only_fuzz:
        res = fuzz(target_file, target_func_name)
    else:
        con = duckdb.connect(database = ":memory:")
        table_schema = ", ".join([f"{p} {type_to_sql(t)}" for p, t in zip(params, sig_types)])
        con.execute(f"CREATE TABLE guesses ({table_schema})")
        prompt_files = [(f, os.path.join(args.prompts, f)) for f in os.listdir(args.prompts) if os.path.isfile(os.path.join(args.prompts, f))]

        prompts = {p[0].replace(".txt", ""): PromptTemplate.from_file(p[1]) for p in prompt_files}

        llm = VLLMOpenAI(
            openai_api_key=api_key,
            openai_api_base=api_endpoint+"/v1",
            model_name=api_model_name,
            model_kwargs={"stop": ["."]},
        )


        guess_chain = prompts["guess"] | llm
        examples_chain = prompts["guess_examples"] | llm
        label_chain = prompts["label_blocks"] | llm
        assemble_chain = prompts["assemble_conditions"] | llm

        for i in range(args.runs):
            guesses = try_guess(guess_chain)
            try_eval(target_func, guesses)
            fuzz(target_file, target_func_name, max_len=args.max_len, runs=args.fuzzer_loops)
            # check
            save_guesses(guesses, corpus_dir=args.corpus_dir)
            insert_guesses(con, arity, guesses)

            guesses = try_examples(examples_chain, examples=topn(con, n=2))
            try_eval(target_func, guesses)
            fuzz(target_file, target_func_name, max_len=args.max_len, runs=args.fuzzer_loops)
            # check
            save_guesses(guesses, corpus_dir=args.corpus_dir)
            insert_guesses(con, arity, guesses)

            guesses = try_solver(label_chain, assemble_chain)
            try_eval(target_func, guesses)
            fuzz(target_file, target_func_name, max_len=args.max_len, runs=args.fuzzer_loops)
            save_guesses(guesses, corpus_dir=args.corpus_dir)
            insert_guesses(con, arity, guesses)

    return 0

def main() -> int:
    rc = fuzzyagent()
    return rc

if __name__ == "__main__":
    sys.exit(main())
