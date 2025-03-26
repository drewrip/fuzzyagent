import sys
import os
import argparse
import importlib
import importlib.machinery
import types
import inspect
import subprocess
import hashlib
import re
import json
import itertools
import keyword
import time
from typing import List, Optional, Tuple
from dataclasses import dataclass
import z3
from langchain_community.llms import VLLMOpenAI
from langchain_core.prompts import PromptTemplate
import duckdb

trace = {
    "start_time": 0,
    "end_time": 0,
    "fuzz_only": False,
    "run_number": 0,
    "caught_in_phase": "",
    "caught_by": "", # Eval or phase
    "results": "",
    "found": False,
    "file": "",
    "function": "",
}

output_path = None

RE_CODE_POINTS = r'# (\[[a-zA-Z0-9\-\_]+\])'
RE_IDENTIFIER = r'([^\d\W]\w*)'
RE_JSON_BLOCK = r'```(json)?([^(```)]*)```'

def sync_trace(time=None, run=None, current_phase=None, about_to_test=None, results=None, found=None):
    global trace
    if time is not None:
        trace["end_time"] = time
    if run is not None:
        trace["run_number"] = run
    if current_phase is not None:
        trace["caught_in_phase"] = current_phase
    if about_to_test is not None:
        trace["caught_by"] = about_to_test
    if results is not None:
        trace["results"] = results
    if found is not None:
        trace["found"] = found


def type_to_sql(t: type) -> str:
    if t is float:
        return "DOUBLE"
    else:
        return "INTEGER"

def default_for_type(t: type):
    if t is float:
        return float(0.0)
    else:
        return int(0)

def numeric_z3(t: type, p: str):
    if t is float:
        return z3.Float64(p)
    else:
        return z3.Int(p)

def valid_condition(condition: str, params: List[str]) -> bool:
    if type(condition) is not str:
        return False

    ids = re.findall(RE_IDENTIFIER, condition)
    if len(ids) < 1:
        return False

    for id in ids:
        if id not in params:
            return False
        if keyword.iskeyword(id):
            return False
    return True

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
    ran = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = CmdResult(
        ran.returncode,
        ran.stdout.decode("utf-8"),
        ran.stderr.decode("utf-8"),
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

    res = run(fuzz_cmd, ignore_output=True)

    return res

def try_guess(guess_chain, target_func) -> List[Tuple]:
    code = inspect.getsource(target_func)
    sig = inspect.signature(target_func)
    params = [param for param in sig.parameters.keys()]
    sig_types = [param[1].annotation if param[1].annotation is not inspect.Parameter.empty else int for param in sig.parameters.items()]

    candidates = {param: set() for param in params}

    response = guess_chain.invoke({"code_snippet": code, "language": "Python"})

    json_blocks = re.findall(RE_JSON_BLOCK, response)
    for block in json_blocks:
        try:
            solution = json.loads(block[1])
            for variable, value in solution.items():
                try:
                    if variable in params and type(value) is sig_types[params.index(variable)]:
                        candidates[variable].add(value)
                except:
                    pass
        except:
            pass

    possible = []
    for param in params:
        possible.append(list(candidates[param]))

    return list(itertools.product(*possible))

def try_examples(examples_chain, target_func, examples=[]) -> List[Tuple]:
    code = inspect.getsource(target_func)
    sig = inspect.signature(target_func)
    params = [param for param in sig.parameters.keys()]
    sig_types = [param[1].annotation if param[1].annotation is not inspect.Parameter.empty else int for param in sig.parameters.items()]

    examples_string = "\n".join([json.dumps({p: v for p, v in zip(params, list(example))}, indent=4) for example in examples])
    candidates = {param: set() for param in params}
    response = examples_chain.invoke({"code_snippet": code, "language": "Python", "examples": examples_string})

    json_blocks = re.findall(RE_JSON_BLOCK, response)
    for block in json_blocks:
        try:
            solution = json.loads(block[1])
            for variable, value in solution.items():
                try:
                    if variable in params and type(value) is sig_types[params.index(variable)]:
                        candidates[variable].add(value)
                except:
                    pass
        except:
            pass

    possible = []
    for param in params:
        possible.append(list(candidates[param]))

    return list(itertools.product(*possible))

def try_solver(label_chain, assemble_chain, target_func) -> List[Tuple]:
    code = inspect.getsource(target_func)
    sig = inspect.signature(target_func)
    params = [param for param in sig.parameters.keys()]
    sig_types = [param[1].annotation if param[1].annotation is not inspect.Parameter.empty else int for param in sig.parameters.items()]

    candidates = []

    labeled_code = label_chain.invoke({"code_snippet": code, "language": "Python", "comment_syntax": "#"})
    code_points = re.findall(RE_CODE_POINTS, labeled_code)
    for code_point in code_points:
        conditions_response = assemble_chain.invoke({"code_snippet": labeled_code, "language": "Python", "code_point": code_point})
        json_blocks = re.findall(RE_JSON_BLOCK, conditions_response)
        for block in json_blocks:
            s = z3.Solver()
            variables = {p: numeric_z3(t, p) for p, t in zip(params, sig_types)}
            locals().update(variables) # yikes
            try:
                conditions = json.loads(block[1])
            except json.decoder.JSONDecodeError:
                continue
            for condition in conditions:
                if valid_condition(condition, params):
                    try:
                        s.add(eval(condition))
                    except z3.z3types.Z3Exception:
                        continue
            if s.check() == z3.sat:
                m = s.model()
                included_variables = {str(v): v for v in m.decls()}
                solution = []
                for param, t in zip(params, sig_types):
                    if param in included_variables:
                        solution.append(m[included_variables[param]])
                    else:
                        solution.append(default_for_type(t))
                candidates.append(tuple(solution))

    return candidates

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
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("--prompts", type=str, default="prompts/")
    parser.add_argument("targets", nargs="*")
    args = parser.parse_args()

    if args.output is not None:
        global output_path
        output_path = args.output

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

    trace["start_time"] = int(time.time())
    trace["function"] = target_func_name
    trace["file"] = target_file
    if args.fuzz_only:
        trace["fuzz_only"] = True
        sync_trace(time=int(time.time()))
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
            sync_trace(run=i)

            # Inital Guess Phase
            sync_trace(time=int(time.time()), current_phase="guess")
            guesses = try_guess(guess_chain, target_func)
            sync_trace(time=int(time.time()), about_to_test="eval")
            res = try_eval(target_func, guesses)
            if len(res) > 0:
                sync_trace(time=int(time.time()), results=str(res), found=True)
                return 0
            sync_trace(time=int(time.time()), about_to_test="fuzzer")
            res = fuzz(target_file, target_func_name, max_len=args.max_len, runs=args.fuzzer_loops)
            if res.rc != 0:
                sync_trace(time=int(time.time()), results=str(res.stderr), found=True)
                return 0
            save_guesses(guesses)
            insert_guesses(con, arity, guesses)

            # Guess w/ Examples Phase
            sync_trace(time=int(time.time()), current_phase="examples")
            guesses = try_examples(examples_chain, target_func, examples=topn(con, params[0], n=2))
            sync_trace(time=int(time.time()), about_to_test="eval")
            res = try_eval(target_func, guesses)
            if len(res) > 0:
                sync_trace(time=int(time.time()), results=str(res), found=True)
                return 0
            sync_trace(time=int(time.time()), about_to_test="fuzzer")
            res = fuzz(target_file, target_func_name, max_len=args.max_len, runs=args.fuzzer_loops)
            if res.rc != 0:
                sync_trace(time=int(time.time()), results=str(res.stderr), found=True)
                return 0
            save_guesses(guesses)
            insert_guesses(con, arity, guesses)

            # Attempt to Solve
            sync_trace(time=int(time.time()), current_phase="solver")
            guesses = try_solver(label_chain, assemble_chain, target_func)
            sync_trace(time=int(time.time()), about_to_test="eval")
            res = try_eval(target_func, guesses)
            if len(res) > 0:
                sync_trace(time=int(time.time()), results=str(res), found=True)
                return 0
            sync_trace(time=int(time.time()), about_to_test="fuzzer")
            res = fuzz(target_file, target_func_name, max_len=args.max_len, runs=args.fuzzer_loops)
            if res.rc != 0:
                sync_trace(time=int(time.time()), results=str(res.stderr), found=True)
                return 0
            save_guesses(guesses)
            insert_guesses(con, arity, guesses)
    return 1

def main() -> int:
    rc = fuzzyagent()

    if trace["found"]:
        print(f"Found exception in {trace['file']}:{trace['function']}:\n\n{trace['results']}")

    if output_path is not None:
        with open(output_path, "w") as f:
            json.dump(trace, f, indent=4)
    return rc

if __name__ == "__main__":
    sys.exit(main())
