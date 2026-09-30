"""
python_seed_prompts.py — 140 Python-focused seed prompts for BitQwen 1.0 SFT data generation.

Sources: Generated via Gemini + Copilot, reviewed and organized by category.
All seeds are Python-specific (not generic logic or systems debugging).

Categories:
  - PYTHON_COT_DEBUG_SEEDS (20): Chain-of-thought Python debugging
  - PYTHON_SYLLOGISM_EVOL_SEEDS (20): Python syllogisms + Evol-Instruct
  - PYTHON_EVOL_INSTRUCT_SEEDS (20): Instruction evolution into advanced coding tasks
  - PYTHON_COT_REPAIR_SEEDS (20): Correcting flawed Python reasoning traces
  - PYTHON_ADVERSARIAL_SEEDS (20): Python edge cases / stress tests
  - PYTHON_LATENT_BUG_SEEDS (20): Hidden logic / memory / runtime traps
  - COT_DEBUG_SEEDS (20): General CoT debugging (from first batch)
  - SYLLOGISM_EVOL_SEEDS (20): General syllogisms (from first batch)

Total: 160 seeds across 8 categories.
"""

# ============================================================================
# 20 Python CoT-Debug Seeds (brand-new, Python-focused)
# ============================================================================
PYTHON_COT_DEBUG_SEEDS = [
    "Trace this Python list comprehension step-by-step to find why the output list contains duplicates.",
    "Debug this Python generator: walk through each yield to identify why StopIteration is raised early.",
    "Step through this Python decorator to find why the wrapped function loses its metadata.",
    "Trace this async/await coroutine chain to identify where the event loop stalls.",
    "Debug this Python classmethod: explain why the class state mutates unexpectedly.",
    "Walk through this Python context manager to find why __exit__ is never called.",
    "Trace this pandas DataFrame operation to identify why chained indexing returns a copy instead of a view.",
    "Debug this Python recursion: identify the exact call depth where the recursion limit is hit.",
    "Step through this Python import cycle to find where circular dependency resolution fails.",
    "Trace this Python dict update sequence to identify why keys are overwritten.",
    "Debug this Python slicing operation: explain why negative indices produce unexpected results.",
    "Walk through this Python multiprocessing Pool to find why workers hang on join().",
    "Trace this Python regex substitution to identify why backreferences fail.",
    "Debug this Python property setter: explain why attribute assignment triggers infinite recursion.",
    "Step through this Python lambda capture to find why late binding breaks the loop.",
    "Trace this Python type hint resolution to identify why forward references fail.",
    "Debug this Python f-string formatting: explain why braces are interpreted incorrectly.",
    "Walk through this Python heapq operation to find why the smallest element is not popped.",
    "Trace this Python dataclass default factory to identify why mutable defaults leak state.",
    "Debug this Python exception chain: explain why __cause__ and __context__ differ.",
]

# ============================================================================
# 20 Python Syllogism + Evol-Instruct Seeds (Python-focused)
# ============================================================================
PYTHON_SYLLOGISM_EVOL_SEEDS = [
    "All Python iterables can be looped over. Generators are iterables. Evaluate whether all generators can be rewound.",
    "If all Python objects have a type, and some objects have no attributes, deduce whether all attribute-less objects share a type.",
    "All Python functions can be called. Some callables are classes. Determine whether all classes are functions.",
    "If every Python list preserves order, and tuples preserve order, evaluate whether all ordered collections are mutable.",
    "All Python exceptions inherit from BaseException. No warnings inherit from BaseException. Determine whether warnings can be raised as exceptions.",
    "If A is a Python dict and B is a mapping, and all mappings support key iteration, deduce whether A and B share identical semantics.",
    "All Python integers are immutable. Some immutable objects are interned. Determine whether all integers are interned.",
    "If all Python modules define __dict__, and some modules define __all__, evaluate whether __all__ is required for module introspection.",
    "All Python coroutines require an event loop. Async generators are coroutines. Determine whether async generators require scheduling.",
    "If every Python class has an MRO, and some objects have no MRO, deduce whether all objects are classes.",
    "All Python comprehensions create new objects. Some comprehensions capture outer variables. Determine whether all captured variables are copied.",
    "If all Python strings are sequences, and no sequences are hashable except strings and tuples, evaluate whether strings are special sequences.",
    "All Python decorators wrap callables. Some decorators return non-callables. Determine whether all decorators preserve callability.",
    "If every Python iterator exhausts, and some iterators restart, deduce whether restarting violates iterator protocol.",
    "All Python sets are unordered. Frozen sets are sets. Determine whether frozen sets are unordered.",
    "If all Python classes inherit from object, and some objects do not behave like classes, evaluate whether inheritance implies behavior.",
    "All Python context managers implement __enter__. Some objects implement __enter__ but not __exit__. Determine whether they qualify as context managers.",
    "If every Python file object is iterable, and some iterables are infinite, deduce whether file objects can be infinite.",
    "All Python namedtuples are tuples. Some tuples are not namedtuples. Determine whether namedtuples add semantic constraints.",
    "If all Python dicts map keys to values, and some dicts use custom hashing, evaluate whether custom hashing affects equality semantics.",
]

# ============================================================================
# 20 Python Evol-Instruct Seeds (Mutations into Advanced Coding Tasks)
# ============================================================================
PYTHON_EVOL_INSTRUCT_SEEDS = [
    "Mutate a basic Python dictionary lookup into a custom class implementing __getitem__ with fallback chain resolution.",
    "Evolve a simple for loop over a file into a zero-copy generator using mmap and memoryview slice parsing.",
    "Transform a standard dataclass into an immutable slot-backed model enforcing dynamic type validation at instantiation.",
    "Evolve a synchronous web request wrapper into an asyncio batch fetcher with exponential backoff and rate-limiting semaphores.",
    "Mutate a basic try/except block into a custom context manager that captures, logs, and rewrites frame tracebacks.",
    "Evolve a recursive factorial function into a tail-call optimized trampoline function using generators.",
    "Transform a standard list sorting task into a custom Timsort wrapper with a user-defined comparator operating on weak references.",
    "Evolve a naive global cache into a thread-safe LRU cache using threading.Lock and a doubly-linked list dict.",
    "Mutate a simple function call into a dynamic dispatch matrix using functools.singledispatch and type annotations.",
    "Evolve a plain text parser into an AST visitor subclassing ast.NodeTransformer to rewrite binary operations.",
    "Transform a standard CSV reader into a memory-efficient chunked streaming pipeline using yield from.",
    "Evolve a basic subclass hierarchy into an abstract base class using abc.ABCMeta with strict __init_subclass__ validation.",
    "Mutate a simple print statement logger into a custom logging.Handler that queues records asynchronously across processes.",
    "Evolve a standard matrix multiplication loop into a vectorized ctypes wrapper calling a C shared library.",
    "Transform a simple attribute assignment into a custom descriptor with type enforcement and access logging.",
    "Evolve a basic json.loads call into a custom json.JSONDecoder capable of handling circular object graphs.",
    "Mutate a standard boolean check into a bitwise flag mask class backed by enum.Flag.",
    "Evolve a simple string concatenation loop into an in-memory io.StringIO buffer stream with custom flushing limits.",
    "Transform a standard pytest unit test into a parameterized fixture factory utilizing dynamic module importation.",
    "Evolve a standard signal handler into an event-driven observer pattern using weakref.WeakSet.",
]

# ============================================================================
# 20 Python CoT-Repair Seeds (Correcting Flawed Python Reasoning Traces)
# ============================================================================
PYTHON_COT_REPAIR_SEEDS = [
    "Repair this reasoning trace: incorrectly claims isinstance(True, int) evaluates to False due to type hierarchy misunderstandings.",
    "Fix this CoT: incorrectly concludes [[]] * 3 creates three independent list instances rather than aliased references.",
    "Repair this step-by-step trace: wrongly asserts default={} in function arguments creates a fresh dict per execution.",
    "Fix this reasoning block: erroneously states __del__ is guaranteed to execute immediately when an object leaves scope.",
    "Repair this trace: incorrectly assumes sys.getrefcount(obj) returns 1 for a newly created unreferenced object.",
    "Fix this CoT: falsely claims asyncio.gather() cancels all pending tasks instantly when one task raises an exception.",
    "Repair this trace: misinterprets tuple([1, 2, [3]]) as fully immutable, ignoring the nested list mutation.",
    "Fix this reasoning block: incorrectly asserts dict key lookup is always O(1) regardless of hash collision density.",
    "Repair this trace: wrongly claims global var inside a nested function modifies the variable in the enclosing outer function scope.",
    "Fix this CoT: erroneously concludes yield from is identical to an inner for item in gen: yield item without handling thrown exceptions.",
    "Repair this step-by-step trace: falsely states __slots__ prevents adding new dynamic attributes to subclasses.",
    "Fix this reasoning block: incorrectly asserts str.encode() mutates the original string buffer in place.",
    "Repair this trace: wrongly concludes math.isclose() and == yield identical results for subnormal floating-point numbers.",
    "Fix this CoT: erroneously claims collections.defaultdict missing key access does not modify .keys().",
    "Repair this trace: falsely asserts threading.Thread provides true parallel CPU core execution in CPython without releasing GIL.",
    "Fix this reasoning block: incorrectly claims gc.collect() instantly reclaims memory occupied by circular reference cycles with overridden __del__.",
    "Repair this trace: wrongly states functools.lru_cache works on unhashable arguments if maxsize=None.",
    "Fix this CoT: erroneously asserts copy.copy() performs a deep copy of nested tuples containing lists.",
    "Repair this step-by-step trace: falsely claims itertools.tee() creates independent memory streams without buffering overhead.",
    "Fix this reasoning block: incorrectly concludes @staticmethod passes implicit self instance during metaclass instantiation.",
]

# ============================================================================
# 20 Python Adversarial Seeds (Stress-Testing Python Edge Cases)
# ============================================================================
PYTHON_ADVERSARIAL_SEEDS = [
    "Examine x = (1, 2, [3, 4]); x[2] += [5, 6]: predict whether TypeError is raised AND whether the list mutates.",
    "Analyze all([]) vs any([]): evaluate why both return different booleans and stress-test the vacuous truth principle.",
    "Evaluate def f(a, b=[]): b.append(a); return b: predict outputs for three consecutive calls with no explicit b argument.",
    "Analyze a = 256; b = 256; a is b versus a = 257; b = 257; a is b in CPython REPL vs script compilation modes.",
    "Stress-test class A: pass; a = A(); a.__class__ = int: explain why CPython rejects dynamic type re-assignment.",
    "Evaluate [i for i in range(5) if i % 2 == 0] vs scope leakage in Python 2 vs Python 3 list comprehensions.",
    "Examine try: return 1 finally: return 2: predict the return value and trace frame unwinding mechanics.",
    "Analyze False == 0 == '0' vs chained comparison expansion: prove why 0 == '0' breaks transitivity.",
    "Stress-test d = {True: 'a', 1: 'b', 1.0: 'c'}: evaluate the final size of d and explain key collision semantics.",
    "Evaluate round(2.5) vs round(3.5): explain banker's rounding and floating-point representation limits.",
    "Examine super() inside a standalone function outside a class: predict the raised RuntimeError message and frame lookup.",
    "Analyze class Metaclass(type): pass vs class Class(metaclass=Metaclass): pass: trace the type.__new__ call order.",
    "Stress-test del x inside a closure: explain why accessing x in the outer frame raises UnboundLocalError.",
    "Evaluate list.__init__(self) overridden inside __new__: analyze why initialization steps fail silently.",
    "Examine import math; import sys; sys.modules['math'] = None; import math: predict the ModuleNotFoundError trigger.",
    "Analyze isinstance(type, object) and isinstance(object, type): resolve the circular type-object boot trap.",
    "Stress-test sys.setrecursionlimit(10) vs native C-stack overflow: evaluate crash vs Python RecursionError.",
    "Evaluate from module import * inside a function body: explain why CPython syntax parser raises SyntaxError.",
    "Examine slice(None) == slice(None) vs hashing slice objects in dict keys: resolve unhashable slice type errors.",
    "Analyze exec('a = 10'); print(a) inside a function: explain local scope dict detachment and NameError.",
]

# ============================================================================
# 20 Python Latent Bug Seeds (Hidden Logic & Memory/Runtime Traps)
# ============================================================================
PYTHON_LATENT_BUG_SEEDS = [
    "Identify the latent bug in a weakref.finalize callback referencing the target object directly, creating a hard reference cycle.",
    "Spot the hidden bug in multiprocessing.Process passing a SQLite connection object across fork boundaries.",
    "Find the latent race condition in double-checked locking using a simple Python boolean flag instead of threading.Lock.",
    "Locate the hidden memory leak in an @lru_cache decorated instance method holding implicit strong references to self.",
    "Identify the latent trap in dataclass(slots=True) inheriting from a non-slot parent class containing __dict__.",
    "Spot the bug in asyncio.create_task() where the returned task object is unreferenced and subject to garbage collection.",
    "Find the subtle bug in signal.signal() calling non-reentrant logic or allocating memory inside signal handlers.",
    "Locate the latent bug in custom __eq__ implementation failing to return NotImplemented for unrecognized types.",
    "Identify the hidden bug in ctypes passing a Python string directly to c_char_p without keeping a reference to the temporary byte buffer.",
    "Spot the subtle bug in subprocess.Popen(stdout=PIPE) without reading output, causing OS pipe buffer deadlock.",
    "Find the latent bug in tempfile.NamedTemporaryFile accessed by name on Windows while still open.",
    "Locate the hidden bug in dict.keys() mutation during iteration in a multi-threaded background task.",
    "Identify the bug in custom __hash__ implementation returning a float, causing hash truncation and bucket mismatch.",
    "Spot the latent trap in sys.excepthook override failing to call the original hook for unhandled worker thread exceptions.",
    "Find the subtle bug in inspect.stack() called inside a high-throughput loop creating massive frame object allocations.",
    "Locate the hidden bug in re.compile() using raw string literals missing escape characters for dynamic string interpolations.",
    "Identify the latent bug in pathlib.Path.resolve() under symlink loops throwing unexpected RuntimeError.",
    "Spot the hidden trap in unittest.mock.patch.object mocking a property without specifying new_callable=PropertyMock.",
    "Find the latent bug in asyncio.wait() with FIRST_COMPLETED leaving remaining futures unawaited/uncanceled.",
    "Locate the subtle bug in __getattribute__ calling self.attr internally instead of super().__getattribute__().",
]

# ============================================================================
# 20 General CoT-Debug Seeds (from first batch — broader systems debugging)
# ============================================================================
COT_DEBUG_SEEDS = [
    "Debug this off-by-one error in binary search: trace each iteration step-by-step.",
    "Walk through the execution of this recursive DFS function and identify where the infinite loop occurs.",
    "Trace the memory allocation in this Rust snippet to find why a double-free condition is triggered.",
    "Analyze this SQL query execution plan line-by-line to explain why the index scan is bypassed.",
    "Step through this async Python task group to diagnose where the race condition occurs.",
    "Examine this matrix multiplication loop order and trace cache misses step-by-step.",
    "Walk through this regular expression state machine to explain why catastrophic backtracking happens.",
    "Trace this C++ pointer arithmetic step-by-step to locate the buffer overflow bug.",
    "Debug this JWT signature validation workflow by step-checking key extraction and padding steps.",
    "Analyze this PyTorch autograd graph flow step-by-step to trace why gradients vanish at layer 3.",
    "Walk through this React re-render cycle to identify which dependency array causes infinite renders.",
    "Trace this Go channel synchronization flow to find where the goroutine deadlock originates.",
    "Step through this Kubernetes ingress rule routing logic to identify why 404s are returned.",
    "Trace the token stream in this custom AST parser step-by-step to find the missing semicolon bug.",
    "Debug this multi-threaded queue pop operation step-by-step to identify the missed lock release.",
    "Walk through this Redis distributed lock implementation to explain where the split-brain state happens.",
    "Trace this Docker build layer cache step-by-step to identify why layer invalidation fails.",
    "Examine this floating-point rounding sequence step-by-step to isolate precision drift.",
    "Walk through this OAuth PKCE code exchange handshake step-by-step to find where state validation fails.",
    "Trace this linked list cycle detection algorithm step-by-step to confirm fast/slow pointer alignment.",
]

# ============================================================================
# 20 General Syllogism + Evol Seeds (from first batch — formal logic)
# ============================================================================
SYLLOGISM_EVOL_SEEDS = [
    "All birds lay eggs. Some aquatic creatures lay eggs. Suppose penguins are aquatic creatures that do not fly: construct a multi-step syllogism proving or disproving whether all egg-laying creatures fly.",
    "If all A are B, and no B are C, but some C are D under condition X: analyze whether any A can ever be D under condition X using formal categorical propositions.",
    "Premise 1: No compiled languages are purely interpreted. Premise 2: Java uses a JIT compiler. Premise 3: All JIT-compiled languages execute via bytecode. Evaluate the validity of deriving 'No bytecode languages are purely interpreted'.",
    "Given that all strict logicians avoid fallacies, and some programmers avoid fallacies, evaluate if any programmer is guaranteed to be a logician under classical syllogistic laws.",
    "All encrypted payloads require a key. No unauthenticated key is secure. Some secure keys are stored in HSMs. Determine the necessary truth value of 'Some encrypted payloads rely on HSMs'.",
    "If every continuous function on a closed interval is bounded, and f(x) is unbounded on [0,1], determine the formal syllogistic deduction regarding the continuity of f(x).",
    "Construct a valid categorical syllogism of form AAA-1, then evolve it by converting Premise 2 into a conditional statement and test for soundness.",
    "If system throughput exceeds 10k RPS, then queue latency spikes. Latency did not spike, but memory spiked. Deduce all valid logical conclusions using Modus Tollens.",
    "Chain four premises: All immutable objects are thread-safe. All thread-safe objects prevent race conditions. No object with race conditions is reliable. System X is immutable. Deduce System X's reliability.",
    "Either server A processes the job or server B processes the job, but not both simultaneously. Server A fails if load > 90%. Load is 95%. Deduce server B's processing state.",
    "All necessary truths are possible truths. No contingent statements are necessary truths. Evaluate whether contingent statements can be possible truths using modal syllogistic logic.",
    "No bug-free code is un-tested. Some un-tested code is deployed to production. Determine what can be validly inferred regarding code deployed to production.",
    "All deterministic algorithms yield identical outputs for identical inputs. Hash functions are deterministic algorithms. If Hash function H uses a seed based on system time, is H still a deterministic algorithm?",
    "If Agent X knows P implies Q, and Agent X knows P, but Agent X forgets Q when under memory pressure: evaluate the validity of 'Agent X always knows Q'.",
    "Some microservices are scalable. All scalable systems use async I/O. Therefore, some microservices use async I/O. Validate this syllogism's figure and mood.",
    "All software engineers write code. All data scientists write code. Therefore, all data scientists are software engineers. Identify the precise syllogistic fallacy and rewrite it into a valid form.",
    "If A then B; if B then C; if C then D. Suppose D is false and A is true: identify which premise must be violated.",
    "Element X is strictly greater than Element Y. Element Y is equal to or greater than Element Z. Prove whether Element X can ever equal Element Z.",
    "All valid API calls return 200 OK or 201 Created. No request with an invalid token is a valid API call. Request R returned 200 OK. Can we validly conclude Request R had a valid token?",
    "All verified theorems are proven. No unproven hypothesis is a verified theorem. Some AI outputs are unproven hypotheses. Evaluate whether any AI output can be a verified theorem.",
]

# ============================================================================
# 20 Python CoT-Debug Seeds (batch 2 — trace, debug, step-through)
# ============================================================================
PYTHON_COT_DEBUG_SEEDS_2 = [
    "Step through this Python list flattening function to find why nested lists are skipped.",
    "Trace this Python async task cancellation to identify why the coroutine never exits.",
    "Debug this Python class inheritance chain to find why super() resolves incorrectly.",
    "Walk through this Python slicing logic to explain why the step parameter breaks ordering.",
    "Trace this Python dictionary merge to find why nested keys are overwritten.",
    "Debug this Python iterator to identify why StopIteration is swallowed.",
    "Step through this Python file read loop to find why only half the file is processed.",
    "Trace this Python regex group capture to identify why named groups return None.",
    "Debug this Python subprocess call to find why stdout is empty.",
    "Step through this Python heap push/pop sequence to find why ordering is reversed.",
    "Trace this Python functools.lru_cache behavior to identify why cache misses spike.",
    "Debug this Python context manager to find why exceptions are suppressed.",
    "Step through this Python type hint resolution to find why generics fail.",
    "Trace this Python JSON serialization to identify why custom objects lose fields.",
    "Debug this Python lambda closure to find why variables bind late.",
    "Step through this Python multiprocessing queue to find why items vanish.",
    "Trace this Python decorator stacking to identify why return values change.",
    "Debug this Python property getter to find why recursion occurs.",
    "Step through this Python import hook to find why modules load twice.",
    "Trace this Python dataclass repr to identify why fields appear out of order.",
]

# ============================================================================
# 20 Python Semantic Reasoning Seeds (meaning, inference, conceptual)
# ============================================================================
PYTHON_SEMANTIC_REASONING_SEEDS = [
    "Explain why Python treats strings as sequences but not as collections.",
    "Determine whether Python's None can ever be equal to a falsy integer.",
    "Evaluate whether Python's is operator implies semantic equality.",
    "Explain why Python integers behave like singletons for small values.",
    "Determine whether Python's dict ordering guarantees semantic stability.",
    "Explain why Python treats functions as first-class objects.",
    "Evaluate whether Python's yield implies coroutine semantics.",
    "Explain why Python's True and False are integers.",
    "Determine whether Python's range is a container or a sequence.",
    "Explain why Python's set cannot contain lists.",
    "Determine whether Python's tuple immutability implies hashability.",
    "Explain why Python's float cannot represent all decimal values.",
    "Determine whether Python's bytes and bytearray share semantics.",
    "Explain why Python's global keyword does not affect module imports.",
    "Determine whether Python's with statement implies transactional semantics.",
    "Explain why Python's __dict__ is not always present.",
    "Determine whether Python's super() resolves methods deterministically.",
    "Explain why Python's async functions return coroutines.",
    "Determine whether Python's __slots__ affects inheritance semantics.",
    "Explain why Python's __eq__ does not imply __hash__.",
]

# ============================================================================
# 20 Python Structural Logic Seeds (logic chains, syllogisms, inference)
# ============================================================================
PYTHON_STRUCTURAL_LOGIC_SEEDS = [
    "All Python lists preserve order. Some ordered types are immutable. Determine whether any immutable ordered type must be a tuple.",
    "All Python exceptions inherit from BaseException. Some warnings do not. Determine whether warnings can be raised as exceptions.",
    "All Python iterators exhaust. Some generators restart. Determine whether restarting violates iterator protocol.",
    "All Python classes inherit from object. Some objects do not behave like classes. Determine whether inheritance implies behavior.",
    "All Python sets are unordered. Frozen sets are sets. Determine whether frozen sets are unordered.",
    "All Python modules define __dict__. Some objects define __dict__. Determine whether all objects are modules.",
    "All Python coroutines require scheduling. Some async generators are coroutines. Determine whether async generators require scheduling.",
    "All Python strings are sequences. Some sequences are not hashable. Determine whether strings are special sequences.",
    "All Python comprehensions create new objects. Some comprehensions capture outer variables. Determine whether captured variables are copied.",
    "All Python integers are immutable. Some immutable objects are interned. Determine whether all integers are interned.",
    "All Python decorators wrap callables. Some decorators return non-callables. Determine whether decorators preserve callability.",
    "All Python file objects are iterable. Some iterables are infinite. Determine whether file objects can be infinite.",
    "All Python namedtuples are tuples. Some tuples are not namedtuples. Determine whether namedtuples add semantic constraints.",
    "All Python dicts map keys to values. Some dicts use custom hashing. Determine whether custom hashing affects equality.",
    "All Python functions have code objects. Some callables do not. Determine whether all callables are functions.",
    "All Python classes have MRO. Some objects do not. Determine whether all objects are classes.",
    "All Python generators yield values. Some yield from other generators. Determine whether delegation changes generator identity.",
    "All Python comprehensions evaluate expressions. Some expressions mutate state. Determine whether comprehensions can mutate state.",
    "All Python coroutines await values. Some awaitables are not coroutines. Determine whether await implies coroutine.",
    "All Python dicts preserve insertion order. Some mappings do not. Determine whether dict is a special mapping.",
]

# ============================================================================
# 20 Python Evol-Instruct Seeds (batch 2 — instruction mutation)
# ============================================================================
PYTHON_EVOL_INSTRUCT_SEEDS_2 = [
    "Rewrite this Python instruction to require multi-file reasoning.",
    "Expand this Python debugging instruction into a multi-step diagnostic workflow.",
    "Compress this Python refactoring instruction into a single concise directive.",
    "Add constraints to this Python function design instruction to increase difficulty.",
    "Remove unnecessary constraints from this Python class design instruction.",
    "Rewrite this Python testing instruction to require property-based tests.",
    "Rewrite this Python data processing instruction to require streaming semantics.",
    "Add a real-world scenario to this Python concurrency instruction.",
    "Remove the scenario and generalize the Python concurrency instruction.",
    "Rewrite this Python parsing instruction to require AST manipulation.",
    "Rewrite this Python optimization instruction to require profiling.",
    "Add an evaluation metric to this Python algorithm instruction.",
    "Rewrite this Python debugging instruction to require step-by-step output.",
    "Rewrite this Python code generation instruction to require template rendering.",
    "Rewrite this Python data analysis instruction to require statistical validation.",
    "Rewrite this Python class design instruction to require mixin composition.",
    "Add a hypothetical constraint to this Python scheduling instruction.",
    "Remove the hypothetical constraint and generalize the instruction.",
    "Rewrite this Python packaging instruction to require dependency resolution.",
    "Rewrite this Python CLI instruction to require subcommand routing.",
]

# ============================================================================
# 20 Python Adversarial Reasoning Seeds (batch 2 — edge cases, traps)
# ============================================================================
PYTHON_ADVERSARIAL_SEEDS_2 = [
    "Determine why sorted([float('nan'), 1, 2]) behaves inconsistently across Python versions.",
    "Explain why 0.1 + 0.2 != 0.3 in Python.",
    "Determine why True == 1 but True is not 1.",
    "Explain why [] is [] is False but () is () is True.",
    "Determine why hash(-1) == hash(-2) in some Python versions.",
    "Explain why dict.fromkeys(['a','b'], []) causes shared state.",
    "Determine why lambda x: x inside a loop captures late binding.",
    "Explain why re.match behaves differently from re.search.",
    "Determine why int('09') fails in Python 2 but not Python 3.",
    "Explain why list.remove() raises ValueError even when the element exists.",
    "Determine why __getattr__ is not called for existing attributes.",
    "Explain why asyncio.sleep(0) yields control.",
    "Determine why defaultdict(list) shares no state but dict.setdefault does.",
    "Explain why functools.partial changes function signatures.",
    "Determine why pickle cannot serialize lambda functions.",
    "Explain why deepcopy fails on objects with custom __slots__.",
    "Determine why multiprocessing spawns extra processes on Windows.",
    "Explain why json.dumps cannot serialize sets.",
    "Determine why str.join fails when joining non-strings.",
    "Explain why zip truncates to the shortest iterable.",
]

# ============================================================================
# ALL_SEED_GROUPS — aggregated for the generator
# ============================================================================
ALL_SEED_GROUPS = {
    # Batch 1 (160 seeds)
    "python_cot_debug": PYTHON_COT_DEBUG_SEEDS,
    "python_syllogism_evol": PYTHON_SYLLOGISM_EVOL_SEEDS,
    "python_evol_instruct": PYTHON_EVOL_INSTRUCT_SEEDS,
    "python_cot_repair": PYTHON_COT_REPAIR_SEEDS,
    "python_adversarial": PYTHON_ADVERSARIAL_SEEDS,
    "python_latent_bug": PYTHON_LATENT_BUG_SEEDS,
    "general_cot_debug": COT_DEBUG_SEEDS,
    "general_syllogism_evol": SYLLOGISM_EVOL_SEEDS,
    # Batch 2 (100 seeds)
    "python_cot_debug_2": PYTHON_COT_DEBUG_SEEDS_2,
    "python_semantic_reasoning": PYTHON_SEMANTIC_REASONING_SEEDS,
    "python_structural_logic": PYTHON_STRUCTURAL_LOGIC_SEEDS,
    "python_evol_instruct_2": PYTHON_EVOL_INSTRUCT_SEEDS_2,
    "python_adversarial_2": PYTHON_ADVERSARIAL_SEEDS_2,
}

# ============================================================================
# Verification
# ============================================================================
if __name__ == "__main__":
    total = 0
    for name, seeds in ALL_SEED_GROUPS.items():
        assert len(seeds) == 20, f"{name} has {len(seeds)} seeds, expected 20"
        print(f"  {name}: {len(seeds)} seeds")
        total += len(seeds)
    print(f"\nTotal: {total} seeds across {len(ALL_SEED_GROUPS)} categories")
    assert total == 260, f"Expected 260 total, got {total}"
    print("DoD Verification PASSED: All 13 categories have exactly 20 seeds.")
