"""Two seams get a consumer registry: the plane-wire flag, and the HTTP route table.

A seam with a registry is one where "who reads this?" is answered by a test that reds with a
`file:line` when somebody becomes the next reader. Three seams already have one — the host
memory-manager import, the shipped-surface call families, the product environment switches — and
the pattern is the same each time: enumerate the consumers from the source, compare against a
NAMED list where every entry says why it is legitimate, and refuse anything the list does not
hold. The seams without one are a `grep` every later lane repeats and a count every later brief
gets wrong.

These two are the ones that already bit.

**The plane-wire flag.** `planes_wires_enabled()` is process-global state the front end reads:
while it is off, every dotted `@name.seg` is a swizzle; while it is on, it can be a plane read.
So one source has TWO parses, and every memo holding a value derived from one of them has to
carry the flag in its key or it serves one profile's answer under the other. A hand-back costed
that fix at four call sites; the lane that did it found FIVE memos behind four key sites, and
recorded the miss so the next reader would not pay it again. A recorded miss is a note. This is
the derivation.

**The route table.** Eleven route decorators register ten paths under an embedding host, and the
shipped frontend calls seven of them. A handler nobody drives and no test calls is a handler
that rots in silence — its failure branch can stop compiling with every suite still green — so
the three uncalled ones are driven by name elsewhere. Naming them is only worth something if a
FOURTH uncalled route cannot appear without anybody noticing, which is what the registry below
is for. It also checks the other direction: a frontend that fetches a path nobody registers is
a 404 the suite should find before a user does.

Both registries are source censuses: no import of the product package, no host, no HTTP client,
no CUDA. They answer the same in every environment, which is the only way a registry is worth
putting in the cheap tier.
"""
import ast
import pathlib
import re

from helpers import SubTestResult

_PKG = pathlib.Path(__file__).resolve().parent.parent

#: Directories that are not the product: their reads of a seam are a consumer of the TEST tree,
#: not of the shipped one, and a registry that counted them would red on every new benchmark.
_NOT_PRODUCT = {"tests", "benchmarks", "tools", "examples", "editor_build", "wiki", "docs"}


# ── the plane-wire flag ───────────────────────────────────────────────────────

#: The names that ARE the seam: the predicate, and the single alias every memo key reads it
#: through. The alias exists so the key costs one read and so there is ONE name to enumerate —
#: a `planes_wires_enabled()` spelled per site is a per-site opportunity to forget one, which
#: is how the blindness reached five memos at once.
_FLAG_NAMES = ("planes_wires_enabled", "_profile_key")

#: Where the predicate is defined. A definition is not a consumption, and this entry is here so
#: that MOVING the definition reds rather than quietly emptying the census.
_FLAG_DEFINED_IN = "tex_compiler/types.py"

#: Every product consumer, as `module::scope::name`, with the reason it is legitimate.
_FLAG_CONSUMERS = {
    "tex_cache.py::<module>::planes_wires_enabled":
        "the import that lets the splitback read the profile",
    "tex_cache.py::_DottedBindingSplitback.__init__::planes_wires_enabled":
        "snapshots the profile once per splitter, so one parse cannot straddle two profiles",
    "tex_lazy.py::<module>::planes_wires_enabled":
        "the import behind the alias below",
    "tex_lazy.py::<module>::_profile_key":
        "defines the ONE alias all four key sites read the flag through",
    "tex_lazy.py::_pristine_parse::_profile_key":
        "keys the shared parse memo - this single site serves BOTH tex_lazy._parse_memo and "
        "tex_roi._parse_memo, which is why five memos sit behind four key sites",
    "tex_lazy.py::lazy_required_bindings::_profile_key":
        "keys tex_lazy._memo: the required-binding answer moves with the profile",
    "tex_marshalling.py::<module>::planes_wires_enabled":
        "the import behind the three egress reads below",
    "tex_marshalling.py::PlanesValue.__init__::planes_wires_enabled":
        "refuses to mint a PLANES value while the profile forbids plane wires",
    "tex_marshalling.py::expand_plane_bindings::planes_wires_enabled":
        "expands a PLANES wire into plane rows only while the profile allows it",
    "tex_marshalling.py::infer_binding_type::planes_wires_enabled":
        "types an incoming value as PLANES only while the profile allows it",
    "tex_roi.py::<module>::_profile_key":
        "imports the alias rather than re-spelling the predicate",
    "tex_roi.py::region_dependent_cached::_profile_key":
        "keys tex_roi._region_dep_memo: the verdict is read off a profile-dependent parse",
    "tex_roi.py::_walk::_profile_key":
        "keys tex_roi._walk_memo: the walk's answer is derived from that same parse",
}

#: The memos whose KEY carries the flag, and the site that puts it there. Five memos, four
#: sites: a brief once costed this at four memos and the lane found the fifth.
_FLAG_KEYED_MEMOS = {
    "tex_lazy._parse_memo": "tex_lazy.py::_pristine_parse::_profile_key",
    "tex_roi._parse_memo": "tex_lazy.py::_pristine_parse::_profile_key",
    "tex_lazy._memo": "tex_lazy.py::lazy_required_bindings::_profile_key",
    "tex_roi._region_dep_memo": "tex_roi.py::region_dependent_cached::_profile_key",
    "tex_roi._walk_memo": "tex_roi.py::_walk::_profile_key",
}


class _SeamVisitor(ast.NodeVisitor):
    """Every reference to a seam name, tagged with the scope that makes it."""

    def __init__(self, names):
        self.names, self.stack, self.hits = set(names), [], []

    def _scoped(self, node):
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _scoped

    def _record(self, name, lineno):
        self.hits.append((".".join(self.stack) or "<module>", name, lineno))

    def visit_Name(self, node):
        if node.id in self.names:
            self._record(node.id, node.lineno)

    def visit_Attribute(self, node):
        if node.attr in self.names:
            self._record(node.attr, node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            if alias.name in self.names:
                self._record(alias.name, node.lineno)


def product_modules():
    for p in sorted(_PKG.rglob("*.py")):
        rel = p.relative_to(_PKG).as_posix()
        if rel.split("/")[0] in _NOT_PRODUCT:
            continue
        yield rel, p


def census_flag_consumers() -> dict:
    """`{module::scope::name: "module:line"}` over the product tree."""
    found = {}
    for rel, p in product_modules():
        v = _SeamVisitor(_FLAG_NAMES)
        v.visit(ast.parse(p.read_text(encoding="utf-8")))
        for scope, name, lineno in v.hits:
            found.setdefault(f"{rel}::{scope}::{name}", f"{rel}:{lineno}")
    return found


def test_simp3_plane_wire_flag_consumer_registry(r: SubTestResult):
    print("\n--- SIMP-3: every reader of the plane-wire flag is in the registry ---")
    found = census_flag_consumers()
    new = sorted(f"{k}  ({found[k]})" for k in found if k not in _FLAG_CONSUMERS)
    gone = sorted(k for k in _FLAG_CONSUMERS if k not in found)
    if new:
        r.fail("SIMP-3 plane-wire registry (new consumer)",
               f"{len(new)} reader(s) of the plane-wire flag are not in the registry. A read "
               f"that keys a memo needs the flag IN that key, or it serves one egress "
               f"profile's answer under the other:\n  " + "\n  ".join(new))
        return
    if gone:
        r.fail("SIMP-3 plane-wire registry (stale entry)",
               f"the registry names {len(gone)} consumer(s) that no longer exist - a list that "
               f"over-states the surface stops being read:\n  " + "\n  ".join(gone))
        return
    r.ok(f"all {len(found)} plane-wire readers are registered with a reason")


def test_simp3_plane_wire_flag_is_defined_where_the_registry_says(r: SubTestResult):
    print("\n--- SIMP-3: the plane-wire predicate is defined in one place ---")
    definers = []
    for rel, p in product_modules():
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(node, ast.FunctionDef) and node.name == "planes_wires_enabled":
                definers.append(f"{rel}:{node.lineno}")
    if len(definers) != 1 or not definers[0].startswith(_FLAG_DEFINED_IN + ":"):
        r.fail("SIMP-3 plane-wire definer",
               f"expected exactly one definition in {_FLAG_DEFINED_IN}, found {definers} - "
               f"the consumer census enumerates READS, so a moved or duplicated definition "
               f"would empty it without reddening anything else")
        return
    r.ok(f"planes_wires_enabled is defined once, at {definers[0]}")


def test_simp3_every_profile_keyed_memo_names_its_key_site(r: SubTestResult):
    print("\n--- SIMP-3: the five profile-keyed memos name a real key site ---")
    found = census_flag_consumers()
    bad = [f"{memo}: names {site!r}, which is not a registered consumer"
           for memo, site in _FLAG_KEYED_MEMOS.items() if site not in found]
    missing = []
    for memo in _FLAG_KEYED_MEMOS:
        module, attr = memo.split(".", 1)
        path = _PKG / f"{module}.py"
        if not path.is_file():
            missing.append(f"{memo}: {module}.py is not a product module")
            continue
        text = path.read_text(encoding="utf-8")
        if not re.search(rf"^{re.escape(attr)}\s*[:=]", text, re.M):
            missing.append(f"{memo}: {module}.py declares no module-level {attr}")
    if bad or missing:
        r.fail("SIMP-3 profile-keyed memos", "\n  ".join(bad + missing))
        return
    sites = len(set(_FLAG_KEYED_MEMOS.values()))
    r.ok(f"{len(_FLAG_KEYED_MEMOS)} memos carry the profile, through {sites} key sites")


# ── the HTTP route table ──────────────────────────────────────────────────────

_INIT = _PKG / "__init__.py"
_FRONTEND = _PKG / "js" / "tex_extension.js"

#: `(METHOD, path)` -> `(handler, who drives it)`. "frontend" means the shipped extension
#: fetches it; anything else must name what drives the handler instead, because a registered
#: handler nobody calls is one whose failure branch can stop compiling unnoticed.
_ROUTES = {
    ("GET", "/tex_wrangle/snippets"): ("get_snippets", "frontend"),
    ("POST", "/tex_wrangle/free_caches"):
        ("free_caches", "no frontend caller - driven by name in tests/test_v035_hygiene.py"),
    ("GET", "/tex_wrangle/doctor"): ("doctor", "frontend"),
    ("POST", "/tex_wrangle/chain_preflight"): ("chain_preflight", "frontend"),
    ("POST", "/tex_wrangle/detect_regions"): ("detect_regions", "frontend"),
    ("POST", "/tex_wrangle/check"): ("check_source", "frontend"),
    ("GET", "/tex_wrangle/user_snippets"): ("get_user_snippets", "frontend"),
    ("POST", "/tex_wrangle/user_snippets"): ("set_user_snippets", "frontend"),
    ("POST", "/tex_wrangle/publish_tool"): ("publish_tool", "frontend"),
    ("GET", "/tex_wrangle/list_tools"):
        ("list_installed_tools",
         "no frontend caller - driven by name in tests/test_v035_hygiene.py"),
    ("GET", "/tex_wrangle/docs/{page}"):
        ("get_offline_docs",
         "no frontend caller - driven by name in tests/test_v035_hygiene.py, including the "
         "path-traversal refusal the whitelist owes"),
}

_FETCH_RE = re.compile(r"/tex_wrangle/([A-Za-z0-9_]+)")


def census_routes() -> dict:
    """`{(METHOD, path): (handler, "file:line")}` from the route decorators."""
    src = _INIT.read_text(encoding="utf-8")
    out = {}
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for d in node.decorator_list:
            if not (isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
                    and isinstance(d.func.value, ast.Name) and d.func.value.id == "routes"):
                continue
            path = d.args[0].value if d.args and isinstance(d.args[0], ast.Constant) else None
            out[(d.func.attr.upper(), path)] = (node.name, f"__init__.py:{node.lineno}")
    return out


def frontend_paths() -> set:
    """The first path segment of every `/tex_wrangle/...` the shipped extension fetches."""
    if not _FRONTEND.is_file():
        return None
    return set(_FETCH_RE.findall(_FRONTEND.read_text(encoding="utf-8")))


def test_simp3_route_table_consumer_registry(r: SubTestResult):
    print("\n--- SIMP-3: every registered route is in the registry, with its driver ---")
    found = census_routes()
    new = sorted(f"{m} {p}  -> {h}  ({where})" for (m, p), (h, where) in found.items()
                 if (m, p) not in _ROUTES)
    gone = sorted(f"{m} {p}" for (m, p) in _ROUTES if (m, p) not in found)
    moved = sorted(f"{m} {p}: registry says {_ROUTES[(m, p)][0]}, decorator says {h}"
                   for (m, p), (h, _w) in found.items()
                   if (m, p) in _ROUTES and _ROUTES[(m, p)][0] != h)
    if new:
        r.fail("SIMP-3 route registry (new route)",
               f"{len(new)} route(s) register under a host and are in no registry. A route "
               f"needs a caller or a test that drives it by name, or its failure branch rots "
               f"with every suite green:\n  " + "\n  ".join(new))
    elif gone:
        r.fail("SIMP-3 route registry (stale entry)",
               "the registry names route(s) that no longer register:\n  " + "\n  ".join(gone))
    elif moved:
        r.fail("SIMP-3 route registry (handler moved)", "\n  ".join(moved))
    else:
        driven = sum(1 for h, w in _ROUTES.values() if w != "frontend")
        r.ok(f"{len(found)} route decorators over {len({p for _m, p in found})} paths, all "
             f"registered; {driven} have no frontend caller and name what drives them")


def test_simp3_the_frontend_and_the_route_table_agree(r: SubTestResult):
    print("\n--- SIMP-3: the shipped frontend fetches exactly the routes it claims ---")
    js = frontend_paths()
    if js is None:
        r.fail("SIMP-3 route/frontend agreement",
               "js/tex_extension.js is missing; the frontend half of the registry cannot be "
               "derived")
        return
    registered = {p.split("/")[2] for _m, p in census_routes()}
    bad = []
    for seg in sorted(js):
        if seg not in registered:
            bad.append(f"the frontend fetches /tex_wrangle/{seg}, which no route registers "
                       f"(a 404 in the shipped extension)")
    for (m, p), (_h, who) in sorted(_ROUTES.items()):
        seg = p.split("/")[2]
        if who == "frontend" and seg not in js:
            bad.append(f"{m} {p} is registered as frontend-driven, but the extension never "
                       f"fetches it - either it lost its caller or the registry is wrong")
        if who != "frontend" and seg in js:
            bad.append(f"{m} {p} is registered as having no frontend caller, but the "
                       f"extension fetches it; it no longer needs to be driven by name")
    if bad:
        r.fail("SIMP-3 route/frontend agreement", "\n  ".join(bad))
        return
    r.ok(f"the extension fetches {len(js)} of {len(registered)} registered paths, and the "
         f"registry names the other {len(registered) - len(js)}")
