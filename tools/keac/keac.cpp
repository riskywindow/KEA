// SPDX-License-Identifier: Apache-2.0
//
// keac --- the KEA compiler driver.
//
// ADR-0001 draws the pipeline and calls this "a thin driver that runs it end to
// end". That is exactly what this is: it chains four processes and does no
// compilation of its own.
//
//     model.tosa.mlir
//         |  kea-opt   --tosa-to-kea --kea-fuse --kea-tile [--kea-schedule] --kea-alloc
//         v
//     model.l2.mlir
//         |  kea-translate
//         v
//     model.kasm + model.weights.bin + model.map.json
//         |  kea-as
//         v
//     model.keaf                                  --> kea-rt / kea-sim
//
// It lives in the NATIVE half of the tree, next to kea-as, because it links
// nothing from either half -- it only spawns them. The MLIR tools are found by
// flag, by environment variable, next to this binary, or in the sibling
// build/compiler/bin, and a failure to find one says all four.

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

const char *kUsage =
    "usage: keac <model.mlir> [-o <model.keaf>] [options]\n"
    "\n"
    "  Compiles a TOSA (or Level 1 kea) MLIR module into a KEAF artifact by\n"
    "  running kea-opt, kea-translate and kea-as in turn.\n"
    "\n"
    "  -o, --output <file>     artifact to write (default: input with .keaf)\n"
    "      --function <name>   which function to compile; required when the\n"
    "                          module has more than one\n"
    "      --keep-intermediates\n"
    "                          keep <base>.l2.mlir, <base>.kasm,\n"
    "                          <base>.weights.bin and <base>.map.json\n"
    "      --emit <stage>      stop after mlir | kasm | keaf (default keaf)\n"
    "      --pipeline <passes> override the kea-opt pass list\n"
    "      --schedule          insert --kea-schedule into the pipeline\n"
    "      --spm-reserve <n>   -kea-tile=spm-reserve-factor=<n>\n"
    "      --imem-budget <n>   -kea-tile=imem-budget=<n>: the whole-function\n"
    "                          instruction budget the tiler plans against\n"
    "                          (default 20480 of the 32768 IMEM holds)\n"
    "      --sync <mode>       kea-translate --sync=auto|insert|none\n"
    "      --io-quant <spec>   <tensor>=<scale>[,<zp>], repeatable\n"
    "      --annotate          annotate the .kasm (implies keeping it)\n"
    "      --strip             kea-as --strip: drop the debug sections\n"
    "      --kea-opt <path>    override a tool location (also --kea-translate,\n"
    "                          --kea-as); or set $KEA_OPT / $KEA_TRANSLATE /\n"
    "                          $KEA_AS\n"
    "  -v, --verbose           echo every command\n"
    "  -h, --help\n";

bool verbose = false;

bool isFile(const std::string &p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0 && (st.st_mode & S_IFREG) &&
         ::access(p.c_str(), X_OK) == 0;
}

std::string dirOf(const std::string &p) {
  size_t s = p.find_last_of('/');
  return s == std::string::npos ? std::string(".") : p.substr(0, s);
}

/// Where this binary lives, so the sibling compiler build can be found without
/// anyone having to set PATH.
std::string selfDir(const char *argv0) {
  std::string a(argv0);
  if (a.find('/') != std::string::npos)
    return dirOf(a);
  const char *path = ::getenv("PATH");
  if (path)
    for (const char *p = path; *p;) {
      const char *e = std::strchr(p, ':');
      std::string dir(p, e ? size_t(e - p) : std::strlen(p));
      if (!dir.empty() && isFile(dir + "/" + a))
        return dir;
      if (!e)
        break;
      p = e + 1;
    }
  return ".";
}

std::string findTool(const std::string &name, const std::string &override_,
                     const char *envVar, const std::string &self) {
  if (!override_.empty())
    return override_;
  if (const char *e = ::getenv(envVar))
    if (*e)
      return e;
  // Next to keac, then the sibling MLIR build, then two common relative spots,
  // then PATH.
  const std::string candidates[] = {
      self + "/" + name,
      self + "/../../compiler/bin/" + name,   // build/native/bin -> build/compiler/bin
      self + "/../../../build/compiler/bin/" + name,
      "build/compiler/bin/" + name,
  };
  for (const std::string &c : candidates)
    if (isFile(c))
      return c;
  return name; // let execvp search PATH, and report if it cannot
}

int runCommand(const std::vector<std::string> &argv) {
  if (verbose) {
    std::fprintf(stderr, "+");
    for (const std::string &a : argv)
      std::fprintf(stderr, " %s", a.c_str());
    std::fprintf(stderr, "\n");
  }
  std::vector<char *> cargv;
  for (const std::string &a : argv)
    cargv.push_back(const_cast<char *>(a.c_str()));
  cargv.push_back(nullptr);

  pid_t pid = ::fork();
  if (pid < 0) {
    std::fprintf(stderr, "keac: fork: %s\n", std::strerror(errno));
    return 1;
  }
  if (pid == 0) {
    ::execvp(cargv[0], cargv.data());
    std::fprintf(stderr, "keac: cannot run '%s': %s\n", cargv[0],
                 std::strerror(errno));
    ::_exit(127);
  }
  int status = 0;
  if (::waitpid(pid, &status, 0) < 0) {
    std::fprintf(stderr, "keac: waitpid: %s\n", std::strerror(errno));
    return 1;
  }
  if (WIFSIGNALED(status)) {
    std::fprintf(stderr, "keac: %s died on signal %d\n", cargv[0],
                 WTERMSIG(status));
    return 1;
  }
  return WEXITSTATUS(status);
}

std::string stripSuffix(std::string s, const char *suffix) {
  size_t n = std::strlen(suffix);
  if (s.size() > n && s.compare(s.size() - n, n, suffix) == 0)
    s.resize(s.size() - n);
  return s;
}

bool endsWith(const std::string &s, const char *suffix) {
  size_t n = std::strlen(suffix);
  return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

} // namespace

int main(int argc, char **argv) {
  std::string input, output, function, pipelineOverride, emit = "keaf";
  std::string syncMode, spmReserve, imemBudget;
  std::string optPath, translatePath, asPath;
  std::vector<std::string> ioQuant;
  bool keep = false, schedule = false, annotate = false, strip = false;

  auto needValue = [&](int &i) -> std::string {
    if (i + 1 >= argc) {
      std::fprintf(stderr, "keac: %s needs a value\n", argv[i]);
      std::exit(2);
    }
    return argv[++i];
  };

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      std::fputs(kUsage, stdout);
      return 0;
    } else if (a == "-o" || a == "--output") {
      output = needValue(i);
    } else if (a == "--function") {
      function = needValue(i);
    } else if (a == "--keep-intermediates") {
      keep = true;
    } else if (a == "--emit") {
      emit = needValue(i);
    } else if (a == "--pipeline") {
      pipelineOverride = needValue(i);
    } else if (a == "--schedule") {
      schedule = true;
    } else if (a == "--spm-reserve") {
      spmReserve = needValue(i);
    } else if (a == "--imem-budget") {
      imemBudget = needValue(i);
    } else if (a == "--sync") {
      syncMode = needValue(i);
    } else if (a == "--io-quant") {
      ioQuant.push_back(needValue(i));
    } else if (a == "--annotate") {
      annotate = true;
      keep = true;
    } else if (a == "--strip") {
      strip = true;
    } else if (a == "--kea-opt") {
      optPath = needValue(i);
    } else if (a == "--kea-translate") {
      translatePath = needValue(i);
    } else if (a == "--kea-as") {
      asPath = needValue(i);
    } else if (a == "-v" || a == "--verbose") {
      verbose = true;
    } else if (!a.empty() && a[0] == '-' && a != "-") {
      std::fprintf(stderr, "keac: unknown option '%s'\n%s", a.c_str(), kUsage);
      return 2;
    } else if (input.empty()) {
      input = a;
    } else {
      std::fprintf(stderr, "keac: more than one input file ('%s' and '%s')\n",
                   input.c_str(), a.c_str());
      return 2;
    }
  }

  if (input.empty()) {
    std::fputs(kUsage, stderr);
    return 2;
  }
  if (emit != "mlir" && emit != "kasm" && emit != "keaf") {
    std::fprintf(stderr, "keac: --emit takes mlir, kasm or keaf, not '%s'\n",
                 emit.c_str());
    return 2;
  }

  if (endsWith(input, ".kgraph.json") || endsWith(input, ".json")) {
    std::fprintf(
        stderr,
        "keac: '%s' looks like a .kgraph.json. keac compiles TOSA MLIR; run\n"
        "      the frontend's emitter first (docs/FRONTEND.md section 5):\n"
        "\n"
        "        cd frontend && ../.venv/bin/python -m kea_frontend.tosa_emit \\\n"
        "            %s -o model.tosa.mlir --function model\n"
        "\n"
        "      then pass model.tosa.mlir to keac. Use --first-index/--last-index\n"
        "      to emit a contiguous node slice as one function.\n",
        input.c_str(), input.c_str());
    return 2;
  }

  const std::string self = selfDir(argv[0]);
  const std::string keaOpt = findTool("kea-opt", optPath, "KEA_OPT", self);
  const std::string keaTranslate =
      findTool("kea-translate", translatePath, "KEA_TRANSLATE", self);
  const std::string keaAs = findTool("kea-as", asPath, "KEA_AS", self);

  if (output.empty()) {
    std::string b = input;
    for (const char *s : {".tosa.mlir", ".mlir"})
      b = stripSuffix(b, s);
    output = b + ".keaf";
  }
  const std::string base = stripSuffix(output, ".keaf");
  const std::string l2 = base + ".l2.mlir";
  const std::string kasm = base + ".kasm";
  const std::string blob = base + ".weights.bin";
  const std::string map = base + ".map.json";

  //--- 1. kea-opt ----------------------------------------------------------
  std::vector<std::string> cmd{keaOpt, input};
  if (!pipelineOverride.empty()) {
    // Split on whitespace so `--pipeline "--kea-tile --kea-alloc"` works.
    std::string tok;
    for (char c : pipelineOverride + " ") {
      if (std::isspace(static_cast<unsigned char>(c))) {
        if (!tok.empty())
          cmd.push_back(tok);
        tok.clear();
      } else {
        tok += c;
      }
    }
  } else {
    cmd.push_back("--tosa-to-kea");
    cmd.push_back("--kea-fuse");
    // `-pass=a=1 b=2` -- MLIR splits a single pass argument on whitespace, and
    // runCommand passes it as one argv element, so the spaces survive.
    std::string tileOpts;
    if (!spmReserve.empty())
      tileOpts = "spm-reserve-factor=" + spmReserve;
    if (!imemBudget.empty())
      tileOpts += (tileOpts.empty() ? "" : " ") + ("imem-budget=" + imemBudget);
    cmd.push_back(tileOpts.empty() ? std::string("--kea-tile")
                                   : "--kea-tile=" + tileOpts);
    if (schedule)
      cmd.push_back("--kea-schedule");
    cmd.push_back("--kea-alloc");
  }
  cmd.push_back("-o");
  cmd.push_back(l2);
  if (int rc = runCommand(cmd)) {
    std::fprintf(stderr, "keac: kea-opt failed (exit %d)\n", rc);
    return rc;
  }
  if (emit == "mlir") {
    std::fprintf(stderr, "keac: wrote %s\n", l2.c_str());
    return 0;
  }

  //--- 2. kea-translate ----------------------------------------------------
  cmd = {keaTranslate, l2, "-o", base};
  if (!function.empty())
    cmd.push_back("--function=" + function);
  if (!syncMode.empty())
    cmd.push_back("--sync=" + syncMode);
  if (annotate)
    cmd.push_back("--annotate");
  for (const std::string &q : ioQuant)
    cmd.push_back("--io-quant=" + q);
  if (verbose)
    cmd.push_back("-v");
  if (int rc = runCommand(cmd)) {
    std::fprintf(stderr, "keac: kea-translate failed (exit %d)\n", rc);
    return rc;
  }
  if (emit == "kasm") {
    std::fprintf(stderr, "keac: wrote %s, %s and %s\n", kasm.c_str(),
                 blob.c_str(), map.c_str());
    if (!keep)
      ::remove(l2.c_str());
    return 0;
  }

  //--- 3. kea-as -----------------------------------------------------------
  cmd = {keaAs, kasm, "--map", map, "--const", blob, "-o", output};
  if (strip)
    cmd.push_back("--strip");
  if (int rc = runCommand(cmd)) {
    std::fprintf(stderr,
                 "keac: kea-as rejected the compiler's own output (exit %d).\n"
                 "      That is a backend bug, not a tooling problem; the\n"
                 "      assembly is in %s\n",
                 rc, kasm.c_str());
    return rc;
  }

  if (!keep) {
    ::remove(l2.c_str());
    ::remove(kasm.c_str());
    ::remove(blob.c_str());
    ::remove(map.c_str());
  } else if (verbose) {
    std::fprintf(stderr, "keac: kept %s %s %s %s\n", l2.c_str(), kasm.c_str(),
                 blob.c_str(), map.c_str());
  }
  return 0;
}
