// SPDX-License-Identifier: Apache-2.0
//
// stats.cpp --- report formatting, both human and machine readable.
//
// The roofline maths is MICROARCH.md §9.2, verbatim.

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <sstream>

#include "kea/isa.h"
#include "kea/sim/stats.h"

namespace kea {
namespace sim {
namespace {

std::string fmt(const char* f, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, f);
  std::vsnprintf(buf, sizeof(buf), f, ap);
  va_end(ap);
  return buf;
}

double pct(std::uint64_t a, std::uint64_t b) {
  return b ? 100.0 * static_cast<double>(a) / static_cast<double>(b) : 0.0;
}

std::string jstr(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    switch (c) {
      case '"':  o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20)
          o += fmt("\\u%04x", static_cast<unsigned>(c) & 0xFF);
        else
          o += c;
    }
  }
  return o + "\"";
}

std::string jnum(double d) {
  if (!std::isfinite(d)) return "null";
  return fmt("%.6g", d);
}

const char* kUnitNames[KEA_NUM_QUEUES] = {"MXU", "DWU", "VPU", "DMA0", "DMA1"};

}  // namespace

Roofline computeRoofline(const Counters& c) {
  Roofline r;
  const double ops_useful = static_cast<double>(c.opsUseful());
  const double ops_issued = static_cast<double>(c.opsIssued());
  const double bytes = static_cast<double>(c.dramBytes());
  const double secs = static_cast<double>(c.cycles) /
                      static_cast<double>(KEA_CLOCK_HZ);

  // The compute roof depends on the datatype mix.  Report the int4 roof only
  // when the majority of useful MXU work actually ran in int4.
  const bool mostly_int4 =
      c.mxu_macs_useful > 0 && c.mxu_macs_useful_int4 * 2 > c.mxu_macs_useful;
  r.peak_ops = mostly_int4 ? KEA_PEAK_INT4_OPS_PER_SEC
                           : KEA_PEAK_INT8_OPS_PER_SEC;

  r.intensity = bytes > 0 ? ops_useful / bytes : 0.0;
  r.achieved = secs > 0 ? ops_useful / secs : 0.0;
  const double bw_roof = r.intensity * KEA_PEAK_DRAM_BYTES_PER_SEC;
  r.attainable = bytes > 0 ? std::min(r.peak_ops, bw_roof) : r.peak_ops;
  r.efficiency = r.attainable > 0 ? r.achieved / r.attainable : 0.0;
  r.padding_loss = ops_issued > 0 ? ops_useful / ops_issued : 0.0;
  r.memory_bound = bytes > 0 && bw_roof < r.peak_ops;
  return r;
}

namespace {

void appendCounterBlock(std::ostringstream& o, const Counters& c,
                        std::uint64_t total_cycles, const char* indent) {
  const Roofline rl = computeRoofline(c);

  o << indent << "cycles                 " << c.cycles;
  if (total_cycles && c.cycles != total_cycles)
    o << fmt("  (%.2f%% of run)", pct(c.cycles, total_cycles));
  o << "\n";

  o << indent << "per-unit (busy / sem-stall / res-stall / idle, cycles)\n";
  for (int u = 0; u < KEA_NUM_QUEUES; ++u) {
    const UnitStats& s = c.unit[u];
    o << indent
      << fmt("  %-4s %9llu %9llu %9llu %9llu   busy %6.2f%%  instrs %llu  "
             "avg-q %.2f  max-q %llu\n",
             kUnitNames[u], (unsigned long long)s.busy_cycles,
             (unsigned long long)s.stall_sem_cycles,
             (unsigned long long)s.stall_res_cycles,
             (unsigned long long)s.idle_cycles, pct(s.busy_cycles, c.cycles),
             (unsigned long long)s.instrs,
             c.cycles ? (double)s.queue_occupancy / (double)c.cycles : 0.0,
             (unsigned long long)s.max_queue_depth);
  }

  o << indent
    << fmt("dispatcher             stalled %llu cycles (%.2f%%), dispatched "
           "%llu\n",
           (unsigned long long)c.dispatch_stall_total,
           pct(c.dispatch_stall_total, c.cycles),
           (unsigned long long)c.dispatched);
  if (c.dispatch_stall_total) {
    o << indent << "  attributed to full queue:";
    for (int u = 0; u < KEA_NUM_QUEUES; ++u)
      if (c.dispatch_stall[u])
        o << fmt(" %s=%llu", kUnitNames[u],
                 (unsigned long long)c.dispatch_stall[u]);
    o << "\n";
  }

  o << indent
    << fmt("MXU                    %llu MATMUL, %llu LOAD_W, useful %llu MAC, "
           "issued %llu MAC\n",
           (unsigned long long)c.mxu_matmuls, (unsigned long long)c.mxu_loadws,
           (unsigned long long)c.mxu_macs_useful,
           (unsigned long long)c.mxu_macs_issued);
  o << indent
    << fmt("  MAC utilisation      %.2f%% of the %d int8 MAC/cycle peak "
           "(issued %.2f%%, padding efficiency %.2f%%)\n",
           pct(c.mxu_macs_useful, c.cycles * KEA_MXU_INT8_MACS_PER_CYCLE),
           KEA_MXU_INT8_MACS_PER_CYCLE,
           pct(c.mxu_macs_issued, c.cycles * KEA_MXU_INT8_MACS_PER_CYCLE),
           pct(c.mxu_macs_useful, c.mxu_macs_issued));
  if (c.mxu_macs_useful_int4)
    o << indent
      << fmt("  of which int4        %llu MAC (peak %d MAC/cycle)\n",
             (unsigned long long)c.mxu_macs_useful_int4,
             KEA_MXU_INT4_MACS_PER_CYCLE);

  o << indent
    << fmt("DWU                    %llu DWCONV, useful %llu MAC, utilisation "
           "%.2f%% of %d MAC/cycle\n",
           (unsigned long long)c.dwu_convs,
           (unsigned long long)c.dwu_macs_useful,
           pct(c.dwu_macs_useful, c.cycles * KEA_DWU_MACS_PER_CYCLE),
           KEA_DWU_MACS_PER_CYCLE);
  o << indent
    << fmt("VPU                    %llu elem (VQUANT %llu, VADD %llu, VPOOL "
           "%llu, VCOPY %llu), utilisation %.2f%% of %d elem/cycle\n",
           (unsigned long long)c.vpu_elems, (unsigned long long)c.vpu_ops[0],
           (unsigned long long)c.vpu_ops[1], (unsigned long long)c.vpu_ops[2],
           (unsigned long long)c.vpu_ops[3],
           pct(c.vpu_elems, c.cycles * KEA_VPU_ELEMS_PER_CYCLE),
           KEA_VPU_ELEMS_PER_CYCLE);

  const double secs =
      static_cast<double>(c.cycles) / static_cast<double>(KEA_CLOCK_HZ);
  for (int e = 0; e < KEA_NUM_DMA_ENGINES; ++e) {
    o << indent
      << fmt("DMA%d                   %llu desc, %llu B, %.3f GB/s achieved, "
             "%llu data cycles (%llu contended)\n",
             e, (unsigned long long)c.dma_descs[e],
             (unsigned long long)c.dma_bytes[e],
             secs > 0 ? static_cast<double>(c.dma_bytes[e]) / secs / 1e9 : 0.0,
             (unsigned long long)c.dram_data_cycles[e],
             (unsigned long long)c.dram_contended_cycles[e]);
  }
  o << indent
    << fmt("DRAM                   %llu B total (%llu load / %llu store), "
           "%.3f GB/s of %.1f GB/s peak, port busy %.2f%%\n",
           (unsigned long long)c.dramBytes(),
           (unsigned long long)c.dram_bytes_load,
           (unsigned long long)c.dram_bytes_store,
           secs > 0 ? static_cast<double>(c.dramBytes()) / secs / 1e9 : 0.0,
           KEA_PEAK_DRAM_BYTES_PER_SEC / 1e9,
           pct(c.dram_busy_cycles, c.cycles));
  o << indent
    << fmt("  lost to port sharing %.1f engine-cycles of DRAM bandwidth\n",
           c.dram_cycles_lost_to_contention);

  o << indent << "roofline\n";
  o << indent
    << fmt("  useful ops           %llu   (issued %llu, padding efficiency "
           "%.2f%%)\n",
           (unsigned long long)c.opsUseful(),
           (unsigned long long)c.opsIssued(), 100.0 * rl.padding_loss);
  o << indent
    << fmt("  arithmetic intensity %.3f ops/DRAM byte  (int8 ridge point "
           "%.1f)\n",
           rl.intensity, KEA_RIDGE_INT8_OPS_PER_BYTE);
  o << indent
    << fmt("  achieved             %.3f GOPS of %.3f GOPS attainable "
           "(%.2f%%); peak %.0f GOPS\n",
           rl.achieved / 1e9, rl.attainable / 1e9, 100.0 * rl.efficiency,
           rl.peak_ops / 1e9);
  o << indent << "  bound                "
    << (rl.memory_bound ? "MEMORY (below the ridge point)"
                        : "COMPUTE (above the ridge point)")
    << "\n";
}

void appendCounterJson(std::ostringstream& o, const Counters& c,
                       const char* indent) {
  const Roofline rl = computeRoofline(c);
  const double secs =
      static_cast<double>(c.cycles) / static_cast<double>(KEA_CLOCK_HZ);

  o << indent << "\"cycles\": " << c.cycles << ",\n";
  o << indent << "\"units\": {\n";
  for (int u = 0; u < KEA_NUM_QUEUES; ++u) {
    const UnitStats& s = c.unit[u];
    o << indent << "  " << jstr(kUnitNames[u]) << ": {"
      << "\"instrs\": " << s.instrs << ", \"occupancy\": " << s.occupancy
      << ", \"busy_cycles\": " << s.busy_cycles
      << ", \"stall_semaphore_cycles\": " << s.stall_sem_cycles
      << ", \"stall_resource_cycles\": " << s.stall_res_cycles
      << ", \"idle_cycles\": " << s.idle_cycles
      << ", \"start_cycles\": " << s.start_cycles
      << ", \"avg_queue_depth\": "
      << jnum(c.cycles ? (double)s.queue_occupancy / (double)c.cycles : 0.0)
      << ", \"max_queue_depth\": " << s.max_queue_depth
      << ", \"utilization\": "
      << jnum(c.cycles ? (double)s.busy_cycles / (double)c.cycles : 0.0)
      << ", \"dispatch_stall_cycles\": " << c.dispatch_stall[u] << "}"
      << (u + 1 < KEA_NUM_QUEUES ? "," : "") << "\n";
  }
  o << indent << "},\n";

  o << indent << "\"dispatcher\": {\"dispatched\": " << c.dispatched
    << ", \"stall_cycles\": " << c.dispatch_stall_total << "},\n";

  o << indent << "\"mxu\": {\"matmuls\": " << c.mxu_matmuls
    << ", \"load_ws\": " << c.mxu_loadws
    << ", \"macs_useful\": " << c.mxu_macs_useful
    << ", \"macs_issued\": " << c.mxu_macs_issued
    << ", \"macs_useful_int4\": " << c.mxu_macs_useful_int4
    << ", \"mac_utilization\": "
    << jnum(c.cycles ? (double)c.mxu_macs_useful /
                           ((double)c.cycles * KEA_MXU_INT8_MACS_PER_CYCLE)
                     : 0.0)
    << ", \"peak_macs_per_cycle\": " << KEA_MXU_INT8_MACS_PER_CYCLE << "},\n";

  o << indent << "\"dwu\": {\"dwconvs\": " << c.dwu_convs
    << ", \"macs_useful\": " << c.dwu_macs_useful
    << ", \"macs_issued\": " << c.dwu_macs_issued
    << ", \"mac_utilization\": "
    << jnum(c.cycles ? (double)c.dwu_macs_useful /
                           ((double)c.cycles * KEA_DWU_MACS_PER_CYCLE)
                     : 0.0)
    << ", \"peak_macs_per_cycle\": " << KEA_DWU_MACS_PER_CYCLE << "},\n";

  o << indent << "\"vpu\": {\"elements\": " << c.vpu_elems
    << ", \"vquant\": " << c.vpu_ops[0] << ", \"vadd\": " << c.vpu_ops[1]
    << ", \"vpool\": " << c.vpu_ops[2] << ", \"vcopy\": " << c.vpu_ops[3]
    << ", \"utilization\": "
    << jnum(c.cycles ? (double)c.vpu_elems /
                           ((double)c.cycles * KEA_VPU_ELEMS_PER_CYCLE)
                     : 0.0)
    << "},\n";

  o << indent << "\"dram\": {\n";
  for (int e = 0; e < KEA_NUM_DMA_ENGINES; ++e) {
    o << indent << "  \"dma" << e << "\": {\"descriptors\": " << c.dma_descs[e]
      << ", \"bytes\": " << c.dma_bytes[e]
      << ", \"gb_per_s\": "
      << jnum(secs > 0 ? (double)c.dma_bytes[e] / secs / 1e9 : 0.0)
      << ", \"data_cycles\": " << c.dram_data_cycles[e]
      << ", \"contended_cycles\": " << c.dram_contended_cycles[e] << "},\n";
  }
  o << indent << "  \"bytes_total\": " << c.dramBytes()
    << ", \"bytes_load\": " << c.dram_bytes_load
    << ", \"bytes_store\": " << c.dram_bytes_store
    << ", \"gb_per_s\": "
    << jnum(secs > 0 ? (double)c.dramBytes() / secs / 1e9 : 0.0)
    << ", \"peak_gb_per_s\": " << jnum(KEA_PEAK_DRAM_BYTES_PER_SEC / 1e9)
    << ", \"port_busy_cycles\": " << c.dram_busy_cycles
    << ", \"engine_cycles_lost_to_contention\": "
    << jnum(c.dram_cycles_lost_to_contention) << "\n";
  o << indent << "},\n";

  o << indent << "\"roofline\": {\"ops_useful\": " << c.opsUseful()
    << ", \"ops_issued\": " << c.opsIssued()
    << ", \"dram_bytes\": " << c.dramBytes()
    << ", \"intensity_ops_per_byte\": " << jnum(rl.intensity)
    << ", \"achieved_ops_per_s\": " << jnum(rl.achieved)
    << ", \"attainable_ops_per_s\": " << jnum(rl.attainable)
    << ", \"peak_ops_per_s\": " << jnum(rl.peak_ops)
    << ", \"efficiency\": " << jnum(rl.efficiency)
    << ", \"padding_efficiency\": " << jnum(rl.padding_loss)
    << ", \"ridge_point_ops_per_byte\": "
    << jnum(KEA_RIDGE_INT8_OPS_PER_BYTE) << ", \"memory_bound\": "
    << (rl.memory_bound ? "true" : "false") << "}\n";
}

}  // namespace

std::string formatReport(const Stats& s, std::uint64_t total_cycles) {
  std::ostringstream o;
  o << "=== KEA-1 simulation report (" << KEA_ARCH_NAME << ", rev "
    << KEA_ISA_REVISION << ") ===\n";
  o << fmt("total cycles           %llu  (%.6f ms at %.2f GHz)\n",
           (unsigned long long)total_cycles,
           1e3 * static_cast<double>(total_cycles) /
               static_cast<double>(KEA_CLOCK_HZ),
           static_cast<double>(KEA_CLOCK_HZ) / 1e9);
  o << "\n-- whole program --\n";
  appendCounterBlock(o, s.global, total_cycles, "  ");

  if (!s.regions.empty()) {
    o << "\n-- TRACE regions --\n";
    for (const Region& r : s.regions) {
      const Roofline rl = computeRoofline(r.c);
      o << fmt("\n[%s tag=%u payload=%u depth=%d] cycles %llu..%llu (%llu)\n",
               unitName(static_cast<Unit>(r.unit)), r.tag, r.payload, r.depth,
               (unsigned long long)r.begin, (unsigned long long)r.end,
               (unsigned long long)r.c.cycles);
      o << fmt("  intensity %.3f ops/B, achieved %.3f GOPS, %.2f%% of "
               "attainable, %s bound\n",
               rl.intensity, rl.achieved / 1e9, 100.0 * rl.efficiency,
               rl.memory_bound ? "MEMORY" : "COMPUTE");
      appendCounterBlock(o, r.c, total_cycles, "    ");
    }
  }

  if (!s.markers.empty()) {
    o << "\n-- TRACE markers --\n";
    for (const Marker& m : s.markers)
      o << fmt("  cycle %llu  %-4s tag=%u payload=%u\n",
               (unsigned long long)m.cycle,
               unitName(static_cast<Unit>(m.unit)), m.tag, m.payload);
  }
  return o.str();
}

std::string formatJson(const Stats& s, std::uint64_t total_cycles,
                       const std::string& status, std::uint32_t exit_code,
                       const std::vector<std::string>& diagnostics) {
  std::ostringstream o;
  o << "{\n";
  o << "  \"format\": \"kea.sim.stats\",\n";
  o << "  \"version\": 1,\n";
  o << "  \"arch\": " << jstr(KEA_ARCH_NAME) << ",\n";
  o << "  \"isa_revision\": " << KEA_ISA_REVISION << ",\n";
  o << "  \"clock_hz\": " << KEA_CLOCK_HZ << ",\n";
  o << "  \"status\": " << jstr(status) << ",\n";
  o << "  \"exit_code\": " << exit_code << ",\n";
  o << "  \"total_cycles\": " << total_cycles << ",\n";
  o << "  \"seconds\": "
    << jnum(static_cast<double>(total_cycles) /
            static_cast<double>(KEA_CLOCK_HZ))
    << ",\n";

  o << "  \"diagnostics\": [";
  for (std::size_t i = 0; i < diagnostics.size(); ++i)
    o << (i ? ", " : "") << jstr(diagnostics[i]);
  o << "],\n";

  o << "  \"global\": {\n";
  appendCounterJson(o, s.global, "    ");
  o << "  },\n";

  o << "  \"regions\": [\n";
  for (std::size_t i = 0; i < s.regions.size(); ++i) {
    const Region& r = s.regions[i];
    o << "    {\n";
    o << "      \"tag\": " << r.tag << ",\n";
    o << "      \"payload\": " << r.payload << ",\n";
    o << "      \"unit\": " << jstr(unitName(static_cast<Unit>(r.unit)))
      << ",\n";
    o << "      \"depth\": " << r.depth << ",\n";
    o << "      \"begin_cycle\": " << r.begin << ",\n";
    o << "      \"end_cycle\": " << r.end << ",\n";
    o << "      \"closed\": " << (r.closed ? "true" : "false") << ",\n";
    appendCounterJson(o, r.c, "      ");
    o << "    }" << (i + 1 < s.regions.size() ? "," : "") << "\n";
  }
  o << "  ],\n";

  o << "  \"markers\": [";
  for (std::size_t i = 0; i < s.markers.size(); ++i) {
    const Marker& m = s.markers[i];
    o << (i ? ", " : "") << "{\"cycle\": " << m.cycle << ", \"unit\": "
      << jstr(unitName(static_cast<Unit>(m.unit))) << ", \"tag\": " << m.tag
      << ", \"payload\": " << m.payload << "}";
  }
  o << "]\n";
  o << "}\n";
  return o.str();
}

}  // namespace sim
}  // namespace kea
