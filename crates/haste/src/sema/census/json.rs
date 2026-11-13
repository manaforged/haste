use super::{CodeSavings, HazardCauses, Histogram, OriginReport, Report, Span};

struct Json {
    buf: String,
}

impl Json {
    fn new() -> Self {
        Self { buf: String::with_capacity(4096) }
    }

    fn line(&mut self, text: &str) {
        self.buf.push_str(text);
        self.buf.push('\n');
    }

    fn raw(&mut self, indent: &str, key: &str, value: &str, comma: bool) {
        self.buf.push_str(indent);
        self.buf.push('"');
        self.buf.push_str(key);
        self.buf.push_str("\": ");
        self.buf.push_str(value);
        if comma {
            self.buf.push(',');
        }
        self.buf.push('\n');
    }

    fn num(&mut self, indent: &str, key: &str, value: u64, comma: bool) {
        self.raw(indent, key, &value.to_string(), comma);
    }

    fn count(&mut self, indent: &str, key: &str, value: usize, comma: bool) {
        self.num(indent, key, value as u64, comma);
    }

    fn f4(&mut self, indent: &str, key: &str, value: f64, comma: bool) {
        self.raw(indent, key, &format!("{value:.4}"), comma);
    }

    fn f3(&mut self, indent: &str, key: &str, value: f64, comma: bool) {
        self.raw(indent, key, &format!("{value:.3}"), comma);
    }
}

pub(super) fn render_json(report: &Report) -> String {
    let mut out = Json::new();
    out.line("{");
    write_top_counts(&mut out, report);
    write_hazards(&mut out, report.n_naive_shape, report.n_naive_shape_hazards, &report.hazard_causes);
    write_footprints_and_ratios(&mut out, report);
    write_histogram(&mut out, &report.histogram);
    write_savings(&mut out, &report.savings);
    write_stages(&mut out, report);
    write_origins(&mut out, report);
    out.line("}");
    out.buf
}

fn ratio(num: usize, den: usize) -> f64 {
    if den == 0 { 0.0 } else { (num as f64) / (den as f64) }
}

fn pct(part: u64, whole: u64) -> f64 {
    if whole == 0 { 0.0 } else { 100.0 * (part as f64) / (whole as f64) }
}

fn ms(ns: u64) -> f64 {
    (ns as f64) / 1_000_000.0
}

fn write_top_counts(out: &mut Json, r: &Report) {
    for (key, val) in [
        ("n_reached", r.n_reached),
        ("n_inst", r.n_inst),
        ("n_fn", r.n_fn),
        ("n_generic_inst", r.n_generic_inst),
        ("n_generic_fn", r.n_generic_fn),
        ("n_compiled_inst", r.n_compiled_inst),
        ("n_foreign", r.n_foreign),
        ("n_failed", r.n_failed),
    ] {
        out.count("  ", key, val, true);
    }
}

fn write_hazards(out: &mut Json, n_naive_shape: usize, n_hazards: usize, c: &HazardCauses) {
    out.count("  ", "n_naive_shape", n_naive_shape, true);
    out.count("  ", "n_naive_shape_hazards", n_hazards, true);
    out.line("  \"naive_shape_hazard_causes\": {");
    out.count("    ", "niche_layout_mismatch", c.niche_layout_mismatch, true);
    out.count("    ", "field_offset_mismatch", c.field_offset_mismatch, true);
    out.count("    ", "assoc_abi_mismatch", c.assoc_abi_mismatch, true);
    out.count("    ", "specialization_mismatch", c.specialization_mismatch, true);
    out.count("    ", "reflection_mismatch", c.reflection_mismatch, true);
    out.count("    ", "other_footprint_mismatch", c.other_footprint_mismatch, false);
    out.line("  },");
}

fn write_footprints_and_ratios(out: &mut Json, r: &Report) {
    let splits = r.n_footprint.saturating_sub(r.n_footprint_no_inline);
    let gen_splits = r.n_generic_footprint.saturating_sub(r.n_generic_footprint_no_inline);
    for (key, val) in [
        ("n_footprint_no_inline", r.n_footprint_no_inline),
        ("n_footprint", r.n_footprint),
        ("n_inlined_splits", splits),
        ("n_alias", r.n_alias),
        ("n_generic_footprint_no_inline", r.n_generic_footprint_no_inline),
        ("n_generic_footprint", r.n_generic_footprint),
        ("n_generic_inlined_splits", gen_splits),
        ("n_generic_alias", r.n_generic_alias),
    ] {
        out.count("  ", key, val, true);
    }
    out.line("  \"ratios\": {");
    out.f4("    ", "inst_per_fn", ratio(r.n_inst, r.n_fn), true);
    out.f4("    ", "inst_per_footprint_no_inline", ratio(r.n_inst, r.n_footprint_no_inline), true);
    out.f4("    ", "inst_per_footprint", ratio(r.n_inst, r.n_footprint), true);
    out.f4("    ", "inst_per_alias", ratio(r.n_inst, r.n_alias), true);
    out.f4("    ", "generic_inst_per_generic_fn", ratio(r.n_generic_inst, r.n_generic_fn), true);
    out.f4("    ", "generic_inst_per_generic_footprint", ratio(r.n_generic_inst, r.n_generic_footprint), true);
    out.f4("    ", "inlined_split_pct", pct(splits as u64, r.n_footprint_no_inline as u64), false);
    out.line("  },");
}

fn write_histogram(out: &mut Json, h: &Histogram) {
    out.line("  \"code_size_histogram\": {");
    out.count("    ", "le_128", h.le_128, true);
    out.count("    ", "le_256", h.le_256, true);
    out.count("    ", "le_512", h.le_512, true);
    out.count("    ", "le_1024", h.le_1024, true);
    out.count("    ", "gt_1024", h.gt_1024, false);
    out.line("  },");
}

fn write_savings(out: &mut Json, s: &CodeSavings) {
    let alias_saved = s.total_code_bytes.saturating_sub(s.unique_alias_code_bytes);
    let projected_saved = s.total_code_bytes.saturating_sub(s.projected_stenciled_bytes);
    out.line("  \"code_size_savings\": {");
    out.count("    ", "compiled_functions", s.compiled_functions, true);
    out.num("    ", "total_code_bytes", s.total_code_bytes, true);
    out.num("    ", "unique_footprint_code_bytes", s.unique_footprint_code_bytes, true);
    out.num("    ", "unique_alias_code_bytes", s.unique_alias_code_bytes, true);
    out.num("    ", "alias_saved_bytes", alias_saved, true);
    out.num("    ", "copy_patch_code_bytes", s.unique_alias_code_bytes, true);
    out.count("    ", "dict_stencil_classes", s.dict_stencil_classes, true);
    out.num("    ", "projected_stenciled_bytes", s.projected_stenciled_bytes, true);
    out.num("    ", "projected_saved_bytes", projected_saved, true);
    out.f4("    ", "projected_saved_pct", pct(projected_saved, s.total_code_bytes), false);
    out.line("  },");
}

fn write_span(out: &mut Json, label: &str, span: Span, comma: bool) {
    let val = format!("{{ \"wall_ms\": {:.3}, \"cpu_ms\": {:.3} }}", ms(span.wall_ns), ms(span.cpu_ns));
    out.raw("    ", label, &val, comma);
}

fn write_stages(out: &mut Json, r: &Report) {
    let t = r.timings;
    let total = Span {
        wall_ns: t.graph.wall_ns.saturating_add(t.outline.wall_ns).saturating_add(t.reach.wall_ns).saturating_add(t.codegen.wall_ns),
        cpu_ns: t.graph.cpu_ns.saturating_add(t.outline.cpu_ns).saturating_add(t.reach.cpu_ns).saturating_add(t.codegen.cpu_ns),
    };
    out.line("  \"stages\": {");
    write_span(out, "graph", t.graph, true);
    write_span(out, "outline", t.outline, true);
    write_span(out, "reach", t.reach, true);
    write_span(out, "codegen", t.codegen, true);
    write_span(out, "census", r.census_span, true);
    write_span(out, "build_total", total, false);
    out.line("  },");
}

fn write_origins(out: &mut Json, r: &Report) {
    out.line("  \"by_origin\": {");
    write_origin_entry(out, "local", &r.origins[0], true);
    write_origin_entry(out, "pinned", &r.origins[1], true);
    write_origin_entry(out, "sysroot", &r.origins[2], false);
    out.line("  }");
}

fn write_origin_entry(out: &mut Json, label: &str, o: &OriginReport, comma: bool) {
    let splits = o.n_footprint.saturating_sub(o.n_footprint_no_inline);
    let saved = o.savings.total_code_bytes.saturating_sub(o.savings.projected_stenciled_bytes);
    let total_cpu_ns = o.outline_span.cpu_ns.saturating_add(o.reach_span.cpu_ns).saturating_add(o.codegen_span.cpu_ns);
    let total_wall_ns = o.outline_span.wall_ns.saturating_add(o.reach_span.wall_ns).saturating_add(o.codegen_span.wall_ns);
    out.line(&format!("    \"{label}\": {{"));
    for (key, val) in [
        ("crates", o.crates),
        ("loaded_modules", o.loaded_modules),
        ("files", o.files),
        ("items", o.items),
        ("expansions", o.expansions),
        ("n_reached", o.n_reached),
        ("n_inst", o.n_inst),
        ("n_fn", o.n_fn),
        ("n_generic_inst", o.n_generic_inst),
        ("n_generic_fn", o.n_generic_fn),
        ("n_foreign", o.n_foreign),
        ("n_failed", o.n_failed),
        ("n_footprint_no_inline", o.n_footprint_no_inline),
        ("n_footprint", o.n_footprint),
        ("n_inlined_splits", splits),
        ("n_alias", o.n_alias),
        ("n_naive_shape", o.n_naive_shape),
        ("n_naive_shape_hazards", o.n_naive_shape_hazards),
        ("compiled_functions", o.savings.compiled_functions),
    ] {
        out.count("      ", key, val, true);
    }
    for (key, val) in [
        ("source_bytes", o.source_bytes),
        ("inferred_exprs", o.inferred_exprs),
        ("total_code_bytes", o.savings.total_code_bytes),
        ("unique_footprint_code_bytes", o.savings.unique_footprint_code_bytes),
        ("unique_alias_code_bytes", o.savings.unique_alias_code_bytes),
        ("projected_stenciled_bytes", o.savings.projected_stenciled_bytes),
        ("projected_saved_bytes", saved),
    ] {
        out.num("      ", key, val, true);
    }
    for (key, ns, more) in [
        ("codegen_worker_ms", o.codegen_worker_ns, true),
        ("outline_cpu_ms", o.outline_span.cpu_ns, true),
        ("reach_cpu_ms", o.reach_span.cpu_ns, true),
        ("codegen_cpu_ms", o.codegen_span.cpu_ns, true),
        ("total_cpu_ms", total_cpu_ns, true),
        ("outline_wall_ms", o.outline_span.wall_ns, true),
        ("reach_wall_ms", o.reach_span.wall_ns, true),
        ("codegen_wall_ms", o.codegen_span.wall_ns, true),
        ("total_wall_ms", total_wall_ns, false),
    ] {
        out.f3("      ", key, ms(ns), more);
    }
    out.line(if comma { "    }," } else { "    }" });
}
