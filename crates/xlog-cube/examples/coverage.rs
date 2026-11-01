//! Which rules of XLOG's own integration programs the cube admits.
//! Usage: coverage <file.rs>... ; every `r#"..."#` literal that compiles is a
//! program. Base relations of arity 2 are given as empty relations.
use std::collections::BTreeMap;
use xlog_cube::CubeExecutor;
use xlog_logic::Compiler;

fn main() {
    let (mut programs, mut rules, mut ran) = (0, 0, 0);
    let mut why: BTreeMap<String, usize> = BTreeMap::new();
    for path in std::env::args().skip(1) {
        let text = std::fs::read_to_string(&path).unwrap();
        for lit in text
            .split("r#\"")
            .skip(1)
            .filter_map(|t| t.split("\"#").next())
        {
            let mut c = Compiler::new();
            let Ok(plan) = c.compile(lit) else { continue };
            programs += 1;
            let mut ex = CubeExecutor::new();
            for (name, id) in c.rel_ids() {
                ex.register_relation(*id, name);
                let derived = plan
                    .sccs
                    .iter()
                    .zip(&plan.rules_by_scc)
                    .any(|(s, r)| !r.is_empty() && s.predicates.contains(name));
                if !derived && plan.rel_arities.get(id) == Some(&2) {
                    ex.put_relation(name, &[]);
                }
            }
            let Ok(report) = ex.execute_plan(&plan) else {
                continue;
            };
            rules += report.ran.len() + report.refused.len();
            ran += report.ran.len();
            for (_, w) in report.refused {
                let key = if w.contains("is not a binary relation in the cube") {
                    "reads a relation that is not binary".into()
                } else {
                    w
                };
                *why.entry(key).or_default() += 1;
            }
        }
    }
    println!(
        "programs {programs}, rules {rules}, admitted {ran}, refused {}",
        rules - ran
    );
    let mut why: Vec<_> = why.into_iter().collect();
    why.sort_by_key(|w| std::cmp::Reverse(w.1));
    for (w, n) in why {
        println!("  {n:4}  {w}");
    }
}
