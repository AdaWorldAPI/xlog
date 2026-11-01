//! Print the distinct shapes of the `MultiWayJoin` bodies XLOG compiles from
//! its integration programs: relation ids are replaced by arities.
use std::collections::BTreeMap;
use xlog_ir::RirNode;
use xlog_logic::Compiler;

fn main() {
    let mut shapes: BTreeMap<String, usize> = BTreeMap::new();
    for path in std::env::args().skip(1) {
        let text = std::fs::read_to_string(&path).unwrap();
        for lit in text
            .split("r#\"")
            .skip(1)
            .filter_map(|t| t.split("\"#").next())
        {
            let mut c = Compiler::new();
            let Ok(plan) = c.compile(lit) else { continue };
            for rules in &plan.rules_by_scc {
                for r in rules {
                    if let RirNode::MultiWayJoin {
                        inputs,
                        slot_vars,
                        output_columns,
                        ..
                    } = &r.body
                    {
                        let kinds: std::collections::BTreeSet<String> = inputs
                            .iter()
                            .map(|i| {
                                format!("{:?}", i)
                                    .split([' ', '{'])
                                    .next()
                                    .unwrap()
                                    .to_string()
                            })
                            .collect();
                        let arities: std::collections::BTreeSet<usize> =
                            slot_vars.iter().map(|v| v.len()).collect();
                        let none = slot_vars.iter().flatten().any(|v| v.is_none());
                        let self_loop = slot_vars
                            .iter()
                            .any(|v| v.len() == 2 && v[0].is_some() && v[0] == v[1]);
                        let computed = output_columns
                            .iter()
                            .any(|c| !matches!(c, xlog_ir::ProjectExpr::Column(_)));
                        let s = format!(
                            "atoms {:>2} kinds {:?} arities {:?} unbound_slot {} self_loop {} computed_out {}",
                            inputs.len(), kinds, arities, none, self_loop, computed
                        );
                        *shapes.entry(s).or_default() += 1;
                    }
                }
            }
        }
    }
    println!("distinct bodies {}", shapes.len());
    for (s, n) in &shapes {
        println!("{n:3} × {s}");
    }
}
