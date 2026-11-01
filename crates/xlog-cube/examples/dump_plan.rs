use xlog_logic::Compiler;
fn main() {
    let src = std::env::args().nth(1).expect("program");
    let mut c = Compiler::new();
    let plan = c.compile(&src).expect("compile");
    println!("rel_ids: {:?}", c.rel_ids());
    println!("arities: {:?}", plan.rel_arities);
    for s in &plan.strata {
        println!("stratum {:?}", s);
    }
    for (i, scc) in plan.sccs.iter().enumerate() {
        println!("scc {i}: {:?}", scc);
        for r in &plan.rules_by_scc[i] {
            println!("  {} :- {:#?}", r.head, r.body);
        }
    }
}
