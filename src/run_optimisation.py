# src/run_optimisation.py
# Min-cost shipment plan using PuLP.
import sys, pandas as pd
from pathlib import Path
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value

# --- Expected CSV long-form columns ---
# supplier, warehouse, cost, capacity, demand
# If your column names differ, change them here:
SUPPLIER_COL = "supplier"
WAREHOUSE_COL = "warehouse"
COST_COL     = "cost"
CAP_COL      = "capacity"
DEM_COL      = "demand"
DATA_FILE    = Path("data/supply_chain_data.csv")
OUTDIR       = Path("outputs"); OUTDIR.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA_FILE)
    need = {w: d for w, d in df.groupby(WAREHOUSE_COL)[DEM_COL].max().items()}
    cap  = {s: c for s, c in df.groupby(SUPPLIER_COL)[CAP_COL].max().items()}
    cost = {(r[SUPPLIER_COL], r[WAREHOUSE_COL]): float(r[COST_COL]) for _, r in df.iterrows()}

    prob = LpProblem("MinCostShipping", LpMinimize)
    # decision vars
    x = {(s,w): LpVariable(f"x_{s}_{w}", lowBound=0) for (s,w) in cost}
    # objective
    prob += lpSum(cost[s,w] * x[s,w] for (s,w) in cost)
    # capacity per supplier
    for s in cap:
        prob += lpSum(x[s,w] for w in need if (s,w) in x) <= cap[s]
    # demand per warehouse
    for w in need:
        prob += lpSum(x[s,w] for s in cap if (s,w) in x) >= need[w]

    prob.solve()

    status = LpStatus[prob.status]
    alloc = []
    for (s,w), var in x.items():
        qty = var.value()
        if qty and qty > 0:
            alloc.append({"supplier": s, "warehouse": w, "qty": qty, "unit_cost": cost[(s,w)], "ship_cost": qty*cost[(s,w)]})

    out_alloc = pd.DataFrame(alloc).sort_values(["supplier","warehouse"])
    total_cost = value(prob.objective)
    out_alloc.to_csv(OUTDIR/"shipment_plan.csv", index=False)
    with open(OUTDIR/"optimisation_summary.txt","w") as f:
        f.write(f"Status: {status}\nTotal cost: {total_cost:,.2f}\n")
    print(f"Status: {status} | Total cost: {total_cost:,.2f}")
    print(f"Saved: outputs/shipment_plan.csv and outputs/optimisation_summary.txt")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.stderr.write(f"\nError: {e}\nCheck column names at top of file and your CSV schema.\n")
        sys.exit(1)
