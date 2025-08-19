# src/run_optimisation.py
import sys, pandas as pd
from pathlib import Path
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value

DATA_FILE = Path("data/supply_chain_data.csv")
OUTDIR    = Path("outputs"); OUTDIR.mkdir(exist_ok=True)

def pick_cols(cols):
    n = {c.lower().strip(): c for c in cols}
    def has(*keys):
        for k in keys:
            for kk in n:
                if k in kk:
                    return n[kk]
        return None
    supplier  = has("supplier")
    warehouse = has("location","destination","store","warehouse","dc")
    cost      = has("shipping cost","costs","cost","price","rate")
    capacity  = has("production volume","stock level","capacity")
    demand    = has("order quant","number of products sold","demand","sales")
    missing = [(k,v) for k,v in [("supplier",supplier),("warehouse",warehouse),
                                 ("cost",cost),("capacity",capacity),("demand",demand)] if v is None]
    if missing:
        raise KeyError(f"Could not detect: {missing}. Available: {list(cols)}")
    return supplier, warehouse, cost, capacity, demand

def main():
    df = pd.read_csv(DATA_FILE)
    sup, wh, cost, cap, dem = pick_cols(df.columns)
    print(f"[optimisation] Using -> supplier:'{sup}', warehouse:'{wh}', cost:'{cost}', capacity:'{cap}', demand:'{dem}'")

    # Keep only needed cols; drop rows missing any of them
    df = df[[sup, wh, cost, cap, dem]].dropna()

    # Unit cost per (supplier, warehouse): mean if multiple rows
    cmat_df = (df.groupby([sup, wh])[cost].mean().reset_index())
    cmat = {(r[sup], r[wh]): float(r[cost]) for _, r in cmat_df.iterrows()}

    # Supplier capacity: sum (or max) across rows
    capd = df.groupby(sup)[cap].sum().to_dict()

    # Warehouse demand: sum across rows
    need = df.groupby(wh)[dem].sum().to_dict()

    # Build model
    prob = LpProblem("MinCostShipping", LpMinimize)
    x = {(s,w): LpVariable(f"x_{s}_{w}", lowBound=0) for (s,w) in cmat}
    prob += lpSum(cmat[s,w] * x[s,w] for (s,w) in cmat)
    for s in capd:
        prob += lpSum(x[s,w] for w in need if (s,w) in x) <= capd[s]
    for w in need:
        prob += lpSum(x[s,w] for s in capd if (s,w) in x) >= need[w]

    prob.solve()
    status = LpStatus[prob.status]
    alloc = []
    for (s,w), var in x.items():
        q = var.value()
        if q and q > 0:
            alloc.append({"supplier": s, "warehouse": w, "qty": q,
                          "unit_cost": cmat[(s,w)], "ship_cost": q*cmat[(s,w)]})
    out = pd.DataFrame(alloc).sort_values(["supplier","warehouse"])
    out.to_csv(OUTDIR/"shipment_plan.csv", index=False)
    with open(OUTDIR/"optimisation_summary.txt","w") as f:
        from pulp import value
        f.write(f"Status: {status}\nTotal cost: {value(prob.objective):,.2f}\n")
    print(f"Status: {status} | Saved outputs/shipment_plan.csv and outputs/optimisation_summary.txt")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.stderr.write(f"\nError: {e}\n"); sys.exit(1)
